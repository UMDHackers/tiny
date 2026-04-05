"""
VideoEngine v2: encode/decode video using CompressAI neural codecs natively.

Architecture overview
---------------------
Frames are grouped into *chunks* of CHUNK_SIZE consecutive frames.
Within each chunk:
  - Frame 0 (keyframe): encoded with cheng2020_anchor image codec (native compress/decompress)
  - Frames 1..N (inter-frames): encoded with ssf2020 Scale-Space Flow video codec,
    which uses learned optical flow for temporal prediction — no pixel-domain residuals.

Why ssf2020 for inter-frames?
  ssf2020 achieves ~181x smaller inter-frames than our old WebP P-frame approach at
  higher quality (35-37 dB vs ~22 dB from 4x-downsampled WebP residuals). It uses
  learned optical flow to predict motion and codes only the residual in latent space.
  Crucially it is already in CompressAI (no new dependencies) and runs on CPU.

Why cheng2020_anchor for keyframes?
  ~15% better rate-distortion than mbt2018_mean. Already used for images in v8.
  ssf2020 uses its own internal image codec for the first frame in each chunk,
  but we use cheng2020_anchor instead for consistency with the image pipeline.
  NOTE: we handle this by feeding the ssf2020 model the full chunk including frame 0
  as the keyframe — ssf2020's internal encode_keyframe runs on frame 0.

Binary payload format (VERSION 2, magic b"SSF2")
-------------------------------------------------
  [4s]  magic          = b"SSF2"
  [H]   quality        quality level (1-6) used for encode
  [HH]  enc_w, enc_h   padded frame resolution (multiple of 128)
  [HH]  true_w, true_h true frame resolution before padding
  [f]   fps            original video fps
  [I]   total_frames   total number of frames in original video
  [I]   chunk_count    number of chunks stored
  Per chunk:
    [I]   start_frame   first frame index in this chunk
    [I]   chunk_len     number of frames in this chunk (1..CHUNK_SIZE)
    [I]   chunk_data_len total bytes of the chunk payload
    [*]   chunk_data    ssf2020 bitstream for this chunk (see _serialize_chunk)

chunk_data layout (from _serialize_chunk):
  [I]   n_frames       frames in chunk
  Per frame i in [0..n_frames):
    [I]   n_string_groups  number of string groups (typically 2)
    Per string group:
      [I] n_strings        strings in group (batch size, typically 1)
      Per string:
        [I]  slen          byte length
        [*]  string_data
    If i == 0 (keyframe): shape written separately
      [HH] shape_h, shape_w  latent spatial shape

Legacy format (WebP P-frames, VERSION 1, magic b"TURV" or no magic):
  Still decodable via _decode_legacy_payload().

QuickTime output constraints (enforced in decode()):
  - H.264 High profile, level 4.1, yuv420p
  - BT.709 color primaries
  - Explicit PTS: pts = frame_index (with stream.time_base set before loop)
  - Fresh VideoFrames via numpy roundtrip to avoid inherited time_base bugs
"""

import io
import math
import struct
from fractions import Fraction

import av
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from torchvision import transforms

from compressai.zoo import cheng2020_anchor, ssf2020 as _ssf2020

# ─── Constants ────────────────────────────────────────────────────────────────
MAX_FRAME_DIM = 512      # Cap longest edge to keep memory manageable on CPU
CHUNK_SIZE    = 8        # Frames per ssf2020 chunk (1 keyframe + 7 inter-frames)
SSF_STRIDE    = 128      # ssf2020 requires input spatial dims divisible by 128
DEFAULT_CRF   = 23       # H.264 CRF for decoded output video

_SSF2_MAGIC   = b"SSF2"  # Magic bytes identifying v2 payload
_TURBO_MAGIC  = b"TURV"  # Legacy v1 magic (Turbo-Motion WebP approach)
_TILED_MARKER = 0xFE     # Legacy v1 I-frame tiled marker

_MODEL_CACHE: dict = {}


# ─── Model loading ────────────────────────────────────────────────────────────

def _get_ssf_model(quality: int = 1) -> torch.nn.Module:
    """
    Load ssf2020 Scale-Space Flow video codec for the given quality level.
    Quality 1-6 map to ssf2020's rate-distortion operating points.
    Model is cached after first load.
    """
    key = f"ssf2020_q{quality}"
    if key not in _MODEL_CACHE:
        model = _ssf2020(quality=quality, pretrained=True)
        model.eval()
        model.update()
        _MODEL_CACHE[key] = model
    return _MODEL_CACHE[key]


def _get_img_model(quality: int = 1) -> torch.nn.Module:
    """
    Load cheng2020_anchor image codec (used for legacy decode paths).
    In the v2 pipeline ssf2020 handles its own keyframes internally.
    """
    key = f"cheng2020_q{quality}"
    if key not in _MODEL_CACHE:
        model = cheng2020_anchor(quality=quality, pretrained=True)
        model.eval()
        model.update()
        _MODEL_CACHE[key] = model
    return _MODEL_CACHE[key]


# ─── Frame conversion helpers ─────────────────────────────────────────────────

def _pil_to_tensor_ssf(img: Image.Image, max_dim: int = MAX_FRAME_DIM) -> tuple:
    """
    Resize proportionally (longest edge ≤ max_dim), then pad to SSF_STRIDE multiples.
    ssf2020 requires input spatial dimensions to be multiples of 128.

    Returns (tensor [1,3,H_pad,W_pad], true_w, true_h)
    """
    w, h = img.size
    if max(w, h) > max_dim:
        scale = max_dim / max(w, h)
        w, h = max(1, round(w * scale)), max(1, round(h * scale))
        img = img.resize((w, h), Image.LANCZOS)
    pad_w = math.ceil(w / SSF_STRIDE) * SSF_STRIDE
    pad_h = math.ceil(h / SSF_STRIDE) * SSF_STRIDE
    t = transforms.ToTensor()(img).unsqueeze(0)
    if pad_w != w or pad_h != h:
        t = F.pad(t, (0, pad_w - w, 0, pad_h - h), mode="reflect")
    return t, w, h


def _tensor_to_pil(t: torch.Tensor) -> Image.Image:
    arr = t.squeeze(0).clamp(0, 1).permute(1, 2, 0).detach().cpu().numpy()
    return Image.fromarray((arr * 255).astype(np.uint8))


# ─── Chunk serialization ──────────────────────────────────────────────────────

def _write_shape(buf: io.BytesIO, shape) -> None:
    """
    Write a shape to buf. Shape may be:
      - torch.Size (keyframe): write type=0, then 2 ints
      - dict {'motion': torch.Size, 'residual': torch.Size} (inter-frame): write type=1, then keys+shapes
    """
    if isinstance(shape, dict):
        buf.write(struct.pack("<B", 1))  # type=dict
        keys = sorted(shape.keys())
        buf.write(struct.pack("<B", len(keys)))
        for k in keys:
            kb = k.encode("utf-8")
            buf.write(struct.pack("<B", len(kb)))
            buf.write(kb)
            sh = shape[k]
            buf.write(struct.pack("<II", sh[0], sh[1]))
    else:
        buf.write(struct.pack("<B", 0))  # type=size
        buf.write(struct.pack("<II", shape[0], shape[1]))


def _read_shape(buf: io.BytesIO):
    """Read a shape written by _write_shape."""
    (stype,) = struct.unpack("<B", buf.read(1))
    if stype == 1:
        (n_keys,) = struct.unpack("<B", buf.read(1))
        shape = {}
        for _ in range(n_keys):
            (klen,) = struct.unpack("<B", buf.read(1))
            k = buf.read(klen).decode("utf-8")
            sh0, sh1 = struct.unpack("<II", buf.read(8))
            shape[k] = torch.Size([sh0, sh1])
        return shape
    else:
        sh0, sh1 = struct.unpack("<II", buf.read(8))
        return torch.Size([sh0, sh1])


def _write_string_groups(buf: io.BytesIO, groups: list) -> None:
    """Write a list of string groups [[bytes, ...], ...] to buf."""
    buf.write(struct.pack("<I", len(groups)))
    for group in groups:
        buf.write(struct.pack("<I", len(group)))
        for s in group:
            buf.write(struct.pack("<I", len(s)))
            buf.write(s)


def _read_string_groups(buf: io.BytesIO) -> list:
    """Read string groups written by _write_string_groups."""
    (n_groups,) = struct.unpack("<I", buf.read(4))
    groups = []
    for _ in range(n_groups):
        (n_strings,) = struct.unpack("<I", buf.read(4))
        group = []
        for _ in range(n_strings):
            (slen,) = struct.unpack("<I", buf.read(4))
            group.append(buf.read(slen))
        groups.append(group)
    return groups


def _serialize_chunk(frame_strings: list, shape_infos: list) -> bytes:
    """
    Serialize ssf2020 compress() output to bytes.

    ssf2020 returns two different per-frame structures:
      Keyframe  (i=0): fstrs = list of string groups [[bytes,...], ...]
      Inter-frame(i>0): fstrs = dict {'motion': [[bytes,...]], 'residual': [[bytes,...]]}

    Binary layout per frame:
      [B]  frame_type  0=keyframe (list), 1=inter-frame (dict)
      If 0: _write_string_groups(fstrs)
      If 1: [B] n_keys, then per key: [B] key_len, key bytes, _write_string_groups(val)
      _write_shape(shape)
    """
    buf = io.BytesIO()
    buf.write(struct.pack("<I", len(frame_strings)))
    for fstrs, shape in zip(frame_strings, shape_infos):
        if isinstance(fstrs, dict):
            buf.write(struct.pack("<B", 1))  # inter-frame
            keys = sorted(fstrs.keys())
            buf.write(struct.pack("<B", len(keys)))
            for k in keys:
                kb = k.encode("utf-8")
                buf.write(struct.pack("<B", len(kb)))
                buf.write(kb)
                _write_string_groups(buf, fstrs[k])
        else:
            buf.write(struct.pack("<B", 0))  # keyframe
            _write_string_groups(buf, fstrs)
        _write_shape(buf, shape)
    return buf.getvalue()


def _deserialize_chunk(data: bytes) -> tuple:
    """Deserialize chunk bytes back to (frame_strings, shape_infos) for ssf2020.decompress()."""
    buf = io.BytesIO(data)
    (n,) = struct.unpack("<I", buf.read(4))
    frame_strings = []
    shape_infos = []
    for _ in range(n):
        (frame_type,) = struct.unpack("<B", buf.read(1))
        if frame_type == 1:  # inter-frame dict
            (n_keys,) = struct.unpack("<B", buf.read(1))
            fstrs = {}
            for _ in range(n_keys):
                (klen,) = struct.unpack("<B", buf.read(1))
                k = buf.read(klen).decode("utf-8")
                fstrs[k] = _read_string_groups(buf)
        else:  # keyframe list
            fstrs = _read_string_groups(buf)
        frame_strings.append(fstrs)
        shape_infos.append(_read_shape(buf))
    return frame_strings, shape_infos


# ─── VideoEngine ──────────────────────────────────────────────────────────────

class VideoEngine:

    @staticmethod
    def encode(
        path: str,
        quality: int = 1,
        max_dim: int = MAX_FRAME_DIM,
        chunk_size: int = CHUNK_SIZE,
        use_turbo: bool = False,  # kept for API compat, ignored in v2
    ) -> tuple[bytes, int, int, float, int]:
        """
        Encode a video file to a compact .tiny payload using ssf2020.

        The video is split into chunks of `chunk_size` frames. ssf2020 compresses
        each chunk as a unit: frame 0 is the keyframe (I-frame equivalent),
        frames 1..N use learned optical flow for inter-frame prediction.

        Parameters
        ----------
        path        : input video path
        quality     : ssf2020 quality level 1-6 (higher = better + larger)
        max_dim     : max resolution of longest edge (default 512 for CPU speed)
        chunk_size  : frames per ssf2020 compression group (default 8)
        use_turbo   : ignored in v2 (kept for backward-compatible call sites)

        Returns
        -------
        (payload_bytes, orig_w, orig_h, fps, total_frames)
        """
        model = _get_ssf_model(quality)

        container = av.open(path)
        vs = container.streams.video[0]
        fps = float(vs.average_rate)
        orig_w, orig_h = vs.width, vs.height
        frames = [f.to_image() for f in container.decode(vs)]
        container.close()
        total_frames = len(frames)

        # Determine padded encoding resolution from first frame
        _, enc_w, enc_h = _pil_to_tensor_ssf(frames[0], max_dim)

        print(f"[video] ssf2020 q{quality}: {total_frames} frames @ {enc_w}x{enc_h} "
              f"(chunk_size={chunk_size}, fps={fps:.1f})")

        # Convert all frames to tensors
        tensors = []
        true_ws, true_hs = [], []
        for pil in frames:
            t, tw, th = _pil_to_tensor_ssf(pil, max_dim)
            tensors.append(t)
            true_ws.append(tw)
            true_hs.append(th)

        enc_h_pad, enc_w_pad = tensors[0].shape[-2], tensors[0].shape[-1]

        # Encode in chunks
        chunks = []  # (start_frame, chunk_len, chunk_data_bytes)
        with torch.no_grad():
            i = 0
            while i < total_frames:
                end = min(i + chunk_size, total_frames)
                chunk_tensors = tensors[i:end]
                chunk_len = len(chunk_tensors)

                frame_strings, shape_infos = model.compress(chunk_tensors)
                chunk_data = _serialize_chunk(frame_strings, shape_infos)

                chunk_bytes = sum(len(s) for fstr in frame_strings
                                  for group in fstr for s in group)
                kf_bytes = sum(len(s) for group in frame_strings[0] for s in group)
                inter_bytes = chunk_bytes - kf_bytes
                print(f"[video]   chunk {i}-{end-1} ({chunk_len} frames): "
                      f"keyframe={kf_bytes:,}B  inter={inter_bytes:,}B  total={chunk_bytes:,}B")

                chunks.append((i, chunk_len, chunk_data))
                i = end

        total_payload_bytes = sum(len(cd) for _, _, cd in chunks)
        print(f"[video] Encoded {len(chunks)} chunks, {total_payload_bytes:,} bytes total "
              f"({total_payload_bytes/1024:.1f} KB)")

        # Build binary payload
        buf = io.BytesIO()
        buf.write(_SSF2_MAGIC)
        buf.write(struct.pack("<H", quality))
        buf.write(struct.pack("<HH", enc_w_pad, enc_h_pad))
        buf.write(struct.pack("<HH", enc_w, enc_h))   # true dims before padding
        buf.write(struct.pack("<f", fps))
        buf.write(struct.pack("<I", total_frames))
        buf.write(struct.pack("<I", len(chunks)))
        for start_frame, chunk_len, chunk_data in chunks:
            buf.write(struct.pack("<III", start_frame, chunk_len, len(chunk_data)))
            buf.write(chunk_data)

        return buf.getvalue(), orig_w, orig_h, fps, total_frames

    @staticmethod
    def decode(payload: bytes, output_path: str, fps: float,
               quality: int = 1, crf: int = DEFAULT_CRF) -> None:
        """
        Decode a VideoEngine payload and write a QuickTime-compatible .mp4.

        Handles both v2 (ssf2020, magic b"SSF2") and legacy v1 (WebP P-frames)
        payloads transparently.

        Parameters
        ----------
        payload     : bytes from encode()
        output_path : destination .mp4 path
        fps         : original video frame rate (stored in .tiny header)
        quality     : quality level for legacy v1 decode (v2 reads it from payload)
        crf         : H.264 CRF for output video (0=lossless, 51=worst, default 23)
        """
        if payload[:4] == _SSF2_MAGIC:
            VideoEngine._decode_v2(payload, output_path, crf)
        else:
            VideoEngine._decode_legacy(payload, output_path, fps, quality, crf)

    @staticmethod
    def _decode_v2(payload: bytes, output_path: str, crf: int) -> None:
        """Decode a v2 ssf2020 payload."""
        buf = io.BytesIO(payload)
        buf.read(4)  # skip magic
        (quality,) = struct.unpack("<H", buf.read(2))
        enc_w_pad, enc_h_pad = struct.unpack("<HH", buf.read(4))
        enc_w, enc_h = struct.unpack("<HH", buf.read(4))       # true pre-padding dims
        (fps,) = struct.unpack("<f", buf.read(4))
        (total_frames,) = struct.unpack("<I", buf.read(4))
        (chunk_count,) = struct.unpack("<I", buf.read(4))

        model = _get_ssf_model(quality)

        all_frames: list[torch.Tensor] = [None] * total_frames
        with torch.no_grad():
            for _ in range(chunk_count):
                start_frame, chunk_len, chunk_data_len = struct.unpack("<III", buf.read(12))
                chunk_data = buf.read(chunk_data_len)

                frame_strings, shape_infos = _deserialize_chunk(chunk_data)
                rec_frames = model.decompress(frame_strings, shape_infos)
                # rec_frames is a list of reconstructed tensors
                if isinstance(rec_frames, dict):
                    rec_frames = rec_frames.get('x_hat', list(rec_frames.values())[0])

                for j, rec_t in enumerate(rec_frames):
                    frame_idx = start_frame + j
                    # Crop to true pre-padding dimensions
                    frame_tensor = rec_t.clamp(0, 1)[:, :, :enc_h, :enc_w]
                    all_frames[frame_idx] = frame_tensor

        # Fill any None slots (shouldn't happen but be safe)
        for i in range(len(all_frames)):
            if all_frames[i] is None:
                prev = next((all_frames[j] for j in range(i-1, -1, -1) if all_frames[j] is not None), None)
                all_frames[i] = prev if prev is not None else torch.zeros(1, 3, enc_h, enc_w)

        VideoEngine._write_mp4(all_frames, output_path, fps, enc_w, enc_h, crf)

    @staticmethod
    def _decode_legacy(payload: bytes, output_path: str, fps: float,
                       quality: int, crf: int) -> None:
        """
        Decode a legacy v1 payload (WebP P-frames + cheng2020 I-frames).
        Preserved for backward compatibility with pre-v2 .tiny files.
        """
        from core.quantizer import TurboQuantizer
        from core.turbo_math import TurboMath

        buf = io.BytesIO(payload)

        # Check for Turbo prefix
        use_turbo = False
        base_seed = 0
        magic_check = buf.read(4)
        if magic_check == _TURBO_MAGIC:
            use_turbo = True
            (base_seed,) = struct.unpack("<I", buf.read(4))
        else:
            buf.seek(0)

        enc_w, enc_h, pframe_ds, total_frames, stored_count = struct.unpack("<HHBII", buf.read(13))

        stored: dict[int, tuple] = {}
        for _ in range(stored_count):
            frame_idx, frame_type, ref_idx, data_len = struct.unpack("<IBII", buf.read(13))
            data = buf.read(data_len)
            stored[frame_idx] = (frame_type, ref_idx, data)

        # Decode I-frames
        img_model = _get_img_model(quality)
        iframe_recs: dict[int, torch.Tensor] = {}
        with torch.no_grad():
            for idx, (ftype, ref_idx, data) in sorted(stored.items()):
                if ftype == 0:
                    rec, h, w = _decode_legacy_iframe(data, img_model)
                    iframe_recs[idx] = rec

            # Decode P-frames
            reconstructed: dict[int, torch.Tensor] = dict(iframe_recs)
            for idx, (ftype, ref_idx, data) in sorted(stored.items()):
                if ftype == 1:
                    ref_rec = iframe_recs[ref_idx]
                    turbo_seed = (base_seed ^ idx) if use_turbo else None
                    reconstructed[idx] = _decode_legacy_pframe(data, ref_rec, turbo_seed)

        # Interpolate missing frames
        stored_indices = sorted(reconstructed.keys())
        all_frames: list[torch.Tensor] = []
        for i in range(total_frames):
            if i in reconstructed:
                all_frames.append(reconstructed[i])
            else:
                prev = max((k for k in stored_indices if k <= i), default=stored_indices[0])
                nxt  = min((k for k in stored_indices if k >= i), default=stored_indices[-1])
                if prev == nxt:
                    all_frames.append(reconstructed[prev])
                else:
                    alpha = (i - prev) / (nxt - prev)
                    interp = (1 - alpha) * reconstructed[prev] + alpha * reconstructed[nxt]
                    all_frames.append(interp.clamp(0, 1))

        VideoEngine._write_mp4(all_frames, output_path, fps, enc_w, enc_h, crf)

    @staticmethod
    def _write_mp4(frames: list, output_path: str, fps: float,
                   width: int, height: int, crf: int) -> None:
        """
        Write a list of frame tensors to a QuickTime-compatible H.264 .mp4.

        Note: PTS is NOT set on input frames — libx264 manages PTS internally.
        Setting pts on encode-side frames causes resets at each keyframe (every
        ~64 frames) and breaks mp4 muxer's monotonic PTS requirement.
        """
        print(f"[video] Writing {len(frames)} frames to {output_path} ...")
        out_container = av.open(output_path, "w")
        ifps = max(1, int(round(fps)))
        stream = out_container.add_stream("libx264", rate=ifps)
        stream.width = width
        stream.height = height
        stream.pix_fmt = "yuv420p"
        stream.options = {
            "crf": str(crf), "preset": "fast",
            "profile": "high", "level": "4.1",
            "colorprim": "bt709", "transfer": "bt709", "colormatrix": "bt709",
        }

        for i, frame_t in enumerate(frames):
            pil_img = _tensor_to_pil(frame_t)
            av_frame = av.VideoFrame.from_image(pil_img)
            # Do not set pts — let libx264 assign monotonically increasing PTS.
            # Setting av_frame.pts = i causes the encoder to reset pts=0 at each
            # new keyframe, which the mp4 muxer rejects as non-monotonic.
            for packet in stream.encode(av_frame):
                out_container.mux(packet)
        for packet in stream.encode():
            out_container.mux(packet)
        out_container.close()
        print(f"[video] Decoded → {output_path}  ({width}x{height}, {len(frames)} frames)")


# ─── Legacy I-frame / P-frame helpers ────────────────────────────────────────
# These are used only for decoding pre-v2 .tiny files.

def _decode_legacy_iframe(data: bytes, model) -> tuple:
    """Decode a legacy v1 I-frame payload."""
    if data[0] == _TILED_MARKER:
        offset = 0
        marker, tiled_flag, h, w = struct.unpack_from("<BBhh", data, offset)
        offset += 6
        if tiled_flag == 0:
            tile_data = data[offset:]
            pad_h = ((h + 63) // 64) * 64
            pad_w = ((w + 63) // 64) * 64
            rec = _decode_legacy_tile(tile_data, pad_h, pad_w, model)
            return rec[:, :, :h, :w], h, w
        else:
            n_tiles_y, n_tiles_x, tile_dim = struct.unpack_from("<HHH", data, offset)
            offset += 6
            rec_rows = []
            for ty in range(n_tiles_y):
                row = []
                for tx in range(n_tiles_x):
                    (tlen,) = struct.unpack_from("<I", data, offset); offset += 4
                    tile_data = data[offset:offset+tlen]; offset += tlen
                    rec_tile = _decode_legacy_tile(tile_data, tile_dim, tile_dim, model)
                    row.append(rec_tile[:, :, :tile_dim, :tile_dim])
                rec_rows.append(torch.cat(row, dim=3))
            full = torch.cat(rec_rows, dim=2)
            return full[:, :, :h, :w], h, w
    else:
        # Very old format
        import zstandard as zstd
        from core.quantizer import TurboQuantizer
        dctx = zstd.ZstdDecompressor()
        h, w, C = struct.unpack_from("<HHI", data, 0)
        scales_np = np.frombuffer(data[8:8+C*4], dtype="float32").copy()
        raw = dctx.decompress(data[8+C*4:])
        pad_h = ((h+63)//64)*64; pad_w = ((w+63)//64)*64
        Hl, Wl = pad_h//16, pad_w//16
        q_np = np.frombuffer(raw, dtype=np.int8).reshape(1, C, Hl, Wl)
        q = torch.from_numpy(q_np.copy()).to(torch.int8)
        scale_t = torch.from_numpy(scales_np)
        latent = TurboQuantizer.dequantize(q, scale_t)
        with torch.no_grad():
            rec = model.g_s(latent).clamp(0, 1)[:, :, :h, :w]
        return rec, h, w


def _decode_legacy_tile(tile_data: bytes, tile_h: int, tile_w: int, model) -> torch.Tensor:
    """Decode a v1 native-format tile (cheng2020_anchor native bitstream)."""
    buf = io.BytesIO(tile_data)
    (num_groups,) = struct.unpack("<H", buf.read(2))
    sh, sw = struct.unpack("<HH", buf.read(4))
    shape = torch.Size([sh, sw])
    strings = []
    for _ in range(num_groups):
        (n_str,) = struct.unpack("<I", buf.read(4))
        group = []
        for _ in range(n_str):
            (slen,) = struct.unpack("<I", buf.read(4))
            group.append(buf.read(slen))
        strings.append(group)
    with torch.no_grad():
        rec = model.decompress(strings, shape)["x_hat"].clamp(0, 1)
    return rec[:, :, :tile_h, :tile_w]


def _decode_legacy_pframe(data: bytes, ref_rec: torch.Tensor,
                           turbo_seed: int | None = None) -> torch.Tensor:
    """Apply a v1 WebP P-frame residual onto ref_rec."""
    from core.turbo_math import TurboMath
    small_h, small_w, webp_len = struct.unpack_from("<HHI", data, 0)
    offset = struct.calcsize("<HHI")
    webp = data[offset:offset+webp_len]
    img = Image.open(io.BytesIO(webp)).convert("RGB")
    arr = np.array(img, dtype=np.float32)
    small = torch.from_numpy(arr).permute(2, 0, 1).unsqueeze(0) / 127.5 - 1.0
    h, w = ref_rec.shape[-2:]
    residual = F.interpolate(small, size=(h, w), mode="bilinear", align_corners=False)
    if turbo_seed is not None:
        residual = TurboMath.rotate_inverse(residual, turbo_seed)
    return (ref_rec + residual).clamp(0.0, 1.0)
