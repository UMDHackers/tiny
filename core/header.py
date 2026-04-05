"""
Binary pack/unpack for the .tiny format.

Header layout (all multi-byte fields are little-endian):
  [4s]  magic        = b'TINY'
  [B]   version      = 1 / 2 / 3
  [H]   entry_count  (uint16) — images + audio files combined

VERSION 1/2 — image-only (no modality byte):
  Per image (repeated entry_count times):
    [H]   name_len   (uint16)
    [*]   name       (UTF-8 bytes, name_len long)
    [H]   orig_w     original pixel width
    [H]   orig_h     original pixel height
    [4I]  lat_shape  latent dims (N, C, H, W) as uint32 each
    [f]   scale      float32 TurboQuant scale (0.0 = per-channel format)
    [I]   data_len   uint32 length of payload
    [*]   data       payload
    VERSION 2 only:
    [I]   res_len    residual data length (0 = no residual)
    [*]   res_data

VERSION 3 — mixed modality (per-entry modality byte):
  Per entry (repeated entry_count times):
    [B]   modality   0 = image, 1 = audio

    modality 0 (image) — same as VERSION 2 image entry above
    modality 1 (audio):
      [H]   name_len
      [*]   name
      [I]   orig_sr      original sample rate
      [f]   bandwidth    kbps used
      [I]   num_samples  sample count at orig_sr
      [B]   num_channels 1 or 2
      [I]   data_len
      [*]   data         AudioEngine payload
"""

import io
import math
import os
import struct

import numpy as np
import torch
import torch.nn.functional as F
import zstandard as zstd
from PIL import Image
from torchvision import transforms

from compressai.zoo import mbt2018_mean, cheng2020_anchor
from core.audio_engine import AudioEngine
from core.quantizer import TurboQuantizer
from core.residual import ResidualEngine
from core.turbo_math import TurboMath
from core.video_engine import VideoEngine

MAGIC = b"TINY"
VERSION = 8
ZSTD_LEVEL = 19  # Ultra compression

MODALITY_IMAGE = 0
MODALITY_AUDIO = 1
MODALITY_VIDEO = 2
MODALITY_AV = 3
MODALITY_SEMANTIC_AUDIO = 4  # Semantic (ASR→TTS) speech compression

# Codec IDs for v8 format
CODEC_CHENG2020 = 0
CODEC_MBT2018_LEGACY = 5

AUDIO_EXTS = {".mp3", ".wav", ".flac", ".ogg", ".aac", ".m4a"}
VIDEO_EXTS = {".mp4", ".mov", ".avi", ".mkv", ".webm"}

_MODEL_CACHE: dict = {}

# mbt2018_mean latent channels (M) per quality level (legacy)
_LATENT_C_TO_MIN_QUALITY = {128: 1, 192: 3, 320: 5}


def _get_model(quality: int = 1) -> torch.nn.Module:
    """Get legacy mbt2018_mean model (used for v7 decode and video)."""
    if quality not in _MODEL_CACHE:
        model = mbt2018_mean(quality=quality, pretrained=True)
        model.eval()
        _MODEL_CACHE[quality] = model
    return _MODEL_CACHE[quality]


def _get_v8_model(quality: int = 1, codec_id: int = CODEC_CHENG2020) -> torch.nn.Module:
    """Get v8 native codec model. Default: cheng2020_anchor (better than mbt2018)."""
    cache_key = f"v8_{codec_id}_q{quality}"
    if cache_key not in _MODEL_CACHE:
        if codec_id == CODEC_CHENG2020:
            model = cheng2020_anchor(quality=quality, pretrained=True)
        else:
            model = mbt2018_mean(quality=quality, pretrained=True)
        model.eval()
        model.update()  # required for native compress/decompress
        _MODEL_CACHE[cache_key] = model
    return _MODEL_CACHE[cache_key]


def _model_for_latent_channels(C: int) -> torch.nn.Module:
    """Return the lowest-quality mbt2018_mean model that handles C latent channels (legacy)."""
    q = _LATENT_C_TO_MIN_QUALITY.get(C, 1)
    return _get_model(q)


MAX_DIM = 1024  # Cap longest edge to keep memory manageable


def _serialize_native_bitstream(compress_out: dict) -> bytes:
    """Serialize CompressAI model.compress() output to bytes for .tiny storage.

    Format:
      [H] num_string_groups (typically 2: y_strings, z_strings)
      [H H] shape (latent spatial dims, e.g. 4x4 for 256x256 input)
      For each string group:
        [I] num_strings (batch size, typically 1)
        For each string:
          [I] string_len
          [*] string_data
    """
    buf = io.BytesIO()
    strings = compress_out["strings"]  # list of list of bytes
    shape = compress_out["shape"]  # torch.Size([H, W])

    buf.write(struct.pack("<H", len(strings)))
    buf.write(struct.pack("<HH", shape[0], shape[1]))

    for group in strings:
        buf.write(struct.pack("<I", len(group)))
        for s in group:
            buf.write(struct.pack("<I", len(s)))
            buf.write(s)

    return buf.getvalue()


def _deserialize_native_bitstream(data: bytes) -> tuple[list, torch.Size]:
    """Deserialize native bitstream back to (strings, shape) for model.decompress()."""
    buf = io.BytesIO(data)

    (num_groups,) = struct.unpack("<H", buf.read(2))
    sh, sw = struct.unpack("<HH", buf.read(4))
    shape = torch.Size([sh, sw])

    strings = []
    for _ in range(num_groups):
        (num_strings,) = struct.unpack("<I", buf.read(4))
        group = []
        for _ in range(num_strings):
            (slen,) = struct.unpack("<I", buf.read(4))
            group.append(buf.read(slen))
        strings.append(group)

    return strings, shape


def _image_to_tensor(path: str, max_dim: int = MAX_DIM) -> tuple[torch.Tensor, int, int]:
    """
    Load image, proportionally resize to fit max_dim, then pad (not truncate)
    to the next 64-multiples. Returns (padded_tensor, true_w, true_h).
    Aspect ratio is perfectly preserved.
    """
    img = Image.open(path).convert("RGB")
    w, h = img.size

    if max(w, h) > max_dim:
        scale = max_dim / max(w, h)
        new_w = max(1, round(w * scale))
        new_h = max(1, round(h * scale))
        img = img.resize((new_w, new_h), Image.LANCZOS)
        w, h = new_w, new_h

    pad_w = ((w + 63) // 64) * 64
    pad_h = ((h + 63) // 64) * 64

    tensor = transforms.ToTensor()(img).unsqueeze(0)  # (1, 3, H, W)
    if pad_w != w or pad_h != h:
        tensor = F.pad(tensor, (0, pad_w - w, 0, pad_h - h), mode="reflect")

    return tensor, w, h


def _tensor_to_image(tensor: torch.Tensor) -> Image.Image:
    arr = tensor.squeeze(0).clamp(0, 1).permute(1, 2, 0).detach().cpu().numpy()
    arr = (arr * 255).astype("uint8")
    return Image.fromarray(arr)


def _psnr(original: torch.Tensor, reconstructed: torch.Tensor) -> float:
    mse = torch.mean((original.clamp(0, 1) - reconstructed.clamp(0, 1)) ** 2).item()
    if mse == 0:
        return float("inf")
    return -10.0 * math.log10(mse)


_SHARPEN_KERNEL = torch.tensor(
    [[ 0, -1,  0],
     [-1,  5, -1],
     [ 0, -1,  0]], dtype=torch.float32
).view(1, 1, 3, 3)


def _sharpen(tensor: torch.Tensor, strength: float = 0.3) -> torch.Tensor:
    """Per-channel unsharp-mask sharpening to counteract VAE smoothing."""
    kernel = _SHARPEN_KERNEL.to(tensor.device)
    channels = []
    for c in range(tensor.shape[1]):
        ch = tensor[:, c:c+1, :, :]
        sharpened = F.conv2d(ch, kernel, padding=1)
        channels.append(ch + strength * (sharpened - ch))
    return torch.cat(channels, dim=1).clamp(0, 1)


def _has_audio_stream(video_path: str) -> bool:
    """Return True if the video file contains an audio stream (uses PyAV)."""
    try:
        import av as _av
        container = _av.open(video_path)
        has = any(s.type == "audio" for s in container.streams)
        container.close()
        return has
    except Exception:
        return False


def _extract_audio_to_wav(video_path: str, wav_path: str) -> None:
    """Extract audio stream from a video file to a 32-bit float WAV using PyAV + soundfile."""
    import av as _av
    import soundfile as _sf

    container = _av.open(video_path)
    audio_stream = next((s for s in container.streams if s.type == "audio"), None)
    if audio_stream is None:
        container.close()
        raise RuntimeError(f"No audio stream in {video_path!r}")

    sr = audio_stream.sample_rate
    frames = []
    for packet in container.demux(audio_stream):
        for frame in packet.decode():
            # to_ndarray() returns (channels, samples) for planar float formats
            arr = frame.to_ndarray()  # float32 planar (C, T)
            frames.append(arr.astype(np.float32))
    container.close()

    if not frames:
        raise RuntimeError(f"No audio frames decoded from {video_path!r}")

    audio = np.concatenate(frames, axis=1).T  # (total_samples, channels)
    _sf.write(wav_path, audio, sr, subtype="FLOAT")
    print(f"[av] Extracted audio: {audio.shape[0]/sr:.1f}s @ {sr}Hz, {audio.shape[1]}ch → {wav_path}")


def pack(
    paths: list[str],
    quality: int = 1,
    with_residual: bool = False,
    target_psnr: float | None = None,
    audio_bandwidth: float = 3.0,
    use_turbo: bool = False,
    norm_bits: int = 3,
    dir_bits: int = 3,
    residual_downsample: int = 2,
    lossless_bypass: bool = False,
    use_semantic: bool = False,
) -> bytes:
    """
    Compress a list of image and/or audio files into a .tiny binary blob.

    Parameters
    ----------
    paths               : list of file paths (images, audio, and/or video)
    quality             : codec quality level (1-6 for cheng2020_anchor)
    with_residual       : include residual correction layer for images (--fidelity)
    target_psnr         : include image residual only when AI-only PSNR < target
    audio_bandwidth     : EnCodec target kbps (1.5, 3.0, 6.0, 12.0, 24.0)
    use_turbo           : (legacy, ignored for images in v8) turbo pipeline for audio
    residual_downsample : downsample factor for residual compression

    Returns
    -------
    bytes — the complete .tiny binary blob (VERSION 8)
    """
    # Each entry: (modality, name, *modality-specific-fields)
    entries: list[tuple] = []

    with torch.no_grad():
        for path in paths:
            name = os.path.basename(path)
            ext = os.path.splitext(name)[1].lower()

            # ── AV entry (video with audio stream) ────────────────────────────
            if ext in VIDEO_EXTS:
                has_audio = _has_audio_stream(path)
                if has_audio:
                    import tempfile
                    base_name = os.path.splitext(name)[0]
                    tmp_wav = os.path.join(tempfile.gettempdir(), f"{base_name}_audio.wav")
                    print(f"[pack] {name}: detected audio stream — extracting to {tmp_wav} ...")
                    _extract_audio_to_wav(path, tmp_wav)

                    print(f"[pack] {name}: encoding video (AV mux) ...")
                    vid_payload, orig_w, orig_h, fps, total_frames = VideoEngine.encode(path, quality=quality, use_turbo=use_turbo)
                    print(f"[pack] {name}: video payload = {len(vid_payload):,} bytes "
                          f"({total_frames} frames @ {fps:.1f}fps, {orig_w}x{orig_h})")

                    print(f"[pack] {name}: encoding audio @ {audio_bandwidth} kbps ...")
                    aud_payload, orig_sr, num_samples, num_channels = AudioEngine.encode(
                        tmp_wav, bandwidth=audio_bandwidth, use_turbo=use_turbo,
                        lossless_bypass=lossless_bypass,
                    )
                    print(f"[pack] {name}: audio payload = {len(aud_payload):,} bytes "
                          f"({num_samples/orig_sr:.1f}s @ {orig_sr}Hz, {audio_bandwidth} kbps)")

                    try:
                        os.remove(tmp_wav)
                    except OSError:
                        pass

                    entries.append((
                        MODALITY_AV, name, orig_w, orig_h, fps, total_frames, vid_payload,
                        orig_sr, audio_bandwidth, num_samples, num_channels, aud_payload,
                    ))
                    continue

                # Video without audio
                print(f"[pack] {name}: encoding video ...")
                payload, orig_w, orig_h, fps, total_frames = VideoEngine.encode(path, quality=quality, use_turbo=use_turbo)
                print(f"[pack] {name}: video payload = {len(payload):,} bytes "
                      f"({total_frames} frames @ {fps:.1f}fps, {orig_w}x{orig_h})")
                entries.append((
                    MODALITY_VIDEO, name, orig_w, orig_h, fps, total_frames, payload,
                ))
                continue

            # ── Audio entry ───────────────────────────────────────────────────
            if ext in AUDIO_EXTS:
                if use_semantic:
                    # Semantic (ASR→TTS) path: extreme compression for speech
                    from core.semantic_audio import SemanticAudioEngine
                    payload, orig_sr, num_samples, num_channels = SemanticAudioEngine.encode(path)
                    print(f"[pack] {name}: semantic payload = {len(payload):,} bytes "
                          f"({num_samples/orig_sr:.1f}s @ {orig_sr}Hz)")
                    entries.append((
                        MODALITY_SEMANTIC_AUDIO, name,
                        orig_sr, num_samples, num_channels, payload,
                    ))
                else:
                    print(f"[pack] {name}: encoding audio @ {audio_bandwidth} kbps ...")
                    payload, orig_sr, num_samples, num_channels = AudioEngine.encode(
                        path, bandwidth=audio_bandwidth, use_turbo=use_turbo,
                        lossless_bypass=lossless_bypass,
                    )
                    print(f"[pack] {name}: audio payload = {len(payload):,} bytes "
                          f"({num_samples/orig_sr:.1f}s @ {orig_sr}Hz, {audio_bandwidth} kbps)")
                    entries.append((
                        MODALITY_AUDIO, name,
                        orig_sr, audio_bandwidth, num_samples, num_channels, payload,
                    ))
                continue

            # ── Image entry (v8: native codec path) ────────────────────────────
            tensor, true_w, true_h = _image_to_tensor(path)

            # v8: Use model's native compress() — no manual quantization
            v8_model = _get_v8_model(quality, codec_id=CODEC_CHENG2020)
            compress_out = v8_model.compress(tensor)
            native_bitstream = _serialize_native_bitstream(compress_out)

            # Optional: compute residual for fidelity mode
            residual_payload = b""
            if with_residual or target_psnr is not None:
                rec_out = v8_model.decompress(compress_out["strings"], compress_out["shape"])
                ai_base = rec_out["x_hat"].clamp(0, 1)
                ai_base_cropped = ai_base[:, :, :true_h, :true_w]
                ai_base_cropped = _sharpen(ai_base_cropped)
                orig_cropped = tensor[:, :, :true_h, :true_w]

                need_residual = with_residual
                if target_psnr is not None:
                    psnr_ai = _psnr(orig_cropped, ai_base_cropped)
                    print(f"[pack] {name}: AI-only PSNR = {psnr_ai:.2f} dB (target {target_psnr} dB)")
                    if psnr_ai < target_psnr:
                        need_residual = True

                if need_residual:
                    residual = ResidualEngine.compute(orig_cropped, ai_base_cropped)
                    residual_payload = ResidualEngine.compress(residual, downsample=residual_downsample)
                    print(f"[pack] {name}: residual payload = {len(residual_payload):,} bytes")

            print(f"[pack] {name}: v8 native bitstream = {len(native_bitstream):,} bytes "
                  f"(cheng2020 q{quality}, {true_w}x{true_h})")

            entries.append((
                MODALITY_IMAGE, name, true_w, true_h,
                CODEC_CHENG2020, quality, native_bitstream, residual_payload,
            ))

    # ── Build binary (VERSION 3) ──────────────────────────────────────────────
    buf = io.BytesIO()
    buf.write(struct.pack("<4sBH", MAGIC, VERSION, len(entries)))

    for entry in entries:
        modality = entry[0]
        buf.write(struct.pack("<B", modality))

        if modality == MODALITY_AV:
            _, name, orig_w, orig_h, fps, total_frames, vid_data, \
                orig_sr, bandwidth, num_samples, num_channels, aud_data = entry
            name_bytes = name.encode("utf-8")
            buf.write(struct.pack("<H", len(name_bytes)))
            buf.write(name_bytes)
            buf.write(struct.pack("<IIfI", orig_w, orig_h, fps, total_frames))
            buf.write(struct.pack("<I", len(vid_data)))
            buf.write(vid_data)
            buf.write(struct.pack("<IfIB", orig_sr, bandwidth, num_samples, num_channels))
            buf.write(struct.pack("<I", len(aud_data)))
            buf.write(aud_data)

        elif modality == MODALITY_VIDEO:
            _, name, orig_w, orig_h, fps, total_frames, data = entry
            name_bytes = name.encode("utf-8")
            buf.write(struct.pack("<H", len(name_bytes)))
            buf.write(name_bytes)
            buf.write(struct.pack("<IIfI", orig_w, orig_h, fps, total_frames))
            buf.write(struct.pack("<I", len(data)))
            buf.write(data)

        elif modality == MODALITY_AUDIO:
            _, name, orig_sr, bandwidth, num_samples, num_channels, data = entry
            name_bytes = name.encode("utf-8")
            buf.write(struct.pack("<H", len(name_bytes)))
            buf.write(name_bytes)
            buf.write(struct.pack("<IfIB", orig_sr, bandwidth, num_samples, num_channels))
            buf.write(struct.pack("<I", len(data)))
            buf.write(data)

        elif modality == MODALITY_SEMANTIC_AUDIO:
            # Semantic audio format:
            #   [H]  name_len
            #   [*]  name
            #   [I]  orig_sr        original sample rate
            #   [I]  num_samples    sample count at orig_sr
            #   [B]  num_channels   1 or 2
            #   [I]  data_len       length of semantic payload blob
            #   [*]  data           SemanticAudioEngine payload (SEMA magic)
            _, name, orig_sr, num_samples, num_channels, data = entry
            name_bytes = name.encode("utf-8")
            buf.write(struct.pack("<H", len(name_bytes)))
            buf.write(name_bytes)
            buf.write(struct.pack("<IIB", orig_sr, num_samples, num_channels))
            buf.write(struct.pack("<I", len(data)))
            buf.write(data)

        else:  # MODALITY_IMAGE (v8: native codec)
            _, name, orig_w, orig_h, codec_id, stored_quality, bitstream, res_data = entry
            name_bytes = name.encode("utf-8")
            buf.write(struct.pack("<H", len(name_bytes)))
            buf.write(name_bytes)
            buf.write(struct.pack("<HH", orig_w, orig_h))
            buf.write(struct.pack("<B", codec_id))
            buf.write(struct.pack("<B", stored_quality))
            buf.write(struct.pack("<I", len(bitstream)))
            buf.write(bitstream)
            # Residual layer (0 length = no residual)
            buf.write(struct.pack("<I", len(res_data)))
            if res_data:
                buf.write(res_data)

    return buf.getvalue()


def _decode_payload(payload: bytes, legacy_scale: float, C: int):
    """
    Decode a stored data payload, returning (scale_tensor, compressed_latent_bytes).

    New format (legacy_scale == 0.0):
      [uint32]      num_channels C
      [C * float32] per-channel scales
      [...]         zstd-compressed int8 latent

    Legacy format (legacy_scale != 0.0):
      [...]         zstd-compressed int8 latent (entire payload)
      scale_tensor is a scalar.
    """
    if legacy_scale == -1.0:
        # Lossless bypass: payload is zstd-compressed float32 latent (no scales)
        return None, payload
    elif legacy_scale == 0.0:
        # New per-channel format
        offset = 0
        (num_ch,) = struct.unpack_from("<I", payload, offset)
        offset += 4
        scales_np = np.frombuffer(payload[offset: offset + num_ch * 4], dtype="float32").copy()
        offset += num_ch * 4
        compressed = payload[offset:]
        scale_t = torch.from_numpy(scales_np)  # shape (C,)
    else:
        # Legacy global-scale format
        compressed = payload
        scale_t = torch.tensor(legacy_scale)
    return scale_t, compressed


def _read_image_entry(buf: io.BytesIO, version: int, model, dctx, output_dir: str) -> str:
    """Read one image entry from buf, decode, save, return output path."""
    (name_len,) = struct.unpack("<H", buf.read(2))
    name = buf.read(name_len).decode("utf-8")
    orig_w, orig_h = struct.unpack("<HH", buf.read(4))
    N, C, Hl, Wl = struct.unpack("<4I", buf.read(16))
    (scale,) = struct.unpack("<f", buf.read(4))
    # VERSION 7+: read stored quality byte for exact model selection
    if version >= 7:
        (stored_quality,) = struct.unpack("<B", buf.read(1))
        model = _get_model(stored_quality)
    else:
        # Fallback for older files: guess from latent channels
        model = _model_for_latent_channels(C)
    (data_len,) = struct.unpack("<I", buf.read(4))
    data = buf.read(data_len)

    # Read the standard residual field (always present in VERSION 2+)
    res_payload = b""
    if version >= 2:
        (res_len,) = struct.unpack("<I", buf.read(4))
        if res_len > 0:
            res_payload = buf.read(res_len)
        # For version < 5, only the standard residual is read here

    # VERSION 5: read turbo_flag byte (may be absent in older v5 files)
    turbo_flag = False
    if version >= 5:
        raw = buf.read(1)
        if len(raw) == 1:
            (turbo_byte,) = struct.unpack("<B", raw)
            turbo_flag = (turbo_byte == 1)

    if turbo_flag:
        # Read turbo fields
        (rotation_seed,) = struct.unpack("<I", buf.read(4))
        (polar_payload_len,) = struct.unpack("<I", buf.read(4))
        polar_payload = buf.read(polar_payload_len)
        (qjl_len,) = struct.unpack("<I", buf.read(4))
        qjl_bytes = buf.read(qjl_len)
        (turbo_res_len,) = struct.unpack("<I", buf.read(4))
        turbo_res_data = buf.read(turbo_res_len) if turbo_res_len > 0 else b""

        # Decode: PolarQuant → QJL correction → rotate_inverse → g_s
        lat_shape = (N, C, Hl, Wl)
        y_rot_hat = TurboMath.polar_quantize_decode(polar_payload, lat_shape)
        if len(qjl_bytes) > 0:
            correction = TurboMath.qjl_reconstruct(
                qjl_bytes, rotation_seed, n_projections=128, shape=lat_shape
            )
            y_rot_hat = y_rot_hat + correction
        latent = TurboMath.rotate_inverse(y_rot_hat, rotation_seed)
        reconstructed = model.g_s(latent).clamp(0, 1)
        reconstructed = reconstructed[:, :, :orig_h, :orig_w]
        # Adaptive sharpen: disable for high-quality models to prevent ringing
        _sharpen_str = 0.0 if C > 192 else 0.3
        reconstructed = _sharpen(reconstructed, strength=_sharpen_str)
        # Texture Injector: subtle seeded grain to prevent "waxy" AI smoothing artifact
        import numpy as _np
        _grain_rng = _np.random.RandomState(hash(name) & 0xFFFFFFFF)
        _grain = torch.from_numpy(_grain_rng.randn(*reconstructed.shape).astype("float32")) * 0.012
        reconstructed = (reconstructed + _grain).clamp(0, 1)

        # Apply turbo residual if present
        if len(turbo_res_data) > 0:
            turbo_residual = ResidualEngine.decompress(turbo_res_data, orig_h, orig_w)
            reconstructed = ResidualEngine.apply(reconstructed, turbo_residual)
    else:
        # Standard decode path
        scale_t, compressed = _decode_payload(data, scale, C)
        if scale == -1.0:
            # Lossless bypass: decompress raw float32 latents directly
            raw_bytes = dctx.decompress(compressed)
            latent_np = np.frombuffer(raw_bytes, dtype=np.float32).reshape(N, C, Hl, Wl)
            latent = torch.from_numpy(latent_np.copy())
        else:
            raw_bytes = dctx.decompress(compressed)
            quantized_np = np.frombuffer(raw_bytes, dtype=np.int8).reshape(N, C, Hl, Wl)
            quantized = torch.from_numpy(quantized_np.copy()).to(torch.int8)
            latent = TurboQuantizer.dequantize(quantized, scale_t)

        reconstructed = model.g_s(latent).clamp(0, 1)
        reconstructed = reconstructed[:, :, :orig_h, :orig_w]
        # Adaptive sharpen: disable for high-quality models to prevent ringing
        _sharpen_str = 0.0 if C > 192 else 0.3
        reconstructed = _sharpen(reconstructed, strength=_sharpen_str)
        # Texture Injector: subtle seeded grain to prevent "waxy" AI smoothing artifact
        import numpy as _np
        _grain_rng = _np.random.RandomState(hash(name) & 0xFFFFFFFF)
        _grain = torch.from_numpy(_grain_rng.randn(*reconstructed.shape).astype("float32")) * 0.012
        reconstructed = (reconstructed + _grain).clamp(0, 1)

        if len(res_payload) > 0:
            residual = ResidualEngine.decompress(res_payload, orig_h, orig_w)
            reconstructed = ResidualEngine.apply(reconstructed, residual)

    out_path = os.path.join(output_dir, name)
    _tensor_to_image(reconstructed).save(out_path)
    return out_path, name, reconstructed, orig_w, orig_h


def _read_image_entry_v8(buf: io.BytesIO, output_dir: str) -> tuple:
    """Read one v8 native-codec image entry, decode, save, return output path."""
    (name_len,) = struct.unpack("<H", buf.read(2))
    name = buf.read(name_len).decode("utf-8")
    orig_w, orig_h = struct.unpack("<HH", buf.read(4))
    (codec_id,) = struct.unpack("<B", buf.read(1))
    (stored_quality,) = struct.unpack("<B", buf.read(1))
    (bitstream_len,) = struct.unpack("<I", buf.read(4))
    bitstream = buf.read(bitstream_len)

    (res_len,) = struct.unpack("<I", buf.read(4))
    res_payload = buf.read(res_len) if res_len > 0 else b""

    # Native decompress
    model = _get_v8_model(stored_quality, codec_id=codec_id)
    strings, shape = _deserialize_native_bitstream(bitstream)
    with torch.no_grad():
        rec_out = model.decompress(strings, shape)
    reconstructed = rec_out["x_hat"].clamp(0, 1)
    reconstructed = reconstructed[:, :, :orig_h, :orig_w]
    reconstructed = _sharpen(reconstructed, strength=0.3)

    # Apply residual correction if present
    if len(res_payload) > 0:
        residual = ResidualEngine.decompress(res_payload, orig_h, orig_w)
        reconstructed = ResidualEngine.apply(reconstructed, residual)

    out_path = os.path.join(output_dir, name)
    _tensor_to_image(reconstructed).save(out_path)
    return out_path, name, reconstructed, orig_w, orig_h


def _mux_av(video_path: str, audio_path: str, out_path: str, crf: int = 23) -> None:
    """Mux a video-only .mp4 and a .wav into a final .mp4 using PyAV (no ffmpeg required)."""
    import av as _av
    import soundfile as _sf

    audio_np, sr = _sf.read(audio_path, always_2d=True)  # (T, C) float32
    audio_np = audio_np.astype(np.float32)
    # Resample to 44100 Hz for Apple compatibility (EnCodec outputs at 24000)
    target_sr = 44100
    if sr != target_sr:
        import torch as _torch
        import torchaudio.functional as _F_audio
        audio_t = _torch.from_numpy(audio_np.T)  # (C, T)
        audio_t = _F_audio.resample(audio_t, sr, target_sr)
        audio_np = audio_t.numpy().T  # (T, C)
        sr = target_sr
    num_channels = audio_np.shape[1]
    layout = "stereo" if num_channels == 2 else "mono"

    with _av.open(video_path) as src:
        in_video = src.streams.video[0]
        fps = float(in_video.average_rate)
        width, height = in_video.width, in_video.height

        with _av.open(out_path, "w") as dst:
            # Re-encode video (libx264, QuickTime-compatible H.264 High profile)
            ifps = int(fps)
            out_video = dst.add_stream("libx264", rate=ifps)
            out_video.width = width
            out_video.height = height
            out_video.pix_fmt = "yuv420p"
            out_video.options = {
                "crf": str(crf), "preset": "medium",
                "profile": "high", "level": "4.1",
                "movflags": "+faststart",
                "colorprim": "bt709", "transfer": "bt709", "colormatrix": "bt709",
            }

            # Add audio stream (AAC-LC at 44100 Hz for max Apple compatibility)
            out_audio = dst.add_stream("aac", rate=44100)

            # Encode video frames with explicit PTS for correct playback timing
            # IMPORTANT: Decoded frames carry source time_base metadata that
            # causes PTS rescaling bugs. Convert to numpy→fresh VideoFrame to
            # get clean frames with no inherited time_base.
            vid_frames = list(src.decode(video=0))
            for i, frame in enumerate(vid_frames):
                arr = frame.to_ndarray(format="yuv420p")
                fresh = _av.VideoFrame.from_ndarray(arr, format="yuv420p")
                fresh.pts = i
                for pkt in out_video.encode(fresh):
                    dst.mux(pkt)
            for pkt in out_video.encode(None):
                dst.mux(pkt)

            # Encode audio in 1024-sample chunks
            chunk = 1024
            T = audio_np.shape[0]
            t, pts = 0, 0
            while t < T:
                block = audio_np[t: t + chunk].T.astype(np.float32)  # (C, N)
                af = _av.AudioFrame.from_ndarray(block, format="fltp", layout=layout)
                af.sample_rate = sr
                af.pts = pts
                for pkt in out_audio.encode(af):
                    dst.mux(pkt)
                pts += block.shape[1]
                t += chunk
            for pkt in out_audio.encode(None):
                dst.mux(pkt)

    print(f"[av] Muxed → {out_path}")


def _read_av_entry(buf: io.BytesIO, output_dir: str,
                   quality: int = 1, crf: int = 23) -> tuple[str, str]:
    """Read one AV entry from buf, decode video+audio, mux with PyAV, return (out_path, name)."""
    import tempfile

    (name_len,) = struct.unpack("<H", buf.read(2))
    name = buf.read(name_len).decode("utf-8")
    orig_w, orig_h, fps, total_frames = struct.unpack("<IIfI", buf.read(16))
    (vid_data_len,) = struct.unpack("<I", buf.read(4))
    vid_data = buf.read(vid_data_len)

    orig_sr, bandwidth, num_samples, num_channels = struct.unpack("<IfIB", buf.read(13))
    (aud_data_len,) = struct.unpack("<I", buf.read(4))
    aud_data = buf.read(aud_data_len)

    base = os.path.splitext(name)[0]
    tmp_dir = tempfile.gettempdir()
    tmp_video = os.path.join(tmp_dir, f"{base}_tmp_video.mp4")
    tmp_audio = os.path.join(tmp_dir, f"{base}_tmp_audio.wav")

    # Decode video to a temp file
    VideoEngine.decode(vid_data, tmp_video, fps, quality=quality, crf=crf)

    # Decode audio to a temp file
    AudioEngine.decode(aud_data, tmp_audio, bandwidth=bandwidth)

    # Mux video + audio into final output using PyAV
    out_name = base + "_recovered.mp4"
    out_path = os.path.join(output_dir, out_name)
    _mux_av(tmp_video, tmp_audio, out_path, crf=crf)

    # Clean up temp files
    for tmp in (tmp_video, tmp_audio):
        try:
            os.remove(tmp)
        except OSError:
            pass

    duration = num_samples / orig_sr
    print(f"[unpack] {name}: AV decoded → {out_path}  "
          f"({total_frames} frames @ {fps:.1f}fps, {orig_w}x{orig_h}, "
          f"{duration:.1f}s audio @ {bandwidth} kbps)")
    return out_path, name


def _read_video_entry(buf: io.BytesIO, output_dir: str,
                      quality: int = 1, crf: int = 23) -> tuple[str, str]:
    """Read one video entry from buf, decode, save as .mp4, return (out_path, name)."""
    (name_len,) = struct.unpack("<H", buf.read(2))
    name = buf.read(name_len).decode("utf-8")
    orig_w, orig_h, fps, total_frames = struct.unpack("<IIfI", buf.read(16))
    (data_len,) = struct.unpack("<I", buf.read(4))
    data = buf.read(data_len)

    base = os.path.splitext(name)[0]
    out_name = base + "_recovered.mp4"
    out_path = os.path.join(output_dir, out_name)
    VideoEngine.decode(data, out_path, fps, quality=quality, crf=crf)
    print(f"[unpack] {name}: video decoded → {out_path}  "
          f"({total_frames} frames @ {fps:.1f}fps, {len(data):,} bytes)")
    return out_path, name


def _read_audio_entry(buf: io.BytesIO, output_dir: str,
                      enhance: bool = False) -> tuple[str, str, float]:
    """Read one audio entry from buf, decode, save as .wav, return (out_path, name, bandwidth)."""
    (name_len,) = struct.unpack("<H", buf.read(2))
    name = buf.read(name_len).decode("utf-8")
    orig_sr, bandwidth, num_samples, num_channels = struct.unpack("<IfIB", buf.read(13))
    (data_len,) = struct.unpack("<I", buf.read(4))
    data = buf.read(data_len)

    # Output as .wav regardless of original extension
    base = os.path.splitext(name)[0]
    out_name = base + ".wav"
    out_path = os.path.join(output_dir, out_name)
    AudioEngine.decode(data, out_path, bandwidth=bandwidth, enhance=enhance)
    duration = num_samples / orig_sr
    print(f"[unpack] {name}: audio decoded → {out_path}  ({duration:.1f}s, {bandwidth} kbps, {len(data):,} bytes)")
    return out_path, name, bandwidth


def _read_semantic_audio_entry(buf: io.BytesIO, output_dir: str) -> tuple[str, str]:
    """Read one semantic audio entry from buf, decode (resynthesize), save as .wav."""
    from core.semantic_audio import SemanticAudioEngine

    (name_len,) = struct.unpack("<H", buf.read(2))
    name = buf.read(name_len).decode("utf-8")
    orig_sr, num_samples, num_channels = struct.unpack("<IIB", buf.read(9))
    (data_len,) = struct.unpack("<I", buf.read(4))
    data = buf.read(data_len)

    base = os.path.splitext(name)[0]
    out_name = base + "_semantic.wav"
    out_path = os.path.join(output_dir, out_name)

    duration = num_samples / orig_sr
    print(f"[unpack] {name}: semantic audio ({duration:.1f}s, {len(data):,} bytes payload) → resynthesizing ...")
    SemanticAudioEngine.decode(data, out_path)
    print(f"[unpack] {name}: semantic audio decoded → {out_path}")
    return out_path, name


def unpack(blob: bytes, output_dir: str, quality: int = 1,
           video_crf: int = 23, enhance: bool = False) -> list[str]:
    """
    Decompress a .tiny blob and write recovered files to output_dir.

    Returns list of written file paths.
    """
    os.makedirs(output_dir, exist_ok=True)

    model = _get_model(quality)
    dctx = zstd.ZstdDecompressor()
    buf = io.BytesIO(blob)

    magic, version, entry_count = struct.unpack("<4sBH", buf.read(7))
    if magic != MAGIC:
        raise ValueError(f"Not a .tiny file (magic={magic!r})")
    if version not in (1, 2, 3, 4, 5, 6, 7, 8):
        raise ValueError(f"Unsupported version {version}")

    written: list[str] = []

    with torch.no_grad():
        for _ in range(entry_count):
            if version >= 3:
                (modality,) = struct.unpack("<B", buf.read(1))
            else:
                modality = MODALITY_IMAGE

            if modality == MODALITY_AV:
                out_path, name = _read_av_entry(buf, output_dir,
                                                quality=quality, crf=video_crf)
                written.append(out_path)
            elif modality == MODALITY_VIDEO:
                out_path, name = _read_video_entry(buf, output_dir,
                                                   quality=quality, crf=video_crf)
                written.append(out_path)
            elif modality == MODALITY_AUDIO:
                out_path, name, _ = _read_audio_entry(buf, output_dir, enhance=enhance)
                written.append(out_path)
            elif modality == MODALITY_SEMANTIC_AUDIO:
                out_path, name = _read_semantic_audio_entry(buf, output_dir)
                written.append(out_path)
            elif version >= 8:
                # v8 native codec path
                out_path, name, _, _, _ = _read_image_entry_v8(buf, output_dir)
                print(f"[unpack] {name}: image decoded → {out_path} (v8 native)")
                written.append(out_path)
            else:
                # Legacy v1-v7 path
                out_path, name, _, _, _ = _read_image_entry(buf, version, model, dctx, output_dir)
                print(f"[unpack] {name}: image decoded → {out_path}")
                written.append(out_path)

    return written


def unpack_with_originals(
    blob: bytes,
    original_paths: list[str],
    output_dir: str,
    quality: int = 1,
) -> list[tuple[str, float]]:
    """
    Decompress a .tiny blob and compare images to originals (prints PSNR).
    Audio entries are decoded without PSNR. Returns list of (output_path, metric).
    """
    os.makedirs(output_dir, exist_ok=True)

    model = _get_model(quality)
    dctx = zstd.ZstdDecompressor()
    buf = io.BytesIO(blob)

    magic, version, entry_count = struct.unpack("<4sBH", buf.read(7))
    if magic != MAGIC:
        raise ValueError(f"Not a .tiny file (magic={magic!r})")
    if version not in (1, 2, 3, 4, 5, 6, 7, 8):
        raise ValueError(f"Unsupported version {version}")

    results: list[tuple[str, float]] = []

    with torch.no_grad():
        for i in range(entry_count):
            if version >= 3:
                (modality,) = struct.unpack("<B", buf.read(1))
            else:
                modality = MODALITY_IMAGE

            if modality == MODALITY_AV:
                out_path, name = _read_av_entry(buf, output_dir)
                results.append((out_path, float("nan")))
                continue
            elif modality == MODALITY_VIDEO:
                out_path, name = _read_video_entry(buf, output_dir)
                results.append((out_path, float("nan")))
                continue
            elif modality == MODALITY_AUDIO:
                out_path, name, _ = _read_audio_entry(buf, output_dir)
                results.append((out_path, float("nan")))
                continue
            elif modality == MODALITY_SEMANTIC_AUDIO:
                out_path, name = _read_semantic_audio_entry(buf, output_dir)
                results.append((out_path, float("nan")))
                continue

            # Image entry
            if version >= 8:
                out_path, name, reconstructed, orig_w, orig_h = _read_image_entry_v8(
                    buf, output_dir
                )
            else:
                out_path, name, reconstructed, orig_w, orig_h = _read_image_entry(
                    buf, version, model, dctx, output_dir
                )

            orig_path = original_paths[i] if i < len(original_paths) else None
            if orig_path is None:
                print(f"[unpack] {name}: PSNR = N/A (original not found)")
                results.append((out_path, float("nan")))
                continue

            original, true_w, true_h = _image_to_tensor(orig_path)
            original = original[:, :, :true_h, :true_w]

            if reconstructed.shape[-2:] != original.shape[-2:]:
                reconstructed = torch.nn.functional.interpolate(
                    reconstructed,
                    size=original.shape[-2:],
                    mode="bilinear",
                    align_corners=False,
                )

            psnr_val = _psnr(original, reconstructed)

            if psnr_val > 30:
                verdict = "Great."
            elif psnr_val < 20:
                verdict = "TurboQuantizing is too aggressive."
            else:
                verdict = "Acceptable."

            print(f"[unpack] {name}: PSNR = {psnr_val:.2f} dB — {verdict}")
            results.append((out_path, psnr_val))

    return results
