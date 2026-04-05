"""
tiny — AI image + audio compressor CLI

Usage:
  tiny pack   <folder|file> [--quality N] [--audio-bandwidth BW] [--out FILE]
  tiny unpack <file>        [--quality N] [--originals FOLDER]
"""

import argparse
import os
import struct as _struct
import sys

from core.header import AUDIO_EXTS, VIDEO_EXTS, MODALITY_AV, pack, unpack, unpack_with_originals
from core.rate_controller import RateController
from core.registry import ModelRegistry

IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp", ".tiff"}
ALL_EXTS = IMAGE_EXTS | AUDIO_EXTS | VIDEO_EXTS

# Speech file detection: enforce minimum bandwidth to ensure voice recognition quality
_SPEECH_KEYWORDS = {"talk", "speech", "voice", "vocal", "speak", "chat", "convers",
                    "interview", "podcast", "narrat", "dialog", "lecture"}
_MIN_SPEECH_BANDWIDTH = 12.0  # kbps — below this, speech becomes unrecognizable

# Music file detection: force 24kbps to prevent "underwater synthesizer" effect
_MUSIC_KEYWORDS = {"song", "music", "instrumental", "beat", "melody", "rhythm",
                   "acoustic", "orchestra", "symphony", "jazz", "rock", "hip-hop",
                   "edm", "remix", "track", "album", "playlist", "studio"}
_MIN_MUSIC_BANDWIDTH = 24.0  # kbps — music needs full bandwidth for fidelity


def _is_speech_file(path: str) -> bool:
    name = os.path.basename(path).lower()
    return any(kw in name for kw in _SPEECH_KEYWORDS)


def _is_music_file(path: str) -> bool:
    name = os.path.basename(path).lower()
    return any(kw in name for kw in _MUSIC_KEYWORDS)


def _collect_files(folder: str) -> list[str]:
    paths = []
    for fname in sorted(os.listdir(folder)):
        ext = os.path.splitext(fname)[1].lower()
        if ext in ALL_EXTS:
            paths.append(os.path.join(folder, fname))
    if not paths:
        print(f"No supported files found in {folder!r}", file=sys.stderr)
        sys.exit(1)
    return paths


def cmd_pack(args: argparse.Namespace) -> None:
    target = args.folder.rstrip("/")
    out_file = args.out or (os.path.splitext(target)[0] + ".tiny")

    if os.path.isfile(target):
        ext = os.path.splitext(target)[1].lower()
        if ext not in ALL_EXTS:
            print(f"Unsupported file type: {ext!r}", file=sys.stderr)
            sys.exit(1)
        paths = [target]
    else:
        paths = _collect_files(target)

    img_count = sum(1 for p in paths if os.path.splitext(p)[1].lower() in IMAGE_EXTS)
    aud_count = len(paths) - img_count
    print(f"[pack] Compressing {img_count} image(s) + {aud_count} audio file(s) from {target!r} ...")

    # Validate --image-model / --audio-model if provided
    if getattr(args, "image_model", None):
        registry = ModelRegistry.load()
        if args.image_model not in registry:
            print(f"[pack] Unknown image model ID {args.image_model!r}. "
                  f"Known: {list(registry.keys())}", file=sys.stderr)
            sys.exit(1)
        print(f"[pack] Using image expert: {args.image_model}")

    if getattr(args, "audio_model", None):
        registry = ModelRegistry.load()
        if args.audio_model not in registry:
            print(f"[pack] Unknown audio model ID {args.audio_model!r}. "
                  f"Known: {list(registry.keys())}", file=sys.stderr)
            sys.exit(1)
        print(f"[pack] Using audio expert: {args.audio_model}")

    # Resolve pack parameters (rate control overrides individual flags)
    audio_bandwidth = args.audio_bandwidth
    quality = args.quality
    with_residual = args.residual
    use_turbo = args.turbo
    norm_bits = 3
    dir_bits = 3
    residual_downsample = 2
    lossless_bypass = False

    if getattr(args, "target_size", None) is not None:
        original_size = sum(os.path.getsize(p) for p in paths)
        params = RateController.get_params(args.target_size, audio_bandwidth)
        target_bytes = RateController.target_bytes(original_size, args.target_size)
        print(f"[pack] Rate control: {args.target_size}% target → {params.describe()}")
        quality = params.quality
        use_turbo = params.use_turbo
        norm_bits = params.norm_bits
        dir_bits = params.dir_bits
        with_residual = params.with_residual
        residual_downsample = params.residual_downsample
        audio_bandwidth = params.audio_bandwidth
        lossless_bypass = params.lossless_bypass
    else:
        target_bytes = None

    # Enforce minimum bandwidth for speech files
    speech_files = [p for p in paths
                    if os.path.splitext(p)[1].lower() in AUDIO_EXTS and _is_speech_file(p)]
    if speech_files and audio_bandwidth < _MIN_SPEECH_BANDWIDTH:
        audio_bandwidth = _MIN_SPEECH_BANDWIDTH
        print(f"[pack] Speech file(s) detected — enforcing minimum {_MIN_SPEECH_BANDWIDTH}kbps "
              f"(was {args.audio_bandwidth}kbps) for: {', '.join(os.path.basename(p) for p in speech_files)}")

    # Enforce 24kbps for music files to prevent "underwater synthesizer" effect
    music_files = [p for p in paths
                   if os.path.splitext(p)[1].lower() in AUDIO_EXTS and _is_music_file(p)]
    if music_files and audio_bandwidth < _MIN_MUSIC_BANDWIDTH:
        audio_bandwidth = _MIN_MUSIC_BANDWIDTH
        print(f"[pack] Music file(s) detected — enforcing {_MIN_MUSIC_BANDWIDTH}kbps "
              f"for: {', '.join(os.path.basename(p) for p in music_files)}")

    use_semantic = getattr(args, "semantic", False)
    if use_semantic:
        print("[pack] --semantic mode: speech will be transcribed and resynthesized at decode")
        print("[pack] WARNING: reconstructed voice will differ from original speaker")

    blob = pack(
        paths,
        quality=quality,
        with_residual=with_residual,
        target_psnr=args.target_psnr,
        audio_bandwidth=audio_bandwidth,
        use_turbo=use_turbo,
        norm_bits=norm_bits,
        dir_bits=dir_bits,
        residual_downsample=residual_downsample,
        lossless_bypass=lossless_bypass,
        use_semantic=use_semantic,
    )

    with open(out_file, "wb") as f:
        f.write(blob)

    original_size = sum(os.path.getsize(p) for p in paths)
    ratio = (1 - len(blob) / original_size) * 100 if original_size else 0
    print(f"[pack] Saved {out_file!r}  ({len(blob):,} bytes, {ratio:.1f}% smaller)")

    if target_bytes is not None:
        actual = len(blob)
        pct_of_target = actual / target_bytes * 100 if target_bytes else float("inf")
        print(f"[pack] Target: {target_bytes:,} bytes / Actual: {actual:,} bytes "
              f"({pct_of_target:.1f}% of target)")

    # Fidelity Score: for lossless tier, unpack and compare hashes
    if lossless_bypass:
        import hashlib
        import tempfile
        tmp_dir = tempfile.mkdtemp(prefix="tiny_fidelity_")
        recovered = unpack(blob, tmp_dir, quality=quality)
        for orig_path in paths:
            orig_name = os.path.basename(orig_path)
            ext = os.path.splitext(orig_name)[1].lower()
            # Find matching recovered file
            for rec_path in recovered:
                if os.path.splitext(os.path.basename(rec_path))[1].lower() == ext:
                    orig_hash = hashlib.sha256(open(orig_path, "rb").read()).hexdigest()[:12]
                    rec_hash = hashlib.sha256(open(rec_path, "rb").read()).hexdigest()[:12]
                    match = "EXACT MATCH" if orig_hash == rec_hash else "LOSSY (AI-reconstructed)"
                    print(f"[fidelity] {orig_name}: {match}  (orig={orig_hash} rec={rec_hash})")
                    break
        # Clean up
        import shutil
        shutil.rmtree(tmp_dir, ignore_errors=True)


def _parse_packed_names(blob: bytes) -> list[str]:
    """Parse .tiny header to extract packed entry names (for --originals matching)."""
    version = _struct.unpack_from("<B", blob, 4)[0]
    entry_count = _struct.unpack_from("<H", blob, 5)[0]
    names = []
    offset = 7
    for _ in range(entry_count):
        if version >= 3:
            modality = _struct.unpack_from("<B", blob, offset)[0]
            offset += 1
        else:
            modality = 0  # image

        name_len = _struct.unpack_from("<H", blob, offset)[0]
        offset += 2
        name = blob[offset: offset + name_len].decode()
        offset += name_len
        names.append(name)

        if modality == 3:  # AV (video + audio muxed)
            # video: orig_w(I) + orig_h(I) + fps(f) + total_frames(I) = 16 bytes
            offset += 16
            vid_data_len = _struct.unpack_from("<I", blob, offset)[0]
            offset += 4 + vid_data_len
            # audio: orig_sr(I) + bandwidth(f) + num_samples(I) + num_channels(B) = 13 bytes
            offset += 13
            aud_data_len = _struct.unpack_from("<I", blob, offset)[0]
            offset += 4 + aud_data_len
        elif modality == 2:  # video
            # orig_w(I) + orig_h(I) + fps(f) + total_frames(I) = 16 bytes
            offset += 16
            data_len = _struct.unpack_from("<I", blob, offset)[0]
            offset += 4 + data_len
        elif modality == 1:  # audio (EnCodec)
            # orig_sr(I) + bandwidth(f) + num_samples(I) + num_channels(B) = 13 bytes
            offset += 13
            data_len = _struct.unpack_from("<I", blob, offset)[0]
            offset += 4 + data_len
        elif modality == 4:  # semantic audio (ASR→TTS)
            # orig_sr(I) + num_samples(I) + num_channels(B) = 9 bytes
            offset += 9
            data_len = _struct.unpack_from("<I", blob, offset)[0]
            offset += 4 + data_len
        else:  # image
            if version >= 8:
                # v8 native: orig_w(H) + orig_h(H) + codec_id(B) + quality(B)
                offset += 4 + 1 + 1
                data_len = _struct.unpack_from("<I", blob, offset)[0]
                offset += 4 + data_len
                res_len = _struct.unpack_from("<I", blob, offset)[0]
                offset += 4 + res_len
            else:
                # Legacy v1-v7
                offset += 4  # orig_w + orig_h
                offset += 16  # lat_shape (4I)
                offset += 4   # scale (f)
                if version >= 7:
                    offset += 1  # quality byte
                data_len = _struct.unpack_from("<I", blob, offset)[0]
                offset += 4 + data_len
                if version >= 2:
                    res_len = _struct.unpack_from("<I", blob, offset)[0]
                    offset += 4 + res_len
                if version >= 5:
                    turbo_flag = _struct.unpack_from("<B", blob, offset)[0]
                    offset += 1
                    if turbo_flag:
                        offset += 4  # seed
                        polar_len = _struct.unpack_from("<I", blob, offset)[0]
                        offset += 4 + polar_len
                        qjl_len = _struct.unpack_from("<I", blob, offset)[0]
                        offset += 4 + qjl_len
                        turbo_res_len = _struct.unpack_from("<I", blob, offset)[0]
                        offset += 4 + turbo_res_len

    return names


def cmd_models() -> None:
    models = ModelRegistry.list_models()
    print(f"{'ID':<25} {'TYPE':<12} {'CACHED':<8} DESCRIPTION")
    print("-" * 80)
    for m in models:
        cached_str = "yes" if m["cached"] else "no"
        print(f"{m['id']:<25} {m['type']:<12} {cached_str:<8} {m['description']}")


def cmd_unpack(args: argparse.Namespace) -> None:
    with open(args.file, "rb") as f:
        blob = f.read()

    base = os.path.splitext(os.path.basename(args.file))[0]
    output_dir = os.path.join(os.getcwd(), base + "_recovered")

    originals_folder = args.originals
    enhance = getattr(args, "enhance", False)
    if originals_folder:
        packed_names = _parse_packed_names(blob)
        avail_images = {os.path.basename(p): p for p in _collect_files(originals_folder)
                        if os.path.splitext(p)[1].lower() in IMAGE_EXTS}
        original_paths = []
        for name in packed_names:
            if name in avail_images:
                original_paths.append(avail_images[name])
            else:
                original_paths.append(None)

        print(f"[unpack] Decompressing {args.file!r} (with PSNR vs {originals_folder!r}) ...")
        results = unpack_with_originals(blob, original_paths, output_dir, quality=args.quality)
        print(f"[unpack] Recovered {len(results)} file(s) → {output_dir!r}")
    else:
        print(f"[unpack] Decompressing {args.file!r} ...")
        written = unpack(blob, output_dir, quality=args.quality,
                         video_crf=args.video_crf, enhance=enhance)
        print(f"[unpack] Recovered {len(written)} file(s) → {output_dir!r}")


def main() -> None:
    parser = argparse.ArgumentParser(prog="tiny", description="AI image + audio compressor (.tiny format)")
    sub = parser.add_subparsers(dest="command", required=True)

    # tiny pack
    p_pack = sub.add_parser("pack", help="Compress images and/or audio into a .tiny file")
    p_pack.add_argument("folder", help="Source file or folder")
    p_pack.add_argument("--quality", type=int, default=1, choices=range(1, 7),
                        metavar="1-6", help="Codec quality level (1-6, default: 1). Higher = better quality, larger file")
    p_pack.add_argument("--audio-bandwidth", type=float, default=3.0,
                        metavar="BW", help="EnCodec audio bandwidth kbps: 1.5, 3, 6, 12, 24 (default: 3.0)")
    p_pack.add_argument("--out", help="Output .tiny file path")
    p_pack.add_argument("--residual", action="store_true",
                        help="Always include residual correction layer for images")
    p_pack.add_argument("--target-psnr", type=float, default=None, metavar="DB",
                        help="Include image residual only when AI-only PSNR is below this target")
    p_pack.add_argument("--turbo", action="store_true",
                        help="Use Turbo-Vision pipeline (polar quantization + QJL correction) for images")
    p_pack.add_argument("--target-size", type=float, default=None, metavar="PERCENT",
                        help="Target output size as %% of original (overrides quality/turbo/residual)")
    p_pack.add_argument("--image-model", default=None, metavar="ID",
                        help="Image expert ID from the model registry (e.g. img_mbt2018_q3)")
    p_pack.add_argument("--audio-model", default=None, metavar="ID",
                        help="Audio expert ID from the model registry (e.g. aud_encodec_24k)")
    p_pack.add_argument("--semantic", action="store_true",
                        help="[EXPERIMENTAL] Compress speech semantically: "
                             "transcribe with Whisper + store transcript/embedding + resynthesize at decode. "
                             "WARNING: voice is RESYNTHESIZED, not reconstructed. ~99.97%% smaller.")

    # tiny models
    sub.add_parser("models", help="List all registered model experts")

    # tiny unpack
    p_unpack = sub.add_parser("unpack", help="Decompress a .tiny file")
    p_unpack.add_argument("file", help=".tiny file to decompress")
    p_unpack.add_argument("--quality", type=int, default=1, choices=range(1, 7),
                          metavar="1-6", help="Quality level for legacy v7 files (v8 auto-detects, default: 1)")
    p_unpack.add_argument("--video-crf", type=int, default=23, metavar="CRF",
                          help="H.264 CRF for decompressed video (0=lossless, 51=worst, default: 23)")
    p_unpack.add_argument("--originals", metavar="FOLDER",
                          help="Original folder for image PSNR measurement")
    p_unpack.add_argument("--enhance", action="store_true",
                          help="Apply neural post-filter to decoded audio: "
                               "spectral Wiener denoising + treble EQ + contrast enhancement. "
                               "Restores sibilants and clarity lost at low bitrates. No file size change.")

    args = parser.parse_args()

    if args.command == "pack":
        cmd_pack(args)
    elif args.command == "unpack":
        cmd_unpack(args)
    elif args.command == "models":
        cmd_models()


if __name__ == "__main__":
    main()
