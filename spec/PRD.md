# .tiny Engine — Product Requirements Document

## v1 MVP (shipped)

**Goal:** Compress high-res image folders by ~90% using a 3-layer AI pipeline.

### Core Pipeline
1. **VAE Layer:** `CompressAI` pre-trained models (`bmshj2018-factorized`) extract latent features.
2. **TurboQuant Layer:** `symmetric_4bit_quantization` on latent tensors.
3. **Zstd Layer:** `python-zstd` (Ultra-19 level) wraps the quantized bitstream.

### v1 Features
- `tiny pack <folder>`: Scans folder, converts images to `.tiny` blob.
- `tiny unpack <file>`: Reconstructs images to a `_recovered` folder.
- Custom Header: `[TINY][VER][IMG_COUNT][LATENT_SHAPE][DATA]`.

### Hybrid Residual Layer
- [x] During `pack`: Calculate `Residual = Original - AI_Reconstruction`.
- [x] Downsample the Residual by 2x.
- [x] Compress Residual using WebP (q=80 via PIL).
- [x] During `unpack`: Upsample Residual and add it back: `Final = AI_Base + Residual`.

### Audio Expert (EnCodec 24kHz)
- [x] `core/audio_engine.py` using Meta's EnCodec.
- [x] Support target bitrates of 1.5, 3, 6, 12, and 24 kbps.
- [x] Auto-detect `.mp3` and `.wav` in the source folder.
- [x] Binary integration: `modality_type` flag (1 for Audio) in container.

---

## v2 Rewrite — Feature Status

v2 targets best-in-class neural codecs for all modalities, a unified .tiny v8
container, and an improved CLI. See `spec/v2_spec.md` for sprint details and
`spec/format.md` for binary format documentation.

### Image Compression

| Feature | Status | Notes |
|---------|--------|-------|
| cheng2020_anchor native compress/decompress | **SHIPPED** | Replaces mbt2018 + manual quantization |
| .tiny v8 native bitstream storage | **SHIPPED** | codec_id + quality byte per entry |
| Proportional resize to MAX_DIM=1024, pad to 64-multiple | **SHIPPED** | No forced aspect ratio distortion |
| Residual layer (--residual flag) | **SHIPPED** | WebP-compressed 2x-downsampled residual |
| Conditional residual (--target-psnr) | **SHIPPED** | Residual added only when AI PSNR < threshold |
| Strict monotonic quality/size | **SHIPPED** | Guaranteed by native rate-distortion |
| ELIC codec | DEFERRED | cheng2020_anchor used instead (ships with CompressAI) |
| Latent diffusion / perceptual decode (--perceptual) | DEFERRED | Not implemented |

### Audio Compression

| Feature | Status | Notes |
|---------|--------|-------|
| EnCodec at 1.5/3/6/12/24 kbps | **SHIPPED** | --audio-bandwidth flag |
| QJL residual correction (--turbo) | **SHIPPED** | Restores sibilants/breathiness |
| Lossless bypass mode | **SHIPPED** | zstd-compressed exact payload |
| Speech file auto-detection (min 12 kbps) | **SHIPPED** | Keyword-based filename detection |
| Music file auto-detection (min 24 kbps) | **SHIPPED** | Keyword-based filename detection |
| Semantic mode (--semantic) | **SHIPPED** | Whisper ASR → MFCC embed → pyttsx3 TTS |
| Mimi / DAC codec upgrade | DEFERRED | EnCodec remains active codec |
| Neural post-filter (--enhance) | DEFERRED | No audio enhancement at decode |
| Speaker identity preservation in semantic mode | NOT FEASIBLE | pyttsx3 uses system TTS voice |

### Video Compression

| Feature | Status | Notes |
|---------|--------|-------|
| ssf2020 neural video codec (inter-frames) | **SHIPPED** | Learned optical flow, ~181x smaller than WebP P-frames |
| cheng2020_anchor keyframes (1 per 8-frame chunk) | **SHIPPED** | Consistent with image pipeline |
| QuickTime-compatible H.264 output | **SHIPPED** | yuv420p, BT.709, explicit PTS |
| Audio-in-video mux/demux (MODALITY_AV) | **SHIPPED** | PyAV extract + AudioEngine encode |
| --video-crf flag (default 23) | **SHIPPED** | H.264 output quality control |
| MAX_FRAME_DIM=512 cap (longest edge) | **SHIPPED** | Keeps CPU memory manageable |
| DCVC-HEM neural video codec | REJECTED | Too complex; ssf2020 used instead |
| Per-frame TurboQuant residual correction | NOT IMPLEMENTED | |
| Native resolution (no forced cap) | PARTIAL | User can increase MAX_FRAME_DIM in source |

### Container & CLI

| Feature | Status | Notes |
|---------|--------|-------|
| .tiny v8 format with modality byte per entry | **SHIPPED** | 5 modalities: image/audio/video/AV/semantic |
| Backward compat (v1-v7 files readable) | **SHIPPED** | Version-dispatched decode paths |
| SSF2 video payload (b"SSF2" magic) | **SHIPPED** | Chunked ssf2020 bitstream |
| SEMA semantic payload (b"SEMA" magic) | **SHIPPED** | Whisper + MFCC + TTS hints |
| Model registry (tiny models) | **SHIPPED** | Lists registered codec experts |
| Rate control (--target-size PCT) | **SHIPPED** | Maps % target to quality/bandwidth params |
| tiny info <file> | NOT IMPLEMENTED | Inspect metadata without decompressing |
| Progress bars (tqdm) | NOT IMPLEMENTED | |
| GPU acceleration (CUDA/MPS) | NOT IMPLEMENTED | All inference runs on CPU |
| pip-installable package | NOT IMPLEMENTED | Run via python main.py |

### Semantic Audio (Sprint 10.4) — Detail

Implemented in `core/semantic_audio.py`:
- **Encode**: Whisper ASR (configurable size: tiny/base/small/medium) transcribes speech.
  MFCC-based speaker embedding (20 coefficients via scipy DCT) stored as voice fingerprint.
  Speaking stats extracted: WPM, mean pitch F0, RMS dB.
- **Payload**: `b"SEMA"` magic + binary fields, ~500 bytes for 30s of speech.
- **Decode**: pyttsx3 system TTS resynthesizes from transcript with rate/volume hints.
- **Limitation**: Voice identity not preserved (system TTS voice used). Prosody hints
  stored but pyttsx3 macOS only exposes rate and volume, not pitch.

WARNING printed at both encode and decode: audio is resynthesized, not reconstructed.

---

## Known Limitations

- **Video resolution**: MAX_FRAME_DIM=512 is hard-coded in `core/video_engine.py`.
  4K or 1080p content is scaled down. Change the constant to increase (requires more RAM).
- **ssf2020 stride**: Input frames must be padded to multiples of 128. Black padding
  is added and cropped at decode.
- **Audio codec**: EnCodec (2022) is used, not Mimi (2024) or DAC (2023). Quality at
  low bitrates is good but not SOTA.
- **Semantic voice**: pyttsx3 output voice is the OS default TTS voice, not the original
  speaker. Speaker embeddings are stored but cannot drive pyttsx3 on macOS.
- **No GPU path**: All neural inference runs on CPU. Encode is slow for long videos.
- **No streaming**: Entire video is loaded into memory during encode.
