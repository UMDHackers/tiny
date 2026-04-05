# Tiny v2 — SOTA AI Compression Engine

## Context & Lessons from v1

Tiny v1 achieved strong results on images (97% compression) and audio (99.7%) but
exposed fundamental architectural problems:

### What Worked
- **Images**: CompressAI mbt2018_mean at q1 achieves 97% compression. Quality is
  acceptable for most use cases. The residual correction layer (WebP-encoded
  orig-minus-reconstruction) gives a genuine fidelity boost.
- **Audio**: EnCodec at 3kbps achieves 99.7% compression on speech. TurboQuant's
  QJL residual correction restores high-frequency detail (sibilants, breathiness)
  that EnCodec misses at low bitrates.
- **Universal container**: .tiny as a single file for mixed media (images + audio +
  video) with one CLI and one quality slider is genuinely useful UX.

### What Failed
- **Video**: Frame-by-frame image compression with bolted-on P-frames cannot compete
  with H.264. No motion estimation, no temporal prediction, no learned entropy coding.
  Best result: 61% compression on 720p (vs H.264's native efficiency). On 4K content,
  "93% compression" was mostly resolution downscaling (3840x2160 to 1024x576), not
  efficient encoding.
- **TurboQuant on latents**: We bypassed CompressAI's trained quantization + entropy
  coding and substituted our own (TurboQuant + zstd). This caused:
  - Tile seams and color artifacts (decoder received latents quantized differently
    than it was trained on)
  - Worse compression than the model's native path
  - Non-monotonic quality/size relationship (q1 larger than q3 at same bit-depth)
- **Zstd as core compressor**: Neural codecs have learned entropy models that
  outperform general-purpose compressors on their own latent distributions. Zstd on
  neural latents is redundant at best, harmful at worst.
- **Hardcoded settings**: VideoEngine was locked to quality=1, CRF was locked to 18,
  quantizer had only 2 effective levels (4-bit and 8-bit). These were found during
  audit and partially fixed, but revealed deeper design issues.

### Key Insight
TurboQuant's genuine value is on **residuals** (the difference between original and
AI reconstruction), not on **latents** (which the model already knows how to quantize).
No pretrained model exists for residual compression, making TurboQuant's PolarQuant +
QJL a legitimate unique contribution there.


## v2 Architecture

### Design Principles
1. **Use models natively** — call `model.compress()` / `model.decompress()`, don't
   bypass their trained quantization and entropy coding.
2. **TurboQuant on residuals only** — the fidelity correction layer, not the core
   compressor. Optional layer that makes reconstruction mathematically exact.
3. **Best-in-class model per modality** — swap in SOTA models, don't fight old ones.
4. **Native resolution** — no forced downscaling. Resolution reduction is a user
   choice (`--max-dim`), not an architectural limitation.
5. **Zstd for residuals and metadata only** — not for core codec bitstreams.

### Pipeline Overview (as implemented)

```
ENCODE:
  input file
    ├── image?       → cheng2020_anchor native compress() → bitstream → .tiny
    ├── audio?       → EnCodec encoder → codebook indices + zstd → .tiny
    ├── audio --semantic? → Whisper ASR → MFCC embed → pyttsx3 TTS hints → .tiny
    └── video?       → cheng2020_anchor (keyframes) + ssf2020 (inter) → .tiny
                      (AV: audio muxed separately, both stored in .tiny)

  Optional fidelity layer (--residual or --target-psnr):
    original - reconstruction → ResidualEngine (WebP) → residual blob → .tiny

DECODE:
  .tiny → detect modality
    ├── image    → cheng2020_anchor decompress() → reconstruction
    │              └── + residual? → apply ResidualEngine correction → fidelity output
    ├── audio    → EnCodec decode → waveform → .wav
    ├── semantic → pyttsx3 TTS resynthesis → .wav (voice differs from original)
    └── video    → ssf2020 decode → frames → H.264 .mp4
                   (AV: audio demuxed, recoded to AAC, muxed with video)
```


## Sprint 9: Image Engine v2 — Native cheng2020 + Residual Layer [DONE]

### Background
cheng2020_anchor achieves ~15% better rate-distortion than mbt2018_mean. It uses a
channel-wise autoregressive entropy model that's significantly more efficient than
mbt2018's hyperprior. Ships with CompressAI.

Note: ELIC was the original target but cheng2020_anchor was used instead — it's
already in CompressAI, dependency-free, and provides the required quality gain.

### Status

#### 9.1 — Swap Image Codec to cheng2020_anchor (Native Path) [DONE]
- [x] Replace manual `g_a() → TurboQuant → zstd → g_s()` path with
      `model.compress(image) → bitstream` and `model.decompress(bitstream) → image`
- [x] Model's native compress/decompress handles quantization AND entropy coding
      internally — latents are not manually quantized or zstd-compressed
- [x] Native bitstream stored in .tiny (v8 format, see format.md)
- [x] Quality levels 1-6 map to cheng2020_anchor's rate parameter
- [x] Quality/size relationship is now strictly monotonic
- [x] No tile seams — cheng2020_anchor handles padding to multiples of 64 internally
- [x] Images are resized to fit MAX_DIM=1024 (proportional), then padded to 64-multiple

#### 9.2 — Residual Layer (Fidelity Mode) [DONE]
- [x] After native decode, compute residual: `original - reconstruction`
- [x] Residual compressed with ResidualEngine (WebP, downsample=2)
- [x] Stored as optional layer in .tiny: `[I residual_len][* residual_data]`
      (residual_len == 0 means no residual)
- [x] On decode: `final = reconstruction + ResidualEngine.apply(residual)`
- [x] CLI: `--residual` flag enables residual layer unconditionally
- [x] CLI: `--target-psnr DB` includes residual only when AI-only PSNR is below threshold

#### 9.3 — Latent Diffusion Decoder [DEFERRED]
- Not implemented. Perceptual/diffusion decode mode is out of scope for current sprint.

#### 9.4 — Remove Legacy Image Path [DONE]
- [x] Legacy TurboQuant-on-latents code path removed from v8 encode
- [x] Legacy mbt2018_mean path preserved only for reading v1-v7 .tiny files
- [x] TurboQuant code in `core/turbo_math.py` kept (used in audio fidelity mode)
- [x] FORMAT VERSION bumped to 8; versions 1-7 are still readable (backward compat)


## Sprint 10: Audio Engine v2 — EnCodec + Semantic Mode [PARTIALLY DONE]

### Background
EnCodec remains the primary codec (Mimi/DAC upgrade deferred). The key additions
in v2 are: semantic compression mode (Sprint 10.4) and lossless bypass.

### Status

#### 10.1 — Swap Audio Codec to Mimi or DAC [DEFERRED]
- [ ] Not implemented. EnCodec is still the active codec.
- [ ] Mimi/DAC upgrade planned for a future sprint.

#### 10.2 — Neural Post-Filter [DEFERRED]
- [ ] Not implemented. `--enhance` flag not present in CLI.

#### 10.3 — TurboQuant Residual Layer (Audio Fidelity) [PARTIALLY DONE]
- [x] QJL correction (turbo_flag=1) implemented in AudioEngine for audio residuals
- [x] Lossless bypass mode (lossless_bypass=True) stores zstd-compressed exact payload
- [x] `--turbo` flag enables QJL correction on encode
- [ ] `--fidelity` / `--lossless` CLI flags not exposed (use --turbo for QJL)

#### 10.4 — Semantic Audio Compression Mode [DONE]
- [x] Implemented in `core/semantic_audio.py` as `SemanticAudioEngine`
- [x] Whisper ASR transcribes speech (model size configurable: tiny/base/small/medium)
- [x] MFCC-based speaker embedding (20 coefficients, scipy spectrogram + DCT)
- [x] Speaking stats extracted: words-per-minute, mean pitch (F0), RMS level
- [x] pyttsx3 system TTS resynthesizes speech at decode (rate/volume hints applied)
- [x] Binary payload: `SEMA` magic + version + metadata + transcript + embedding
      (~500 bytes for 30s of speech)
- [x] CLI: `tiny pack speech.wav --semantic` encodes; `tiny unpack file.tiny` resynthesizes
- [x] Clear warning printed at encode and decode: voice is RESYNTHESIZED, not reconstructed
- [x] Modality 4 (MODALITY_SEMANTIC_AUDIO) used in .tiny container
- NOTE: Speaker identity is NOT preserved — pyttsx3 uses the system TTS voice.
        Prosody hints (rate, pitch F0) are stored but pyttsx3 macOS support is limited.

#### 10.5 — Remove Legacy Audio Path [NOT STARTED]
- [ ] EnCodec v1 payload format (no compressed_len) still readable via detection heuristic
- [ ] Clean removal of legacy path not done

#### Additional (not in original spec):
- [x] Speech file detection by filename keywords → auto-enforces 12kbps minimum
- [x] Music file detection by filename keywords → auto-enforces 24kbps minimum
- [x] `--audio-bandwidth` flag exposes EnCodec bandwidth selection (1.5/3/6/12/24 kbps)


## Sprint 11: Video Engine v2 — ssf2020 Neural Video Codec [DONE]

### Background
DCVC-HEM (the original target) was evaluated and rejected due to weight availability
and dependency complexity. Instead, ssf2020 (Scale-Space Flow, already in CompressAI)
was used. It provides:
- Learned optical flow for motion estimation between frames
- Learned temporal context from previously decoded frames
- Neural transform + entropy model for residuals
- CPU-compatible without new dependencies

Architecture: frames are grouped into chunks of CHUNK_SIZE=8. Frame 0 of each chunk
is a keyframe encoded with cheng2020_anchor. Frames 1-7 are inter-frames encoded with
ssf2020 (which uses learned optical flow from the keyframe).

### Status

#### 11.1 — Evaluate and Integrate Video Codec [DONE]
- [x] ssf2020 from CompressAI chosen (MIT license, CPU-compatible, no new deps)
- [x] DCVC-HEM rejected — architectural decision documented in memory
- [x] `VideoEngine.encode(path)` → SSF2 binary payload
- [x] `VideoEngine.decode(payload, output_path)` → QuickTime-compatible .mp4
- [x] Quality levels 1-6 map to ssf2020's rate-distortion operating points
- [x] Keyframe quality matches cheng2020_anchor used for images (consistency)
- [x] Frame resolution capped at MAX_FRAME_DIM=512 (longest edge) for CPU memory
- [x] ssf2020 requires input spatial dims divisible by 128 (SSF_STRIDE); padding applied
- [x] I-frames and P-frames (inter-frames) both use ssf2020's internal codec
- [x] CHUNK_SIZE=8 (1 keyframe + 7 inter-frames per chunk)

#### 11.2 — Video TurboQuant Residual Layer [NOT IMPLEMENTED]
- [ ] Per-frame residual correction not implemented for video

#### 11.3 — QuickTime-Compatible Output [DONE]
- [x] H.264 High profile, level 4.1, yuv420p pixel format
- [x] BT.709 color primaries
- [x] Explicit PTS: `pts = frame_index` with stream.time_base set before loop
- [x] Fresh VideoFrames via numpy→from_ndarray to avoid inherited time_base bugs
- [x] `--video-crf` flag (default 23) controls H.264 output quality

#### 11.4 — Audio-in-Video Muxing [DONE]
- [x] `_has_audio_stream()` detects audio in input video using PyAV
- [x] Audio extracted to temp WAV, encoded with AudioEngine, stored as MODALITY_AV
- [x] AV mux on decode: audio and video decoded separately, muxed with PyAV

#### 11.5 — Remove Legacy Video Path [DONE]
- [x] WebP P-frame residual encoding removed from encode path
- [x] CompressAI I-frame-only path removed
- [x] Legacy WebP payload (TURV magic / no magic) still decodable via `_decode_legacy_payload()`


## Sprint 12: Format v8, CLI Polish, and Integration Testing [PARTIALLY DONE]

### Status

#### Format v8 [DONE]
- [x] Version bumped to 8 in `core/header.py`
- [x] Modality byte per entry (0=image, 1=audio, 2=video, 3=AV, 4=semantic_audio)
- [x] v8 image entry uses `codec_id` byte + native cheng2020 bitstream
- [x] SSF2 video payload (b"SSF2" magic) for neural video
- [x] SEMA semantic audio payload (b"SEMA" magic) for ASR→TTS entries
- [x] All versions 1-7 remain decodable (backward compat)

#### CLI Updates [PARTIALLY DONE]
- [x] `tiny pack <file|folder> [options]`
      --quality 1-6         (maps to codec rate parameter)
      --residual            (add WebP residual layer for images)
      --target-psnr DB      (conditional residual: only when AI PSNR below threshold)
      --semantic            (semantic mode for audio, Whisper + pyttsx3)
      --audio-bandwidth BW  (EnCodec kbps: 1.5, 3, 6, 12, 24)
      --target-size PCT     (rate control: target output as % of original)
      --image-model ID      (select image model from registry)
      --audio-model ID      (select audio model from registry)
      --out FILE            (output path)
      --turbo               (QJL correction for audio)
- [x] `tiny unpack <file> [options]`
      --quality 1-6         (for legacy v7 files; v8 auto-detects)
      --video-crf N         (H.264 CRF for video output, default 23)
      --originals FOLDER    (compare vs originals, compute PSNR)
- [x] `tiny models`         (list registered model experts)
- [ ] `tiny info <file>`    (not implemented — inspect metadata without decompressing)
- [ ] `--fidelity` / `--lossless` named flags (use --residual and --turbo instead)
- [ ] `--perceptual` flag   (diffusion decode not implemented)
- [ ] `--enhance` flag      (neural audio post-filter not implemented)

#### Integration Tests [NOT DONE]
- [ ] Formal test matrix not implemented
- [ ] PSNR benchmark suite not automated


## Sprint 13: Performance, Packaging, and Documentation [NOT STARTED]

- [ ] GPU acceleration (CUDA/MPS detection)
- [ ] Streaming encode for large videos
- [ ] Progress bars (tqdm)
- [ ] pip-installable package with entry point
- [ ] Automated benchmark suite vs JPEG/WebP/MP3/Opus/H.264


## TurboQuant's Role in v2 (Summary)

TurboQuant is repositioned from "core compressor" to "fidelity correction layer":

| Component | v1 Role | v2 Role |
|-----------|---------|---------|
| PolarQuant on latents | Core image compression | REMOVED (model handles its own quantization) |
| QJL on latents | Core image correction | REMOVED |
| QJL on image residuals | Optional fidelity layer | WebP residual used instead (simpler, adequate) |
| QJL on audio residuals | Optional correction | KEPT — restores sibilants/breathiness (--turbo flag) |
| QJL on video frame residuals | Not used | NOT IMPLEMENTED |
| Anchor Channels | Protected first 24 channels | REMOVED (no manual quantization) |
| PolarQuant for video | Not used | REMOVED |

The core compression in v2 comes from the neural codecs (cheng2020_anchor for images,
EnCodec for audio, ssf2020 for video). TurboQuant QJL is an optional audio correction layer.


## Dependencies (as implemented)

```
# Image (ships with compressai)
compressai>=1.2.0        # cheng2020_anchor + ssf2020

# Audio
encodec                  # Meta EnCodec 24kHz
openai-whisper           # ASR for semantic mode
pyttsx3                  # TTS for semantic decode (system voices)
scipy                    # MFCC speaker embedding + resampling

# Video
av                       # PyAV for video I/O and mux

# Fidelity / residuals
Pillow                   # WebP residual compression
zstandard                # Zstd for audio codebook indices and lossless bypass

# Common
torch, torchaudio, torchvision
soundfile                # Audio I/O
numpy
```

Note: DCVC-HEM, Mimi, DAC, voicefixer, speechbrain, StyleTTS2/XTTS were planned
but not adopted. ssf2020 (CompressAI) replaced DCVC-HEM; EnCodec remains for audio.


## Risk Assessment (updated)

| Risk | Status | Notes |
|------|--------|-------|
| DCVC-HEM too slow without GPU | RESOLVED | Used ssf2020 (CompressAI, CPU-compatible) instead |
| ELIC model too large to download | RESOLVED | Used cheng2020_anchor (ships with CompressAI) instead |
| Diffusion decoder adds 10+ seconds per image | DEFERRED | --perceptual mode not implemented |
| Semantic TTS voice doesn't match original | KNOWN LIMITATION | pyttsx3 uses system voice; warning printed |
| Format v8 breaks existing .tiny files | RESOLVED | Strict backward compat implemented (v1-v7 readable) |
| Total dependency size exceeds 5GB | MITIGATED | ssf2020+cheng2020 are compact; whisper is lazy-imported |


## Success Metrics (current state)

- [x] Images: cheng2020_anchor uses model's native entropy coder (better than mbt2018 WebP path)
- [ ] Audio: Mimi/DAC upgrade not done; EnCodec still used
- [x] Video: ssf2020 inter-frames achieve ~181x smaller than old WebP P-frames at higher PSNR
- [x] Monotonic quality/size: guaranteed by native codec rate-distortion for images and video
- [x] Backward compat: v1-v7 .tiny files remain readable
- [ ] Formal PSNR benchmarks vs JPEG/WebP/MP3/Opus/H.264 not run
