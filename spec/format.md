# .tiny Binary Format Specification

All multi-byte integers are **little-endian**.
Floating-point values follow IEEE 754.

---

## Current Format: VERSION 8

### Global Header (7 bytes)

| Offset | Size | Type   | Field       | Value          |
|--------|------|--------|-------------|----------------|
| 0      | 4    | u8[4]  | magic       | `54 49 4E 59` ("TINY") |
| 4      | 1    | u8     | version     | `08`           |
| 5      | 2    | u16 LE | entry_count | number of entries (images + audio + video) |

### Per-Entry Layout (VERSION 8, repeated `entry_count` times)

Each entry starts with a **modality byte**:

| Value | Modality              |
|-------|-----------------------|
| 0     | Image                 |
| 1     | Audio (EnCodec)       |
| 2     | Video (SSF2)          |
| 3     | AV (Video + Audio muxed) |
| 4     | Semantic Audio (ASR→TTS) |

---

#### Modality 0 — Image (cheng2020_anchor native bitstream)

| Size      | Type    | Field        | Notes |
|-----------|---------|--------------|-------|
| 1         | u8      | modality     | = 0 |
| 2         | u16 LE  | name_len     | byte length of filename |
| name_len  | utf-8   | name         | original filename (no path) |
| 2         | u16 LE  | orig_w       | true pixel width before padding |
| 2         | u16 LE  | orig_h       | true pixel height before padding |
| 1         | u8      | codec_id     | 0 = cheng2020_anchor, 5 = mbt2018_legacy |
| 1         | u8      | quality      | quality level used during encode (1-6) |
| 4         | u32 LE  | bitstream_len | byte length of native bitstream |
| bitstream_len | u8[] | bitstream   | cheng2020_anchor native bitstream (see below) |
| 4         | u32 LE  | residual_len | byte length of residual payload (0 = no residual) |
| residual_len | u8[] | residual_data | WebP-compressed residual (if residual_len > 0) |

**Native Bitstream Layout** (cheng2020_anchor / mbt2018 model.compress() output):

```
[H]  num_string_groups     (typically 2: y_strings and z_strings)
[HH] shape_h, shape_w      (latent spatial dimensions)
For each string group:
  [I]  num_strings          (batch size, typically 1)
  For each string:
    [I]  string_len
    [*]  string_data         (entropy-coded bytes from CompressAI)
```

**Residual Data**: The residual (`original - reconstruction`) is downsampled 2x,
encoded as WebP (via PIL), and stored raw. On decode, it is upsampled and added
back to the reconstruction. Residual is only present when `--residual` or
`--target-psnr` is used at encode time.

---

#### Modality 1 — Audio (EnCodec)

| Size      | Type    | Field        | Notes |
|-----------|---------|--------------|-------|
| 1         | u8      | modality     | = 1 |
| 2         | u16 LE  | name_len     | byte length of filename |
| name_len  | utf-8   | name         | original filename |
| 4         | u32 LE  | orig_sr      | original sample rate (Hz) |
| 4         | f32 LE  | bandwidth    | EnCodec bandwidth in kbps |
| 4         | u32 LE  | num_samples  | sample count at orig_sr |
| 1         | u8      | num_channels | 1 or 2 |
| 4         | u32 LE  | data_len     | byte length of audio payload |
| data_len  | u8[]    | data         | AudioEngine payload (see below) |

**AudioEngine Payload Layout** (VERSION 2):

```
[B]  num_codebooks     (K — typically 32 at 3kbps)
[I]  num_frames        (T — EnCodec frame count)
[I]  compressed_len    (byte length of zstd-compressed codebook indices)
[*]  compressed        (zstd-compressed int16 codebook indices, shape K×T)
[B]  turbo_flag        (0 = standard, 1 = QJL correction follows)
if turbo_flag == 1:
  [I]  seed
  [H]  chunk_size      (C dimension for QJL reshape, default 128)
  [H]  n_projections   (1-bit projections, default 64)
  [I]  qjl_len
  [*]  qjl_bytes       (TurboQuant QJL correction)
```

**Legacy v1 audio payload** (no `compressed_len` field, no `turbo_flag`):

```
[B]  num_codebooks
[I]  num_frames
[*]  compressed   (rest of payload — zstd frame starts at offset 5)
```
Detection: if `payload[5:9] == b'\x28\xb5\x2f\xfd'` (zstd magic), it is legacy v1.

---

#### Modality 2 — Video (SSF2 — ssf2020 Scale-Space Flow)

| Size      | Type    | Field        | Notes |
|-----------|---------|--------------|-------|
| 1         | u8      | modality     | = 2 |
| 2         | u16 LE  | name_len     | byte length of filename |
| name_len  | utf-8   | name         | original filename |
| 4         | u32 LE  | orig_w       | original frame width |
| 4         | u32 LE  | orig_h       | original frame height |
| 4         | f32 LE  | fps          | original video framerate |
| 4         | u32 LE  | total_frames | total frames in original video |
| 4         | u32 LE  | data_len     | byte length of video payload |
| data_len  | u8[]    | data         | VideoEngine SSF2 payload (see below) |

**SSF2 Video Payload** (magic `b"SSF2"`, VERSION 2):

```
[4s]  magic          = b"SSF2"
[H]   quality        quality level (1-6)
[HH]  enc_w, enc_h   padded frame resolution (multiple of 128)
[HH]  true_w, true_h true frame resolution before padding
[f]   fps            original video fps (float32)
[I]   total_frames   total frames in original video
[I]   chunk_count    number of chunks stored
Per chunk:
  [I]   start_frame   first frame index in this chunk
  [I]   chunk_len     frames in this chunk (1..8)
  [I]   chunk_data_len total bytes of chunk payload
  [*]   chunk_data    ssf2020 bitstream (see chunk layout below)
```

**Chunk Data Layout** (from `_serialize_chunk`):

```
[I]   n_frames       frames in this chunk
Per frame i in [0..n_frames):
  [I]   n_string_groups   (typically 2)
  Per string group:
    [I]  n_strings         (batch size, typically 1)
    Per string:
      [I]  slen
      [*]  string_data
  If i == 0 (keyframe):
    [HH]  shape_h, shape_w  (latent spatial shape for cheng2020_anchor keyframe)
```

Frame 0 of each chunk is a **keyframe** encoded with `cheng2020_anchor`.
Frames 1-7 are **inter-frames** encoded with `ssf2020`, which uses learned optical
flow for temporal prediction from the previous frame.

**Legacy video payload** (WebP P-frames, magic `b"TURV"` or no magic):
Still decodable via `_decode_legacy_payload()` in `core/video_engine.py`.

---

#### Modality 3 — AV (Video + Audio muxed)

| Size      | Type    | Field        | Notes |
|-----------|---------|--------------|-------|
| 1         | u8      | modality     | = 3 |
| 2         | u16 LE  | name_len     | byte length of filename |
| name_len  | utf-8   | name         | original filename |
| 4         | u32 LE  | orig_w       | original frame width |
| 4         | u32 LE  | orig_h       | original frame height |
| 4         | f32 LE  | fps          | original video fps |
| 4         | u32 LE  | total_frames | total frames |
| 4         | u32 LE  | vid_data_len | byte length of SSF2 video payload |
| vid_data_len | u8[] | vid_data    | SSF2 video payload (same as modality 2) |
| 4         | u32 LE  | orig_sr      | audio sample rate (Hz) |
| 4         | f32 LE  | bandwidth    | audio bandwidth kbps |
| 4         | u32 LE  | num_samples  | audio sample count at orig_sr |
| 1         | u8      | num_channels | 1 or 2 |
| 4         | u32 LE  | aud_data_len | byte length of audio payload |
| aud_data_len | u8[] | aud_data   | AudioEngine payload (same as modality 1) |

---

#### Modality 4 — Semantic Audio (ASR→TTS)

| Size      | Type    | Field        | Notes |
|-----------|---------|--------------|-------|
| 1         | u8      | modality     | = 4 |
| 2         | u16 LE  | name_len     | byte length of filename |
| name_len  | utf-8   | name         | original filename |
| 4         | u32 LE  | orig_sr      | original sample rate (Hz) |
| 4         | u32 LE  | num_samples  | sample count at orig_sr |
| 1         | u8      | num_channels | 1 or 2 |
| 4         | u32 LE  | data_len     | byte length of semantic payload |
| data_len  | u8[]    | data         | SemanticAudioEngine SEMA payload (see below) |

**SEMA Semantic Payload** (magic `b"SEMA"`, VERSION 1):

```
[4s]  magic           = b"SEMA"
[B]   version         = 1
[I]   orig_sr         original sample rate (Hz)
[f]   duration_s      duration in seconds (float32)
[B]   num_channels    1 or 2
[H]   transcript_len  byte length of UTF-8 transcript
[*]   transcript      UTF-8 text (Whisper ASR output)
[B]   n_mfcc          number of MFCC floats stored (= N_MFCC = 20)
[*]   embedding       float32 MFCC mean vector (n_mfcc × 4 bytes)
[f]   speaking_rate   estimated words-per-minute (TTS rate hint)
[f]   f0_mean         estimated mean pitch in Hz (0.0 if unavailable)
[f]   rms_db          RMS amplitude in dB (loudness proxy)
```

Typical payload size: ~500 bytes for 30 seconds of speech.

WARNING: Semantic mode does NOT reconstruct the original waveform. The
voice will differ — pyttsx3 uses the system TTS voice. Only the text
content is preserved. Use only when maximum compression is required and
voice identity can be sacrificed.

---

## Legacy Formats (VERSION 1-7)

All legacy formats remain readable. The version byte in the global header
determines the decode path:

| Version | Description |
|---------|-------------|
| 1       | Image-only, global zstd scale (bmshj2018 latents) |
| 2       | Image-only, per-image residual field added |
| 3       | Mixed modality byte per entry (image=0, audio=1) |
| 4       | Per-channel scales in image payload |
| 5       | TurboQuant polar+QJL layer added (turbo_flag byte) |
| 6       | (intermediate) |
| 7       | Quality byte added to image entries |
| 8       | Current: native cheng2020_anchor bitstream, SSF2 video, SEMA audio |

Versions 1-7 use `mbt2018_mean` for image decode (not cheng2020_anchor).
The model is selected based on latent channel count `C` in the payload:
128→quality 1, 192→quality 3, 320→quality 5.

---

## C Struct Sketch (VERSION 8)

```c
// Global header
typedef struct {
    char     magic[4];      // "TINY"
    uint8_t  version;       // 8
    uint16_t entry_count;
} __attribute__((packed)) TinyHeader;

// Per-entry: modality byte first, then modality-specific data
// modality 0 (image):
typedef struct {
    uint8_t  modality;      // 0
    uint16_t name_len;
    // uint8_t  name[name_len];
    uint16_t orig_w, orig_h;
    uint8_t  codec_id;      // 0 = cheng2020_anchor
    uint8_t  quality;       // 1-6
    uint32_t bitstream_len;
    // uint8_t  bitstream[bitstream_len];
    uint32_t residual_len;  // 0 = no residual
    // uint8_t  residual_data[residual_len];
} __attribute__((packed)) TinyImageEntry;

// modality 4 (semantic audio):
typedef struct {
    uint8_t  modality;      // 4
    uint16_t name_len;
    // uint8_t  name[name_len];
    uint32_t orig_sr;
    uint32_t num_samples;
    uint8_t  num_channels;
    uint32_t data_len;
    // uint8_t  data[data_len];  // SEMA payload
} __attribute__((packed)) TinySemanticEntry;
```
