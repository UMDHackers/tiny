# TinyVision: AI Night Vision + Ultra-Compression Pipeline

**Version**: 0.1 (design spec)
**Date**: 2026-04-05
**Status**: Design phase — no code yet

---

## 1. Product Thesis

Existing night vision AI products (DeepNight, Topaz, etc.) enhance images. We **enhance + compress + transmit**. The `.tiny` transport layer is the differentiator — nobody else in the night vision space does learned video compression on the output stream.

**Target**: bandwidth-constrained edge devices that capture in low light.

### 1.1 Niche Markets (ranked by viability)

**Tier 1 — Wildlife / Trail Cameras**
- Market: 15M+ trail cams sold/year in US alone ($200M+ market)
- Pain point: cellular trail cams burn through data plans transmitting blurry night footage
- Our value: enhance night footage on-device → compress 90% with ssf2020 → transmit over LTE/LoRa
- Moat: nobody combines AI enhancement + neural compression for trail cams
- Dataset availability: abundant (trail cam forums, iNaturalist, Snapshot Serengeti)
- Audio angle: semantic audio for ambient sounds, animal calls — 300 bytes for 30s over LoRa

**Tier 2 — Astrophotography**
- Market: $500M+ amateur astronomy market, high willingness to pay ($300+ for software)
- Pain point: stacking dozens of noisy long-exposure frames is slow and manual
- Our value: AI-assisted stacking + denoising → compressed storage/sharing
- Critical constraint: ZERO hallucination tolerance. Must denoise, never invent detail.
- Datasets: Astrobin, telescope archives, NASA, ESA Hubble/JWST public datasets
- Risk: community is technically sophisticated and skeptical of AI "enhancement"
- Strategy: position as "noise reduction" not "enhancement" — conservative, provable

**Tier 3 — Security / Surveillance**
- Market: huge but crowded (Hikvision, Dahua own it)
- Entry point: small/medium businesses who want better night footage without $2K cameras
- Our angle: retrofit existing cameras with a processing box that enhances + compresses
- Risk: liability-heavy, competitive, regulatory

**Tier 4 — Automotive / Dashcam**
- Skip for now: liability too high, automotive qualification too slow

### 1.2 What We Don't Build
- We don't build thermal/IR cameras — that's a different sensor, different physics
- We don't compete on raw sensor hardware — Sony STARVIS sensors are commoditized
- We don't build real-time viewfinder enhancement (latency too high on CPU) — we build store-and-forward

---

## 2. Technical Architecture

```
┌─────────────────────────────────────────────┐
│                  EDGE DEVICE                 │
│                                              │
│  Camera Sensor (Sony STARVIS IMX585)         │
│       ↓ RAW Bayer frames (12-bit)            │
│  Sensor Frontend (normalize → float tensor)  │
│       ↓ [B, 4, H, W] RGGB planes            │
│  Enhancement Model (SNR-Aware or LLFlow)     │
│       ↓ enhanced RGB [B, 3, H, W]            │
│  Temporal Fusion (3-5 frame window)          │
│       ↓ stabilized frame sequence            │
│  tiny pack (ssf2020 neural video codec)      │
│       ↓ .tiny payload (80-90% smaller)       │
│  Transmit (LoRa / LTE / WiFi / SD card)     │
│                                              │
│  Optional: semantic audio for ambient sound  │
│       ↓ ~300 bytes per 30s clip              │
└─────────────────────────────────────────────┘
                    ↓
┌─────────────────────────────────────────────┐
│                 CLOUD / PC                   │
│                                              │
│  tiny unpack --enhance                       │
│       ↓ decoded frames + optional post-filter│
│  Display / Storage / Alert                   │
└─────────────────────────────────────────────┘
```

### 2.1 Enhancement Models

Two models selected based on current SOTA benchmarks:

**SNR-Aware Low-Light Image Enhancement (primary)**
- Paper: "SNR-Aware Low-Light Image Enhancement" (CVPR 2022)
- Why: signal-to-noise ratio aware — adapts enhancement strength per-pixel based on local noise level. Avoids over-enhancing already-bright regions.
- Runs on: PyTorch, CPU/MPS compatible
- GitHub: open source, pretrained weights available
- Best for: wildlife/security (real-world noise distributions)

**LLFlow (secondary/astrophotography)**
- Paper: "Low-Light Image Enhancement with Normalizing Flow" (AAAI 2022)
- Why: normalizing flow architecture produces sharper results with fewer artifacts than GAN-based methods. Important for astrophotography where artifacts = fake stars.
- Runs on: PyTorch, CPU/MPS compatible
- GitHub: open source, pretrained weights available
- Best for: astrophotography (artifact-free, conservative enhancement)

**Rejected alternatives:**
- ZeroDCE/ZeroDCE++: fast but lower quality than SNR-Aware
- RetinexFormer: outperformed by SNR-Aware on most benchmarks
- DCVC-HEM: CUDA-only, custom C++ ops (same reason as video codec rejection)

### 2.2 Temporal Fusion

Most enhancement models process frames independently → flickering. Our approach:

1. Compute optical flow between consecutive frames (ssf2020's motion field, already available)
2. Warp previous enhanced frame to current frame's coordinate space
3. Blend: `output = alpha * current_enhanced + (1 - alpha) * warped_previous`
4. Alpha is per-pixel, based on flow confidence (high motion = trust current frame more)

This reuses infrastructure we already built for ssf2020 inter-frame coding.

### 2.3 Sensor Frontend

**RAW normalization layer** — thin adapter per sensor model:

```python
class SensorFrontend:
    """Normalize sensor-specific RAW to a standard [B, 4, H, W] RGGB tensor."""

    def __init__(self, sensor_profile: str):
        # Load sensor-specific params: black level, white level,
        # Bayer pattern, color matrix, noise profile per ISO
        self.profile = load_profile(sensor_profile)

    def process(self, raw_frame: np.ndarray, iso: int) -> torch.Tensor:
        # 1. Subtract black level
        # 2. Normalize to [0, 1] using white level
        # 3. Reorder Bayer channels to canonical RGGB
        # 4. Apply noise profile metadata (SNR map for SNR-Aware model)
        return tensor
```

**If manufacturing our own cameras**: standardize on Sony STARVIS II (IMX585 or IMX678):
- 1/1.2" sensor, 8.3MP, 0.001 lux minimum illumination
- Back-illuminated — critical for low-light quantum efficiency
- $15-40 per unit at 1K quantity
- Well-documented noise characteristics
- 12-bit RAW output via MIPI CSI-2

**If retrofitting existing cameras**: use LibRaw to read any manufacturer's RAW + DNG as fallback.

### 2.4 Audio Pipeline

For devices with microphones (trail cams, security cams):

| Mode | Use case | Bitrate | Quality |
|------|----------|---------|---------|
| EnCodec (default) | Faithful audio reproduction | 1.5-24 kbps | Reconstructed |
| Semantic (--semantic) | Speech/voice transmission | 0.04-0.1 kbps | Resynthesized |
| Semantic + enhance | Best quality semantic | 0.04-0.1 kbps + decode-side filter | Resynthesized + enhanced |

For LoRa transmission (300 bps effective throughput):
- EnCodec at 1.5 kbps: need 5x realtime to transmit (30s audio takes 150s to send)
- Semantic at 0.04 kbps: 30s audio transmits in <1s

---

## 3. Dataset Collection Strategy

### 3.1 Public Datasets (free, start here)

| Dataset | Size | Content | Use |
|---------|------|---------|-----|
| LOL (Low-Light) | 500 pairs | Indoor/outdoor scenes, dark+reference | Training baseline |
| LOL-v2 | 789 pairs | Extended LOL with synthetic + real pairs | Training |
| SID (See in the Dark) | 5094 RAW pairs | Sony/Fuji RAW short/long exposure pairs | RAW pipeline training |
| MCR (Multi-Camera RAW) | 4200 pairs | Multiple camera models | Sensor generalization |
| Snapshot Serengeti | 7.1M images | Trail cam wildlife photos (many night) | Fine-tuning for wildlife niche |
| LSDIR | 84,991 images | High-res diverse scenes | Clean reference images |

### 3.2 Self-Collected Data (for niche fine-tuning)

**Paired capture rig:**
1. Mount camera on tripod
2. Capture scene at target ISO (high, dark) = noisy input
3. Immediately capture same scene at low ISO + long exposure = clean reference
4. Align with homography (tripod reduces but doesn't eliminate motion)
5. Cost: $0 if you own a camera with manual controls

**For astrophotography:**
- Capture 100 short exposures + 1 tracked long exposure of same star field
- Short exposures = noisy input, stacked long exposure = reference
- Telescope + tracker + camera: community members will contribute data for free tool access

**For wildlife:**
- Place trail cam next to DSLR with flash/IR illuminator
- Trail cam captures = dark input, DSLR captures = reference
- Or: use daytime footage as reference, synthetically darken for training pairs
- Warning: synthetic darkening (adding Poisson noise + reducing exposure) produces good but not perfect training data — fine for v1, need real pairs for v2

### 3.3 Synthetic Data (augmentation)

```python
def synthesize_dark(clean_image, target_iso=6400):
    """Create synthetic dark/noisy version of a clean image."""
    # 1. Reduce exposure (divide by exposure_ratio)
    # 2. Add shot noise (Poisson, signal-dependent)
    # 3. Add read noise (Gaussian, sensor-dependent)
    # 4. Quantize to target bit depth
    # 5. Apply inverse camera response curve
    # This gives ~80% of the benefit of real paired data
```

---

## 4. Hardware Reference Design

### 4.1 Bill of Materials (1K unit run)

| Component | Part | Cost | Notes |
|-----------|------|------|-------|
| Image sensor | Sony IMX585 (STARVIS II) | $25 | 1/1.2", 0.001 lux, 4K |
| SoC | Rockchip RK3588S | $35 | 4x A76 + 3 TOPS NPU |
| Lens | M12 f/1.4 4mm | $12 | fast aperture critical for night |
| RAM | 4GB LPDDR4X | $8 | for model inference |
| Storage | 32GB eMMC | $5 | buffer before transmission |
| Comms | Quectel EC200A (LTE Cat-4) | $12 | or SX1276 for LoRa ($4) |
| Battery | 18650 x4 (if battery-powered) | $8 | ~6 months standby for trail cam |
| PCB + assembly | 4-layer PCB | $8 | |
| Housing | IP66 weatherproof | $6 | injection mold amortized |
| Microphone | MEMS (optional) | $1 | for semantic audio |
| **Total BOM** | | **$120** | **At 1K units** |
| **At 10K units** | | **~$85** | Volume pricing |
| **At 100K units** | | **~$60** | Competitive with basic trail cams |

### 4.2 Why Sony STARVIS II

- Back-illuminated structure: 2x quantum efficiency vs front-illuminated
- Dual-gain HDR: captures both highlights and shadows in single exposure
- 0.001 lux minimum illumination (quarter moon is ~0.01 lux)
- Well-characterized noise model: published readout noise, dark current, gain curves
- Used in: most modern security cameras already — supply chain is stable

### 4.3 Processing Budget (RK3588S)

| Stage | Time (720p) | Notes |
|-------|-------------|-------|
| RAW normalize | <1ms | NEON SIMD |
| SNR-Aware inference | 50-80ms | NPU (RKNN), int8 quantized |
| Temporal fusion | 5-10ms | CPU, simple flow warp |
| ssf2020 encode | 200-400ms | CPU (could optimize with NPU) |
| **Total per frame** | **~300-500ms** | **2-3 FPS** |

2-3 FPS is fine for trail cams (motion-triggered, burst capture). Not enough for real-time viewfinder. For security cameras doing continuous recording, you'd process every 3rd-5th frame and interpolate.

---

## 5. Implementation Sprints

### Sprint NV-1: Enhancement Prototype (Python, laptop)
- [ ] Download and run SNR-Aware on LOL dataset, measure PSNR/SSIM
- [ ] Download and run LLFlow on LOL dataset, compare
- [ ] Integrate winner into tiny pipeline: enhance → pack → unpack → compare
- [ ] Test on synthetic night images (darken daytime photos)
- [ ] Deliverable: `tiny pack night_photo.png --enhance-night --out enhanced.tiny`

### Sprint NV-2: Temporal Fusion
- [ ] Extract optical flow from ssf2020's motion field during encode
- [ ] Implement 3-frame temporal blending with flow-warped alignment
- [ ] Test on a dark video clip (synthetic or real)
- [ ] Measure flicker reduction vs frame-independent enhancement
- [ ] Deliverable: `tiny pack dark_video.mp4 --enhance-night` with temporal stability

### Sprint NV-3: Astrophotography Mode
- [ ] Collect/download 50+ star field images with varying exposure
- [ ] Fine-tune SNR-Aware or LLFlow specifically for star fields
- [ ] Add star-preservation constraint: detected point sources must not be removed or added
- [ ] Implement multi-frame stacking integration (align + denoise)
- [ ] Deliverable: `tiny pack starfield/ --astro` mode

### Sprint NV-4: Sensor Frontend + RAW Pipeline
- [ ] Implement SensorFrontend class with IMX585 profile
- [ ] Test with RAW captures from a Sony or Raspberry Pi HQ camera
- [ ] LibRaw integration for arbitrary camera RAW support
- [ ] DNG fallback for unknown sensors
- [ ] Deliverable: can ingest RAW files directly

### Sprint NV-5: Edge Deployment Prototype
- [ ] Export SNR-Aware model to ONNX
- [ ] Quantize to int8 with ONNX Runtime or RKNN toolkit
- [ ] Test inference speed on Raspberry Pi 5 or RK3588 dev board
- [ ] Integrate with tiny encode pipeline on device
- [ ] Deliverable: working edge prototype on dev board

### Sprint NV-6: LoRa Transmission Test
- [ ] Set up two SX1276 modules (transmitter + receiver)
- [ ] Transmit a .tiny video payload over LoRa at 300 bps
- [ ] Calculate: how many seconds of enhanced night video per LoRa transmission?
- [ ] Add semantic audio for ambient sound at near-zero cost
- [ ] Deliverable: end-to-end demo — capture dark → enhance → compress → LoRa → decompress

---

## 6. Competitive Analysis

| Feature | DeepNight | Topaz | Our approach |
|---------|-----------|-------|--------------|
| Enhancement quality | High | High | Medium-High (80% of theirs) |
| Compression | None | None | 80-90% (ssf2020) |
| Transmission | None | None | LoRa / LTE optimized |
| Edge deployment | No | No | Yes (RK3588 NPU) |
| Temporal stability | Unknown | No | Yes (flow-based fusion) |
| Astrophoto mode | No | Yes | Yes (artifact-free LLFlow) |
| Audio | No | No | Yes (EnCodec + semantic) |
| Hardware | Generic | Generic | Optimized sensor pairing |
| Price point | $200/yr | $200 | $60-120 hardware + free software |

**Our moat is the full pipeline, not any single component.** Enhancement alone won't beat DeepNight. Compression alone won't beat H.265. The combination — enhance + compress + transmit from edge device — is what nobody else does.

---

## 7. Risks and Mitigations

| Risk | Likelihood | Mitigation |
|------|-----------|------------|
| Enhancement adds artifacts (fake stars, phantom objects) | High for v1 | Conservative model, artifact detection pass, user-configurable strength |
| Trail cam market requires CE/FCC certification | Certain | Budget $5K-10K per certification, factor into timeline |
| Sony discontinues IMX585 | Low | STARVIS II family has 10+ compatible sensors |
| LoRa bandwidth too low for useful video | Medium | Target 1-5 second bursts, not continuous. Trail cam = motion-triggered bursts |
| Enhancement model too slow on edge | Medium | Quantization to int8 + NPU offload. Worst case: enhance on cloud after transmission |
| Astrophotography community rejects AI enhancement | Medium | Position as "denoising" not "enhancement". Publish PSNR metrics transparently. Never claim "AI-generated detail" |

---

## 8. Open Questions

1. **Build vs partner on hardware?** Building cameras is a different business than building software. Consider licensing the pipeline to existing trail cam manufacturers (Reconyx, Stealth Cam) instead.
2. **Subscription vs one-time?** Cloud decompression service vs on-device only?
3. **How much fine-tuning data is needed?** SNR-Aware was trained on LOL (500 pairs). For wildlife niche, likely need 500-1000 paired trail cam images for good domain adaptation.
4. **Patent landscape?** Night vision + AI compression combo may be novel enough to patent. Worth a search.
