"""
AudioEngine: encode/decode audio using Meta's EnCodec.

Encode pipeline:
  1. Load audio with soundfile (supports .wav, .mp3 via content)
  2. Resample + downmix to model sample rate (24 kHz, mono)
  3. Run EnCodec encode → codebook indices (K × T int16)
  4. Zstd-compress the indices for storage
  [Turbo] 5. Decode back → compute residual → rotate → 1-bit QJL

Decode pipeline:
  1. Unpack codebook indices
  2. Run EnCodec decode → float32 waveform
  3. Save as .wav
  [Turbo] 4. Reconstruct QJL correction, add to waveform

Binary payload format (returned by encode(), VERSION 2+):
  [B]  num_codebooks   (K)
  [I]  num_frames      (T)
  [I]  compressed_len          ← NEW in v2 payload; absent in legacy v1 payloads
  [*]  compressed (compressed_len bytes) — zstd int16 codebook indices
  [B]  turbo_flag      (0 = standard, 1 = turbo QJL correction follows)
  if turbo_flag == 1:
    [I]  seed
    [H]  chunk_size      (C dimension for QJL reshape)
    [H]  n_projections
    [I]  qjl_len
    [*]  qjl_bytes

Legacy v1 payload (no compressed_len, no turbo_flag):
  [B]  num_codebooks
  [I]  num_frames
  [*]  compressed (rest of payload — ends at payload boundary)
Detection: if payload[5:9] == b'\\x28\\xb5\\x2f\\xfd' (zstd magic), it is v1 (legacy).

.tiny entry format (written by header.py for modality=1):
  [H]  name_len
  [*]  name (UTF-8)
  [I]  orig_sr         original sample rate
  [f]  bandwidth       kbps used
  [I]  num_samples     original sample count at orig_sr
  [B]  num_channels    1 or 2
  [I]  data_len
  [*]  data            AudioEngine payload
"""

import os
import struct

import numpy as np
import soundfile as sf
import torch
import zstandard as zstd
from encodec import EncodecModel
from encodec.utils import convert_audio

ZSTD_LEVEL = 19
_ZSTD_MAGIC = b'\x28\xb5\x2f\xfd'  # identifies the start of a zstd frame
_TURBO_CHUNK_SIZE = 128   # treat 128 consecutive samples as a "channel vector"
_TURBO_N_PROJ = 64        # 1-bit QJL projections

_MODEL_CACHE: dict = {}


def _get_model(bandwidth: float) -> EncodecModel:
    key = ("24khz", bandwidth)
    if key not in _MODEL_CACHE:
        model = EncodecModel.encodec_model_24khz()
        model.set_target_bandwidth(bandwidth)
        model.eval()
        _MODEL_CACHE[key] = model
    return _MODEL_CACHE[key]


class AudioEngine:

    @staticmethod
    def encode(
        path: str,
        bandwidth: float = 3.0,
        use_turbo: bool = False,
        lossless_bypass: bool = False,
    ) -> tuple[bytes, int, int, int]:
        """
        Encode an audio file to a compact .tiny audio payload.

        Parameters
        ----------
        path             : path to .wav or .mp3 (soundfile-readable)
        bandwidth        : target kbps — 1.5, 3.0, 6.0, 12.0, or 24.0
        use_turbo        : if True, append a 1-bit QJL residual correction layer
        lossless_bypass  : if True, store waveform residual (original - AI) as int16

        Returns
        -------
        (payload_bytes, orig_sr, num_samples, num_channels)
        """
        from core.turbo_math import TurboMath

        audio_np, orig_sr = sf.read(path, always_2d=True)  # (T, C)
        num_samples = audio_np.shape[0]
        num_channels = audio_np.shape[1]

        # (C, T) float32 tensor
        audio = torch.from_numpy(audio_np.T.astype(np.float32))

        model = _get_model(bandwidth)
        target_sr = model.sample_rate  # 24000

        # Resample + downmix to model's expected channels/SR
        audio_converted = convert_audio(audio, orig_sr, target_sr, model.channels)
        audio_input = audio_converted.unsqueeze(0)  # (1, model.channels, T)

        with torch.no_grad():
            encoded_frames = model.encode(audio_input)

        # encoded_frames is a list of (codes, scale) tuples; codes shape: (1, K, T)
        codes = torch.cat([frame[0] for frame in encoded_frames], dim=-1)
        codes_np = codes.squeeze(0).cpu().numpy().astype(np.int16)  # (K, T)
        num_codebooks, num_frames = codes_np.shape

        raw = codes_np.tobytes()
        compressed = zstd.ZstdCompressor(level=ZSTD_LEVEL).compress(raw)

        # V2 payload format: [B num_codebooks][I num_frames][I compressed_len][* compressed]
        header = struct.pack("<BII", num_codebooks, num_frames, len(compressed))
        payload = header + compressed

        if lossless_bypass:
            # "True Lossless" mode: store waveform residual as zstd-compressed int16
            with torch.no_grad():
                frames_for_decode = [(codes, None)]
                audio_rec = model.decode(frames_for_decode)  # (1, C, T')
            audio_rec_1d = audio_rec.squeeze(0)  # (1, T') at 24kHz
            T_enc = audio_rec_1d.shape[-1]
            T_orig = audio_converted.shape[-1]
            T_common = min(T_enc, T_orig)
            residual = audio_converted[:, :T_common] - audio_rec_1d[:, :T_common]
            # Convert to int16 for compact storage (dynamic range -1..1 → -32767..32767)
            res_np = (residual.numpy().flatten() * 32767.0).clip(-32767, 32767).astype(np.int16)
            res_compressed = zstd.ZstdCompressor(level=ZSTD_LEVEL).compress(res_np.tobytes())
            # turbo_flag = 2 signals "lossless residual" mode
            payload += struct.pack("<B", 2)
            payload += struct.pack("<I", len(res_compressed))
            payload += res_compressed
            print(f"[audio] lossless bypass: waveform residual = {len(res_compressed):,} bytes "
                  f"(int16, {T_common} samples)")
            return payload, orig_sr, num_samples, num_channels

        if not use_turbo:
            payload += struct.pack("<B", 0)  # turbo_flag = 0
            return payload, orig_sr, num_samples, num_channels

        # ── Turbo QJL residual correction ─────────────────────────────────────
        seed = hash(os.path.basename(path)) & 0xFFFFFFFF

        with torch.no_grad():
            # Decode the EnCodec codes to get reconstruction
            frames_for_decode = [(codes, None)]
            audio_rec = model.decode(frames_for_decode)  # (1, C, T')

        audio_rec_squeezed = audio_rec.squeeze(0)  # (1, T') at 24kHz

        # Compute waveform residual at 24kHz
        T_enc = audio_rec_squeezed.shape[-1]
        T_orig = audio_converted.shape[-1]
        T_common = min(T_enc, T_orig)
        residual_1d = audio_converted[:, :T_common] - audio_rec_squeezed[:, :T_common]  # (1, T)

        # Reshape residual to (1, chunk_size, 1, n_chunks) for TurboMath
        chunk_size = _TURBO_CHUNK_SIZE
        T_trim = (T_common // chunk_size) * chunk_size
        if T_trim > 0:
            residual_4d = residual_1d[:, :T_trim].reshape(1, chunk_size, 1, T_trim // chunk_size)
            rotated = TurboMath.rotate(residual_4d, seed)
            qjl_bytes = TurboMath.qjl_project(rotated, seed, n_projections=_TURBO_N_PROJ)
            print(f"[audio] turbo QJL: residual_rms={residual_1d.abs().mean():.4f}, "
                  f"qjl={len(qjl_bytes):,}B")
        else:
            qjl_bytes = b""

        payload += struct.pack("<B", 1)  # turbo_flag = 1
        payload += struct.pack("<IHHI", seed, chunk_size, _TURBO_N_PROJ, len(qjl_bytes))
        payload += qjl_bytes

        # ── High-Frequency Injector (8kHz–12kHz crispness layer) ──────────────
        # Apply a high-pass filter to the residual to isolate upper-freq content
        # that EnCodec typically discards (S/T consonants, breath, room air).
        # Use QJL (1-bit projection) to keep the layer compact (~5-10KB for 23s).
        try:
            from scipy.signal import butter, sosfilt
            nyquist = target_sr / 2.0  # 12000 Hz for 24kHz
            hp_cutoff = 8000.0 / nyquist  # normalized cutoff (~0.667)
            sos = butter(4, hp_cutoff, btype="highpass", output="sos")
            residual_np = residual_1d.numpy()[0]  # (T,) float32
            hf_np = sosfilt(sos, residual_np).astype(np.float32)

            # Reshape HF residual for QJL: (1, chunk_size, 1, n_chunks)
            hf_tensor = torch.from_numpy(hf_np).unsqueeze(0)  # (1, T)
            T_hf = (len(hf_np) // chunk_size) * chunk_size
            if T_hf > 0:
                hf_4d = hf_tensor[:, :T_hf].reshape(1, chunk_size, 1, T_hf // chunk_size)
                hf_seed = seed ^ 0xCAFEBABE
                hf_rotated = TurboMath.rotate(hf_4d, hf_seed)
                hf_qjl = TurboMath.qjl_project(hf_rotated, hf_seed, n_projections=32)
                print(f"[audio] HF injector: hf_rms={np.abs(hf_np).mean():.4f}, "
                      f"hf_qjl={len(hf_qjl):,}B")
            else:
                hf_qjl = b""

            # Store: [H hf_n_proj][I hf_qjl_len][* hf_qjl_bytes]
            payload += struct.pack("<HI", 32, len(hf_qjl))
            payload += hf_qjl
        except ImportError:
            # scipy not available — skip HF layer
            payload += struct.pack("<HI", 32, 0)

        return payload, orig_sr, num_samples, num_channels

    @staticmethod
    def decode(payload: bytes, output_path: str, bandwidth: float = 3.0,
               enhance: bool = False) -> None:
        """
        Decode an AudioEngine payload and write a .wav file.

        Parameters
        ----------
        payload     : bytes from encode()
        output_path : destination .wav path
        bandwidth   : must match what was used in encode()
        enhance     : if True, run decoded waveform through NeuralPostFilter
        """
        from core.turbo_math import TurboMath

        offset = 0
        num_codebooks, num_frames = struct.unpack_from("<BI", payload, offset)
        offset += struct.calcsize("<BI")  # = 5

        # Detect v1 (legacy) vs v2 (compressed_len present):
        # In v1, bytes at offset 5+ start with the zstd magic (0xFD2FB528 LE = 28 B5 2F FD)
        # In v2, bytes at offset 5 are a uint32 compressed_len (rarely matches zstd magic)
        is_legacy_v1 = (len(payload) > offset + 4 and
                        payload[offset:offset + 4] == _ZSTD_MAGIC)

        if is_legacy_v1:
            # Old format: all remaining bytes are compressed data
            compressed = payload[offset:]
            raw = zstd.ZstdDecompressor().decompress(compressed)
            turbo_flag = 0  # legacy payloads have no turbo
        else:
            # V2 format: compressed_len is stored
            (compressed_len,) = struct.unpack_from("<I", payload, offset)
            offset += 4
            compressed = payload[offset: offset + compressed_len]
            raw = zstd.ZstdDecompressor().decompress(compressed)
            offset += compressed_len

            turbo_flag = 0
            if offset < len(payload):
                (turbo_flag,) = struct.unpack_from("<B", payload, offset)
                offset += 1

        codes_np = np.frombuffer(raw, dtype=np.int16).reshape(num_codebooks, num_frames).copy()
        codes = torch.from_numpy(codes_np).to(torch.int64).unsqueeze(0)  # (1, K, T)

        model = _get_model(bandwidth)

        with torch.no_grad():
            frames = [(codes, None)]
            audio_out = model.decode(frames)  # (1, C, T)

        waveform = audio_out.squeeze(0).cpu()  # (1, T) or (C, T)
        audio_np_out = waveform.numpy().T  # (T, C)

        # ── Apply turbo QJL correction if present ─────────────────────────────
        if turbo_flag == 1 and offset < len(payload):
            seed, chunk_size, n_projections, qjl_len = struct.unpack_from("<IHHI", payload, offset)
            offset += struct.calcsize("<IHHI")
            qjl_bytes = payload[offset: offset + qjl_len]

            if qjl_len > 0:
                # Compute n_chunks from the packed signs, not from T_audio
                # (T_audio may differ slightly from encode-time T_trim)
                n_chunks = (qjl_len * 8) // n_projections
                T_trim = n_chunks * chunk_size
                if T_trim > 0 and n_chunks > 0:
                    correction_shape = (1, chunk_size, 1, n_chunks)
                    rotated_correction = TurboMath.qjl_reconstruct(
                        qjl_bytes, seed, n_projections, correction_shape, scale=0.05
                    )
                    correction = TurboMath.rotate_inverse(rotated_correction, seed)
                    correction_1d = correction.reshape(1, -1).numpy().T  # (T_trim, 1)
                    audio_np_out[:T_trim] += correction_1d
                    audio_np_out = np.clip(audio_np_out, -1.0, 1.0)
                    print(f"[audio] turbo QJL correction applied: rms={np.abs(correction_1d).mean():.4f}")

            offset += qjl_len

            # ── High-Frequency Injector ────────────────────────────────────────
            if offset + 6 <= len(payload):
                hf_n_proj, hf_qjl_len = struct.unpack_from("<HI", payload, offset)
                offset += 6
                if hf_qjl_len > 0:
                    hf_qjl_bytes = payload[offset: offset + hf_qjl_len]
                    n_chunks_hf = (hf_qjl_len * 8) // hf_n_proj
                    T_hf = n_chunks_hf * chunk_size
                    if T_hf > 0 and n_chunks_hf > 0:
                        hf_seed = seed ^ 0xCAFEBABE
                        hf_shape = (1, chunk_size, 1, n_chunks_hf)
                        hf_rotated = TurboMath.qjl_reconstruct(
                            hf_qjl_bytes, hf_seed, hf_n_proj, hf_shape, scale=0.02
                        )
                        hf_correction = TurboMath.rotate_inverse(hf_rotated, hf_seed)
                        hf_1d = hf_correction.reshape(1, -1).numpy().T  # (T_hf, 1)
                        T_audio_out = audio_np_out.shape[0]
                        T_apply = min(T_hf, T_audio_out)
                        audio_np_out[:T_apply, 0:1] += hf_1d[:T_apply]
                        audio_np_out = np.clip(audio_np_out, -1.0, 1.0)
                        print(f"[audio] HF injector applied: rms={np.abs(hf_1d).mean():.4f}")
                offset += hf_qjl_len

        # ── Apply lossless waveform residual if present (turbo_flag == 2) ─────
        if turbo_flag == 2 and offset < len(payload):
            (res_compressed_len,) = struct.unpack_from("<I", payload, offset)
            offset += 4
            res_compressed = payload[offset: offset + res_compressed_len]
            res_raw = zstd.ZstdDecompressor().decompress(res_compressed)
            res_int16 = np.frombuffer(res_raw, dtype=np.int16).copy()
            res_float = res_int16.astype(np.float32) / 32767.0
            T_res = len(res_float)
            T_audio = audio_np_out.shape[0]
            T_apply = min(T_res, T_audio)
            audio_np_out[:T_apply, 0] += res_float[:T_apply]
            audio_np_out = np.clip(audio_np_out, -1.0, 1.0)
            print(f"[audio] lossless residual applied: {T_apply} samples, "
                  f"residual_rms={np.abs(res_float[:T_apply]).mean():.4f}")

        if enhance:
            waveform_t = torch.from_numpy(audio_np_out.T.astype(np.float32))  # (C, T)
            waveform_t = NeuralPostFilter.apply(waveform_t, model.sample_rate)
            audio_np_out = waveform_t.numpy().T  # (T, C)

        sf.write(output_path, audio_np_out, model.sample_rate, subtype="PCM_16")
        print(f"[audio] Decoded → {output_path}  ({audio_np_out.shape[0]/model.sample_rate:.1f}s)")


class NeuralPostFilter:
    """
    Decode-side audio enhancement filter.

    Takes EnCodec-decoded audio (which sounds muffled at low bitrates) and applies
    a cascade of torchaudio spectral operations to restore sibilants, breathiness,
    and perceived clarity. No file size change — applied only at decode time.

    Enhancement chain:
      1. Pre-emphasis (spectral tilt) — boosts high-frequency energy lost by EnCodec
      2. Treble shelf EQ (+3 dB above 4 kHz) — restores sibilant presence
      3. Spectral Wiener-like denoising via STFT magnitude smoothing — reduces
         the "grainy" codec artifacts in mid-frequencies
      4. Mild contrast enhancement — perceived loudness/clarity lift
      5. De-emphasis (inverts pre-emphasis) so overall tonal balance is preserved
      6. Normalise to the original RMS level — prevents volume change
    """

    @staticmethod
    def apply(waveform: torch.Tensor, sample_rate: int) -> torch.Tensor:
        """
        Apply enhancement to a decoded waveform.

        Parameters
        ----------
        waveform    : float32 tensor, shape (channels, samples)
        sample_rate : audio sample rate (typically 24000 for EnCodec)

        Returns
        -------
        Enhanced float32 tensor, same shape.
        """
        import torchaudio.functional as TAF

        # Keep a reference RMS so we can normalise loudness afterwards
        orig_rms = waveform.pow(2).mean().sqrt().clamp(min=1e-8)

        enhanced = waveform.clone()

        # ── Step 1: Pre-emphasis ──────────────────────────────────────────────
        # Boosts high-frequency content to counteract EnCodec's low-pass tendency.
        # coefficient 0.97 is a standard speech/audio pre-emphasis value.
        PRE_COEFF = 0.97
        enhanced = TAF.preemphasis(enhanced, coeff=PRE_COEFF)

        # ── Step 2: Treble shelf EQ (+3 dB above ~4 kHz) ─────────────────────
        # Restores "air" and sibilant clarity that collapses at low bitrates.
        # equalizer_biquad args: waveform, sample_rate, center_freq, gain_dB, Q
        TREBLE_FREQ = 4000.0   # Hz — start of sibilant / presence band
        TREBLE_GAIN = 3.0      # dB boost
        TREBLE_Q = 0.707       # Butterworth-like shelf slope
        enhanced = TAF.equalizer_biquad(enhanced, sample_rate, TREBLE_FREQ,
                                        TREBLE_GAIN, TREBLE_Q)

        # ── Step 3: Spectral Wiener-like denoising ────────────────────────────
        # EnCodec at 1.5 kbps introduces "boxy" quantisation noise in the 1–4 kHz
        # band.  We do a soft-mask pass in the STFT domain: attenuate bins where
        # the magnitude is below the local spectral floor estimate.
        N_FFT = 512
        HOP = N_FFT // 4
        WIN = torch.hann_window(N_FFT)
        NOISE_FLOOR_DB = -40.0   # bins below this fraction of peak → attenuate
        WIENER_ALPHA = 0.85      # blend: 0 = bypass, 1 = full denoising

        denoised_channels = []
        for ch in range(enhanced.shape[0]):
            x = enhanced[ch]  # (T,)
            # Compute STFT
            stft = torch.stft(x, n_fft=N_FFT, hop_length=HOP, win_length=N_FFT,
                              window=WIN, return_complex=True)  # (F, T_frames)
            mag = stft.abs()  # (F, T_frames)
            phase = stft.angle()

            # Estimate noise floor as a percentile of each frequency bin's magnitude
            # over time — a simple but effective Wiener-style estimate.
            noise_est = torch.quantile(mag, 0.10, dim=1, keepdim=True)  # (F, 1)

            # Wiener gain: suppress bins where signal ~ noise
            gain = (mag - noise_est).clamp(min=0.0) / mag.clamp(min=1e-10)
            # Soft threshold — blend with identity to avoid musical noise
            gain = WIENER_ALPHA * gain + (1.0 - WIENER_ALPHA)

            mag_clean = mag * gain
            stft_clean = torch.polar(mag_clean, phase)
            x_clean = torch.istft(stft_clean, n_fft=N_FFT, hop_length=HOP,
                                  win_length=N_FFT, window=WIN,
                                  length=x.shape[0])
            denoised_channels.append(x_clean)

        enhanced = torch.stack(denoised_channels, dim=0)  # (C, T)

        # ── Step 4: Mild contrast enhancement ────────────────────────────────
        # torchaudio contrast() is a soft-saturation / exciter that adds perceived
        # brightness without boosting overall level.  Enhancement level 50 is gentle.
        enhanced = TAF.contrast(enhanced, enhancement_amount=50.0)

        # ── Step 5: De-emphasis (undo pre-emphasis spectral tilt) ────────────
        enhanced = TAF.deemphasis(enhanced, coeff=PRE_COEFF)

        # ── Step 6: Loudness normalisation to original RMS ───────────────────
        new_rms = enhanced.pow(2).mean().sqrt().clamp(min=1e-8)
        enhanced = enhanced * (orig_rms / new_rms)

        # Hard clip — should be essentially a no-op after RMS normalisation
        enhanced = enhanced.clamp(-1.0, 1.0)

        print(f"[enhance] NeuralPostFilter applied — pre-emphasis + treble EQ + "
              f"Wiener denoising + contrast (Δrms={float(new_rms/orig_rms):.3f}→1.000)")

        return enhanced
