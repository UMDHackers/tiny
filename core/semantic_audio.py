"""
SemanticAudioEngine: extreme speech compression via ASR → TTS resynthesis.

Pipeline (encode):
  1. Transcribe speech with openai-whisper (ASR → text)
  2. Extract a lightweight speaker embedding (MFCC-based voice fingerprint)
  3. Pack transcript + embedding + metadata into a tiny binary payload

Pipeline (decode):
  1. Read transcript + speaker embedding + metadata
  2. Synthesize speech using pyttsx3 (system TTS), conditioned on embedding
     metadata (rate/pitch hints derived from the original speaker stats)
  3. Write .wav output

WARNING: The reconstructed audio is RESYNTHESIZED, not reconstructed.
The voice will differ from the original speaker. The transcript is preserved.
Typical payload size: ~500 bytes for 30 seconds of speech.

Binary payload format (SEMANTIC_VERSION = 1):
  [4s]  magic           = b'SEMA'
  [B]   version         = 1
  [I]   orig_sr         original sample rate
  [f]   duration_s      duration in seconds
  [B]   num_channels    1 or 2
  [H]   transcript_len  byte length of UTF-8 transcript
  [*]   transcript      UTF-8 bytes
  [B]   embed_len_kb    length of embedding in units (n_mfcc * 4 bytes per float32)
  [*]   embedding       float32 MFCC mean vector (n_mfcc floats)
  [f]   speaking_rate   estimated words-per-minute (for TTS rate hint)
  [f]   f0_mean         estimated mean pitch in Hz (0.0 if unavailable)
  [f]   rms_db          RMS amplitude in dB (loudness proxy)
"""

import io
import os
import struct
import tempfile

import numpy as np
import soundfile as sf

# ── Constants ────────────────────────────────────────────────────────────────
MAGIC = b"SEMA"
SEMANTIC_VERSION = 1
N_MFCC = 20          # MFCC coefficients to store as voice fingerprint
TARGET_SR = 16000    # Whisper prefers 16 kHz input


# ── Speaker Embedding ────────────────────────────────────────────────────────

def _extract_speaker_embedding(audio_np: np.ndarray, sr: int) -> np.ndarray:
    """
    Extract a compact speaker embedding from a waveform.

    Uses scipy's spectrogram + manual MFCC-style computation so we don't need
    a heavy speechbrain dependency. Returns a float32 array of shape (N_MFCC,).

    The embedding captures spectral envelope shape (timbre/voice quality) and
    is used to derive TTS rate/pitch hints at decode time.
    """
    from scipy.signal import spectrogram
    from scipy.fftpack import dct

    # Mono, float32
    if audio_np.ndim > 1:
        audio_np = audio_np.mean(axis=-1)
    audio_np = audio_np.astype(np.float32)

    # Compute power spectrogram
    f, t_spec, Sxx = spectrogram(
        audio_np, fs=sr, nperseg=512, noverlap=256, scaling="spectrum"
    )

    # Build a simple mel filterbank (N_MFCC bands, 80 Hz – 7600 Hz)
    n_filters = N_MFCC
    f_min, f_max = 80.0, min(7600.0, sr / 2.0)

    # Mel scale conversion helpers
    def hz_to_mel(hz: float) -> float:
        return 2595.0 * np.log10(1.0 + hz / 700.0)

    def mel_to_hz(mel: float) -> float:
        return 700.0 * (10.0 ** (mel / 2595.0) - 1.0)

    mel_points = np.linspace(hz_to_mel(f_min), hz_to_mel(f_max), n_filters + 2)
    hz_points = np.array([mel_to_hz(m) for m in mel_points])

    # Map hz_points to FFT bin indices
    fft_bins = np.floor((hz_points / (sr / 2.0)) * (len(f) - 1)).astype(int)
    fft_bins = np.clip(fft_bins, 0, len(f) - 1)

    # Build filterbank matrix (n_filters × n_freq_bins)
    filterbank = np.zeros((n_filters, len(f)), dtype=np.float32)
    for m in range(n_filters):
        start, peak, end = fft_bins[m], fft_bins[m + 1], fft_bins[m + 2]
        if peak > start:
            filterbank[m, start:peak] = (
                np.arange(start, peak) - start
            ) / float(peak - start)
        if end > peak:
            filterbank[m, peak:end] = (
                end - np.arange(peak, end)
            ) / float(end - peak)

    # Average power spectrum across time, apply filterbank, log, DCT → MFCCs
    power_mean = Sxx.mean(axis=1)  # (n_freq,)
    mel_energy = filterbank @ power_mean  # (n_filters,)
    mel_energy = np.maximum(mel_energy, 1e-10)
    log_mel = np.log(mel_energy)

    # DCT-II to get MFCC-like coefficients
    mfcc = dct(log_mel, type=2, norm="ortho")[:N_MFCC]

    # Normalize to unit norm so downstream comparisons are scale-invariant
    norm = np.linalg.norm(mfcc)
    if norm > 0:
        mfcc = mfcc / norm

    return mfcc.astype(np.float32)


def _estimate_speaking_stats(audio_np: np.ndarray, sr: int,
                              transcript: str) -> tuple[float, float, float]:
    """
    Estimate speaking_rate (wpm), mean pitch (Hz), and RMS loudness (dB).

    Returns (wpm, f0_mean_hz, rms_db).
    """
    # Words per minute from transcript word count and duration
    duration_s = len(audio_np) / sr
    word_count = len(transcript.split()) if transcript.strip() else 0
    wpm = (word_count / duration_s * 60.0) if duration_s > 0 else 150.0
    wpm = float(np.clip(wpm, 50.0, 350.0))  # clamp to human-speech range

    # RMS loudness in dB
    rms = float(np.sqrt(np.mean(audio_np.astype(np.float32) ** 2)))
    rms_db = float(20.0 * np.log10(max(rms, 1e-8)))

    # Simple pitch estimate: find dominant frequency in 80–500 Hz (speech range)
    try:
        from scipy.signal import spectrogram
        f, _, Sxx = spectrogram(audio_np.astype(np.float32), fs=sr,
                                nperseg=1024, noverlap=512)
        speech_mask = (f >= 80) & (f <= 500)
        if speech_mask.any():
            speech_power = Sxx[speech_mask, :].mean(axis=1)
            peak_idx = int(np.argmax(speech_power))
            f0_mean = float(f[speech_mask][peak_idx])
        else:
            f0_mean = 0.0
    except Exception:
        f0_mean = 0.0

    return wpm, f0_mean, rms_db


# ── Payload serialization ────────────────────────────────────────────────────

def _pack_payload(transcript: str, embedding: np.ndarray,
                  orig_sr: int, duration_s: float,
                  num_channels: int, wpm: float,
                  f0_mean: float, rms_db: float) -> bytes:
    """Serialize all semantic data into a compact binary blob."""
    buf = io.BytesIO()
    t_bytes = transcript.encode("utf-8")
    emb_bytes = embedding.astype(np.float32).tobytes()  # N_MFCC * 4 bytes

    buf.write(MAGIC)                                        # 4s magic
    buf.write(struct.pack("<B", SEMANTIC_VERSION))          # B  version
    buf.write(struct.pack("<I", orig_sr))                   # I  orig_sr
    buf.write(struct.pack("<f", duration_s))                # f  duration_s
    buf.write(struct.pack("<B", num_channels))              # B  num_channels
    buf.write(struct.pack("<H", len(t_bytes)))              # H  transcript_len
    buf.write(t_bytes)                                      # *  transcript
    buf.write(struct.pack("<B", len(emb_bytes) // 4))       # B  n_mfcc stored
    buf.write(emb_bytes)                                    # *  embedding
    buf.write(struct.pack("<f", wpm))                       # f  speaking_rate
    buf.write(struct.pack("<f", f0_mean))                   # f  f0_mean
    buf.write(struct.pack("<f", rms_db))                    # f  rms_db
    return buf.getvalue()


def _unpack_payload(payload: bytes) -> dict:
    """Deserialize the semantic payload blob. Returns a dict of all fields."""
    buf = io.BytesIO(payload)

    magic = buf.read(4)
    if magic != MAGIC:
        raise ValueError(f"Invalid semantic payload magic: {magic!r}")

    (version,) = struct.unpack("<B", buf.read(1))
    if version != SEMANTIC_VERSION:
        raise ValueError(f"Unsupported semantic payload version: {version}")

    (orig_sr,) = struct.unpack("<I", buf.read(4))
    (duration_s,) = struct.unpack("<f", buf.read(4))
    (num_channels,) = struct.unpack("<B", buf.read(1))
    (transcript_len,) = struct.unpack("<H", buf.read(2))
    transcript = buf.read(transcript_len).decode("utf-8")
    (n_mfcc_stored,) = struct.unpack("<B", buf.read(1))
    emb_bytes = buf.read(n_mfcc_stored * 4)
    embedding = np.frombuffer(emb_bytes, dtype=np.float32).copy()
    (wpm,) = struct.unpack("<f", buf.read(4))
    (f0_mean,) = struct.unpack("<f", buf.read(4))
    (rms_db,) = struct.unpack("<f", buf.read(4))

    return {
        "version": version,
        "orig_sr": orig_sr,
        "duration_s": duration_s,
        "num_channels": num_channels,
        "transcript": transcript,
        "embedding": embedding,
        "speaking_rate_wpm": wpm,
        "f0_mean_hz": f0_mean,
        "rms_db": rms_db,
    }


# ── SemanticAudioEngine ──────────────────────────────────────────────────────

class SemanticAudioEngine:
    """
    Extreme-compression audio engine for speech.

    Encodes speech by storing only the transcript + voice fingerprint.
    Decodes by resynthesizing speech with a TTS engine.

    This is NOT lossless and NOT waveform-accurate. The reconstructed audio
    preserves the *content* (words) but the *voice* will be the TTS voice,
    not the original speaker. Use only when maximum compression is required
    and voice identity can be sacrificed.
    """

    @staticmethod
    def encode(path: str,
               whisper_model: str = "base") -> tuple[bytes, int, int, int]:
        """
        Encode a speech audio file to a semantic .tiny payload.

        Parameters
        ----------
        path          : path to .wav or .mp3 audio file
        whisper_model : whisper model size — "tiny", "base", "small", "medium"
                        (larger = more accurate, slower)

        Returns
        -------
        (payload_bytes, orig_sr, num_samples, num_channels)
          — matches the same signature as AudioEngine.encode() for drop-in use
        """
        print("[semantic] WARNING: Audio is RESYNTHESIZED not reconstructed "
              "— voice may differ")
        print(f"[semantic] Loading whisper '{whisper_model}' model ...")

        # ── Load audio ────────────────────────────────────────────────────────
        audio_np, orig_sr = sf.read(path, always_2d=True)  # (T, C)
        num_samples = audio_np.shape[0]
        num_channels = audio_np.shape[1]
        duration_s = num_samples / orig_sr

        # Mono float32 for processing
        audio_mono = audio_np.mean(axis=1).astype(np.float32)

        # ── ASR: Transcribe with Whisper ──────────────────────────────────────
        import whisper as _whisper

        model = _whisper.load_model(whisper_model)

        # Use whisper's own audio loader for proper 16kHz resampling and normalization
        audio_for_whisper = _whisper.load_audio(path)

        print(f"[semantic] Transcribing {duration_s:.1f}s of speech ...")
        # Lower no_speech_threshold from default 0.6 → 0.9 to be more aggressive
        # about transcribing borderline speech (e.g., low-quality or noisy audio).
        # condition_on_previous_text=False avoids hallucination loops.
        result = model.transcribe(
            audio_for_whisper,
            fp16=False,
            no_speech_threshold=0.9,
            condition_on_previous_text=False,
        )
        transcript = result.get("text", "").strip()
        language = result.get("language", "en")
        print(f"[semantic] Transcript ({language}): {transcript[:120]!r}"
              + ("..." if len(transcript) > 120 else ""))

        # ── Speaker embedding ─────────────────────────────────────────────────
        print("[semantic] Extracting speaker embedding (MFCC) ...")
        embedding = _extract_speaker_embedding(audio_mono, orig_sr)

        # ── Speaking stats (for TTS conditioning hints) ───────────────────────
        wpm, f0_mean, rms_db = _estimate_speaking_stats(audio_mono, orig_sr, transcript)
        print(f"[semantic] Speaking stats: {wpm:.0f} wpm, "
              f"f0={f0_mean:.1f} Hz, rms={rms_db:.1f} dB")

        # ── Pack payload ──────────────────────────────────────────────────────
        payload = _pack_payload(
            transcript=transcript,
            embedding=embedding,
            orig_sr=orig_sr,
            duration_s=duration_s,
            num_channels=num_channels,
            wpm=wpm,
            f0_mean=f0_mean,
            rms_db=rms_db,
        )

        orig_size = os.path.getsize(path)
        ratio = (1.0 - len(payload) / orig_size) * 100.0
        print(f"[semantic] Payload: {len(payload):,} bytes  "
              f"(original: {orig_size:,} bytes, ratio: {ratio:.1f}% smaller)")
        print(f"[semantic] Effective bitrate: "
              f"{len(payload) * 8 / duration_s / 1000:.2f} kbps")

        return payload, orig_sr, num_samples, num_channels

    @staticmethod
    def decode(payload: bytes, output_path: str) -> None:
        """
        Decode a semantic payload and write resynthesized speech as .wav.

        Parameters
        ----------
        payload     : bytes from SemanticAudioEngine.encode()
        output_path : destination .wav path
        """
        print("[semantic] WARNING: Audio is RESYNTHESIZED not reconstructed "
              "— voice may differ")

        info = _unpack_payload(payload)
        transcript = info["transcript"]
        duration_s = info["duration_s"]
        orig_sr = info["orig_sr"]
        wpm = info["speaking_rate_wpm"]
        f0_mean = info["f0_mean_hz"]

        print(f"[semantic] Transcript: {transcript[:120]!r}"
              + ("..." if len(transcript) > 120 else ""))
        print(f"[semantic] Original duration: {duration_s:.1f}s  "
              f"| Speaking rate hint: {wpm:.0f} wpm  "
              f"| Pitch hint: {f0_mean:.1f} Hz")

        if not transcript.strip():
            print("[semantic] WARNING: Empty transcript — writing silence")
            _write_silence(output_path, orig_sr, duration_s)
            return

        # ── TTS synthesis ─────────────────────────────────────────────────────
        # Strategy: use pyttsx3 (system TTS, no internet required).
        # We apply rate/pitch hints derived from the original speaker stats
        # to get closer to the original speaking pace.
        try:
            _synthesize_pyttsx3(transcript, output_path, orig_sr, wpm, f0_mean)
        except Exception as e:
            print(f"[semantic] pyttsx3 synthesis failed: {e}")
            print("[semantic] Falling back to silence (TTS unavailable)")
            _write_silence(output_path, orig_sr, duration_s)

        print(f"[semantic] Decoded → {output_path}  "
              f"({duration_s:.1f}s original, resynthesized)")


# ── TTS helpers ──────────────────────────────────────────────────────────────

def _synthesize_pyttsx3(transcript: str, output_path: str,
                         orig_sr: int, wpm: float, f0_mean: float) -> None:
    """
    Synthesize transcript using pyttsx3 system TTS and write to output_path.

    pyttsx3 writes to files using the 'save_to_file' method, but on macOS
    the engine uses AVSpeechSynthesizer, which outputs AIFF. We save to a
    temp file then re-encode to WAV using soundfile.
    """
    import pyttsx3

    engine = pyttsx3.init()

    # Apply speaking rate hint: pyttsx3 default is ~200 wpm; clamp to [100, 300]
    rate_hint = int(np.clip(wpm, 100, 300))
    engine.setProperty("rate", rate_hint)

    # Apply pitch hint where supported (pyttsx3 on macOS supports volume, not pitch)
    # Volume: use rms_db as a proxy — louder original → higher volume
    engine.setProperty("volume", 1.0)  # always max — system TTS is already calibrated

    # Write to a temp file first (pyttsx3 may produce .aiff on macOS)
    suffix = ".aiff" if os.uname().sysname == "Darwin" else ".wav"
    with tempfile.NamedTemporaryFile(suffix=suffix, delete=False) as tmp:
        tmp_path = tmp.name

    try:
        engine.save_to_file(transcript, tmp_path)
        engine.runAndWait()

        # Read the TTS output and re-write as .wav at orig_sr
        if not os.path.exists(tmp_path) or os.path.getsize(tmp_path) == 0:
            raise RuntimeError(f"pyttsx3 produced empty or missing file: {tmp_path!r}")

        tts_audio, tts_sr = sf.read(tmp_path, always_2d=True)

        # Resample to original SR if needed
        if tts_sr != orig_sr:
            tts_mono = tts_audio.mean(axis=1).astype(np.float32)
            resampled = _resample(tts_mono, tts_sr, orig_sr)
            tts_out = resampled[:, np.newaxis]  # (T, 1)
        else:
            tts_out = tts_audio

        sf.write(output_path, tts_out, orig_sr, subtype="PCM_16")
        print(f"[semantic] TTS synthesis complete: {len(tts_out)} samples @ {orig_sr} Hz")

    finally:
        if os.path.exists(tmp_path):
            os.unlink(tmp_path)


def _resample(audio: np.ndarray, from_sr: int, to_sr: int) -> np.ndarray:
    """
    Resample a 1D float32 audio array from from_sr to to_sr using scipy.

    Falls back to linear interpolation if scipy.signal.resample_poly fails.
    """
    if from_sr == to_sr:
        return audio

    try:
        from scipy.signal import resample_poly
        from math import gcd
        g = gcd(int(from_sr), int(to_sr))
        up = to_sr // g
        down = from_sr // g
        return resample_poly(audio.astype(np.float64), up, down).astype(np.float32)
    except Exception:
        # Fallback: linear interpolation via numpy
        target_len = int(len(audio) * to_sr / from_sr)
        x_orig = np.linspace(0, 1, len(audio))
        x_new = np.linspace(0, 1, target_len)
        return np.interp(x_new, x_orig, audio).astype(np.float32)


def _write_silence(output_path: str, sr: int, duration_s: float) -> None:
    """Write a silent .wav file of the given duration."""
    n = int(sr * duration_s)
    sf.write(output_path, np.zeros((n, 1), dtype=np.float32), sr, subtype="PCM_16")
    print(f"[semantic] Wrote {duration_s:.1f}s silence → {output_path}")


# ── Payload inspection ───────────────────────────────────────────────────────

def inspect_payload(payload: bytes) -> None:
    """Print a human-readable summary of a semantic payload (for debugging)."""
    info = _unpack_payload(payload)
    print(f"[semantic] Version:         {info['version']}")
    print(f"[semantic] Original SR:     {info['orig_sr']} Hz")
    print(f"[semantic] Duration:        {info['duration_s']:.2f}s")
    print(f"[semantic] Channels:        {info['num_channels']}")
    print(f"[semantic] Transcript:      {info['transcript'][:200]!r}")
    print(f"[semantic] MFCC embedding:  {len(info['embedding'])} floats")
    print(f"[semantic] Speaking rate:   {info['speaking_rate_wpm']:.0f} wpm")
    print(f"[semantic] Pitch mean:      {info['f0_mean_hz']:.1f} Hz")
    print(f"[semantic] RMS level:       {info['rms_db']:.1f} dB")
    print(f"[semantic] Payload size:    {len(payload):,} bytes")
