"""
Smoke-test that exercises all three PSNR verdict tiers:
  >30 dB  → "Great."
  20-30   → "Acceptable."
  <20 dB  → "TurboQuantizing is too aggressive."
"""

import io
import struct
import tempfile

import torch
import zstandard as zstd
from PIL import Image

from core.header import MAGIC, VERSION, _psnr, _tensor_to_image, unpack_with_originals
from core.quantizer import TurboQuantizer
from compressai.zoo import mbt2018_mean


def _save_solid_image(path: str, color=(128, 180, 90), size=(256, 256)) -> None:
    """Save a flat solid-color image — very easy to compress/reconstruct."""
    Image.new("RGB", size, color).save(path)


def _make_tiny_blob(tensor: torch.Tensor, name: str, bits: int) -> bytes:
    """
    Build a minimal .tiny blob from a raw tensor using custom quantization bits.
    Uses legacy global-scale format (non-zero scale field) for simplicity.
    """
    model = mbt2018_mean(quality=1, pretrained=True)
    model.eval()

    q_max = 2 ** (bits - 1) - 1
    q_min = -(2 ** (bits - 1))

    with torch.no_grad():
        y = model.g_a(tensor)
        scale = y.abs().max() / q_max
        quantized = torch.round(y / scale).clamp(q_min, q_max).to(torch.int8)

    N, C, Hl, Wl = quantized.shape
    _, _, orig_h, orig_w = tensor.shape
    raw_bytes = quantized.cpu().numpy().tobytes()
    compressed = zstd.ZstdCompressor(level=1).compress(raw_bytes)

    buf = io.BytesIO()
    buf.write(struct.pack("<4sBH", MAGIC, VERSION, 1))
    name_b = name.encode()
    buf.write(struct.pack("<H", len(name_b)))
    buf.write(name_b)
    buf.write(struct.pack("<HH", orig_w, orig_h))
    buf.write(struct.pack("<4I", N, C, Hl, Wl))
    buf.write(struct.pack("<f", scale.item()))  # non-zero → legacy global scale
    buf.write(struct.pack("<I", len(compressed)))
    buf.write(compressed)
    buf.write(struct.pack("<I", 0))  # residual_data_len = 0 (VERSION 2, no residual)
    return buf.getvalue()


def verdict(psnr: float) -> str:
    if psnr > 30:
        return "Great."
    if psnr < 20:
        return "TurboQuantizing is too aggressive."
    return "Acceptable."


def test_great(tmpdir: str) -> None:
    """Solid-color image round-tripped at 8-bit → PSNR should be >30 dB."""
    orig = f"{tmpdir}/solid.png"
    _save_solid_image(orig, color=(100, 149, 237), size=(256, 256))

    from core.header import _image_to_tensor
    tensor, _, _ = _image_to_tensor(orig)
    blob = _make_tiny_blob(tensor, "solid.png", bits=8)

    results = unpack_with_originals(blob, [orig], f"{tmpdir}/out_great", quality=1)
    psnr = results[0][1]
    v = verdict(psnr)
    print(f"[test_great]       PSNR={psnr:.2f} dB → {v}")
    assert psnr > 30, f"Expected >30, got {psnr:.2f}"
    assert v == "Great."


def test_acceptable(tmpdir: str) -> None:
    """Real photo at quality=1 with 4-bit quant → 20-30 dB range."""
    from core.header import pack
    blob = pack(["samples/hill.jpg"], quality=1)
    results = unpack_with_originals(blob, ["samples/hill.jpg"], f"{tmpdir}/out_accept", quality=1)
    psnr = results[0][1]
    v = verdict(psnr)
    print(f"[test_acceptable]  PSNR={psnr:.2f} dB → {v}")
    assert 20 <= psnr <= 30, f"Expected 20-30, got {psnr:.2f}"
    assert v == "Acceptable."


def test_too_aggressive(tmpdir: str) -> None:
    """2-bit quantization → PSNR should drop below 20 dB."""
    from core.header import _image_to_tensor
    tensor, _, _ = _image_to_tensor("samples/hill.jpg")
    blob = _make_tiny_blob(tensor, "hill.jpg", bits=2)
    results = unpack_with_originals(blob, ["samples/hill.jpg"], f"{tmpdir}/out_aggressive", quality=1)
    psnr = results[0][1]
    v = verdict(psnr)
    print(f"[test_too_aggressive] PSNR={psnr:.2f} dB → {v}")
    assert psnr < 20, f"Expected <20, got {psnr:.2f}"
    assert v == "TurboQuantizing is too aggressive."


if __name__ == "__main__":
    with tempfile.TemporaryDirectory() as tmpdir:
        print("Running PSNR verdict tests...\n")
        test_great(tmpdir)
        test_acceptable(tmpdir)
        test_too_aggressive(tmpdir)
        print("\nAll three verdict tiers verified.")
