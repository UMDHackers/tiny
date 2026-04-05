"""
Encoder: maps image files to quantized latents using CompressAI + TurboQuant.

Uses mbt2018_mean (mean-scale hyperprior) for better rate-distortion performance.
Aspect ratio is preserved by padding to 64-multiples instead of truncation-resizing.
A sharpening filter is applied post-decode to counteract VAE smoothing.
"""

from dataclasses import dataclass

import torch
import torch.nn.functional as F
from PIL import Image
from torchvision import transforms

from compressai.zoo import mbt2018_mean
from core.quantizer import TurboQuantizer

MAX_DIM = 1024  # maximum longest edge before resizing

_MODEL_CACHE: dict = {}

# Sharpening kernel (unsharp-mask style Laplacian boost)
_SHARPEN_KERNEL = torch.tensor(
    [[ 0, -1,  0],
     [-1,  5, -1],
     [ 0, -1,  0]], dtype=torch.float32
).view(1, 1, 3, 3)


@dataclass
class EncodedImage:
    name: str
    orig_w: int       # true pixel width before padding (used to crop after decode)
    orig_h: int       # true pixel height before padding
    pad_w: int        # padded width fed to the model (64-multiple)
    pad_h: int        # padded height fed to the model (64-multiple)
    quantized: torch.Tensor   # int8, shape (1, C, Hl, Wl)
    scales: torch.Tensor      # float32, shape (C,)
    quality: int


def _get_model(quality: int) -> torch.nn.Module:
    if quality not in _MODEL_CACHE:
        model = mbt2018_mean(quality=quality, pretrained=True)
        model.eval()
        _MODEL_CACHE[quality] = model
    return _MODEL_CACHE[quality]


def _load_image(path: str, max_dim: int = MAX_DIM) -> tuple[torch.Tensor, int, int, int, int]:
    """
    Load an image, proportionally resize to fit max_dim, then pad (not crop)
    to the next 64-multiples. Returns (padded_tensor, true_w, true_h, pad_w, pad_h).
    Aspect ratio is perfectly preserved — only zero-padding is added.
    """
    img = Image.open(path).convert("RGB")
    w, h = img.size

    # Step 1: proportional resize so longest edge ≤ max_dim
    if max(w, h) > max_dim:
        scale = max_dim / max(w, h)
        new_w = max(1, round(w * scale))
        new_h = max(1, round(h * scale))
        img = img.resize((new_w, new_h), Image.LANCZOS)
        w, h = new_w, new_h

    # Step 2: pad to next 64-multiples (preserves aspect ratio)
    pad_w = ((w + 63) // 64) * 64
    pad_h = ((h + 63) // 64) * 64

    tensor = transforms.ToTensor()(img).unsqueeze(0)  # (1, 3, H, W)

    if pad_w != w or pad_h != h:
        # Pad right and bottom only — keeps (0,0) anchor intact
        pad_right = pad_w - w
        pad_bottom = pad_h - h
        tensor = F.pad(tensor, (0, pad_right, 0, pad_bottom), mode="reflect")

    return tensor, w, h, pad_w, pad_h


def sharpen(tensor: torch.Tensor, strength: float = 0.3) -> torch.Tensor:
    """
    Apply a per-channel unsharp-mask sharpening to a (1, 3, H, W) float32 tensor.
    strength=0 → no sharpening, strength=1 → full Laplacian boost.
    """
    kernel = _SHARPEN_KERNEL.to(tensor.device)
    # Apply per channel
    channels = []
    for c in range(tensor.shape[1]):
        ch = tensor[:, c:c+1, :, :]
        sharpened = F.conv2d(ch, kernel, padding=1)
        # Blend: original + strength * (sharpened - original)
        channels.append(ch + strength * (sharpened - ch))
    return torch.cat(channels, dim=1).clamp(0, 1)


def encode(image_path: str, quality: int = 1) -> EncodedImage:
    """
    Encode a single image to a quantized latent.

    Parameters
    ----------
    image_path : path to source image
    quality    : model quality level (1–8); >5 uses 8-bit quantization

    Returns
    -------
    EncodedImage with quantized int8 latent and per-image metadata
    """
    model = _get_model(quality)
    tensor, true_w, true_h, pad_w, pad_h = _load_image(image_path)
    name = image_path.split("/")[-1]

    with torch.no_grad():
        y = model.g_a(tensor)
        quantized, scales = TurboQuantizer.quantize_at_quality(y, quality)

    return EncodedImage(
        name=name,
        orig_w=true_w,
        orig_h=true_h,
        pad_w=pad_w,
        pad_h=pad_h,
        quantized=quantized,
        scales=scales,
        quality=quality,
    )


def encode_batch(image_paths: list[str], quality: int = 1) -> list[EncodedImage]:
    """Encode multiple images, reusing the loaded model."""
    model = _get_model(quality)
    results = []
    with torch.no_grad():
        for path in image_paths:
            tensor, true_w, true_h, pad_w, pad_h = _load_image(path)
            name = path.split("/")[-1]
            y = model.g_a(tensor)
            quantized, scales = TurboQuantizer.quantize_at_quality(y, quality)
            results.append(EncodedImage(
                name=name,
                orig_w=true_w,
                orig_h=true_h,
                pad_w=pad_w,
                pad_h=pad_h,
                quantized=quantized,
                scales=scales,
                quality=quality,
            ))
    return results


def decode(encoded: EncodedImage, sharpen_strength: float = 0.3) -> torch.Tensor:
    """
    Decode a quantized latent back to an image tensor.

    Crops padding and applies sharpening to counteract VAE smoothing.

    Returns a float32 tensor of shape (1, 3, H, W) clamped to [0, 1],
    sized to the original (pre-pad) dimensions.
    """
    model = _get_model(encoded.quality)
    latent = TurboQuantizer.dequantize(encoded.quantized, encoded.scales)
    with torch.no_grad():
        out = model.g_s(latent).clamp(0, 1)

    # Crop back to pre-pad true dimensions
    out = out[:, :, :encoded.orig_h, :encoded.orig_w]

    # Sharpening post-process
    if sharpen_strength > 0:
        out = sharpen(out, strength=sharpen_strength)

    return out
