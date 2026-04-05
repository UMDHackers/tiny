"""
ResidualEngine: compute, compress, decompress, and apply error maps.

During pack:
  residual = original - ai_base           (float32, range approx [-1, 1])
  residual_small = downsample(residual, 2x)
  payload = int8-quantize + zstd-compress(residual_small)

During unpack:
  residual_small = zstd-decompress + int8-dequantize(payload)
  residual = upsample(residual_small, target_size)
  final = clamp(ai_base + residual * strength, 0, 1)
"""

import io
import struct

import numpy as np
import torch
import torch.nn.functional as F
import zstandard as zstd
from PIL import Image

ZSTD_LEVEL = 19
WEBP_QUALITY = 90  # lossy WebP quality for residual — high quality for max PSNR gain


class ResidualEngine:

    @staticmethod
    def compute(original: torch.Tensor, ai_base: torch.Tensor) -> torch.Tensor:
        """
        Compute the residual error map.

        Parameters
        ----------
        original : (1, 3, H, W) float32, range [0, 1]
        ai_base  : (1, 3, H, W) float32, range [0, 1]

        Returns
        -------
        residual : (1, 3, H, W) float32, range [-1, 1]
        """
        # Match spatial size in case of rounding differences
        if original.shape[-2:] != ai_base.shape[-2:]:
            ai_base = F.interpolate(
                ai_base, size=original.shape[-2:], mode="bilinear", align_corners=False
            )
        return (original - ai_base).clamp(-1.0, 1.0)

    @staticmethod
    def compress(residual: torch.Tensor, downsample: int = 2) -> bytes:
        """
        Downsample, encode as WebP, and return a compact residual payload.

        The residual (range [-1, 1]) is shifted to [0, 255] uint8 for WebP,
        then stored as lossy WebP for compact size.

        Parameters
        ----------
        residual   : (1, 3, H, W) float32, values in [-1, 1]
        downsample : spatial downsampling factor (default 2)

        Returns
        -------
        bytes — packed payload:
          [HH]  downsampled (Hl, Wl) for upsample target
          [I]   webp_len
          [*]   WebP-encoded uint8 residual image
        """
        h, w = residual.shape[-2:]
        new_h = max(1, h // downsample)
        new_w = max(1, w // downsample)

        # High-quality downsampling via bilinear interpolation
        small = F.interpolate(
            residual, size=(new_h, new_w), mode="bilinear", align_corners=False
        )

        # Shift [-1, 1] → [0, 255] uint8 for image encoding
        uint8 = ((small.squeeze(0).permute(1, 2, 0).clamp(-1, 1) + 1.0) * 127.5)
        uint8_np = uint8.detach().cpu().numpy().astype(np.uint8)
        img = Image.fromarray(uint8_np, mode="RGB")

        # Encode as lossy WebP
        buf = io.BytesIO()
        img.save(buf, format="WEBP", quality=WEBP_QUALITY)
        webp_bytes = buf.getvalue()

        header = struct.pack("<HHI", new_h, new_w, len(webp_bytes))
        return header + webp_bytes

    @staticmethod
    def decompress(
        payload: bytes,
        target_h: int,
        target_w: int,
        strength: float = 1.0,
    ) -> torch.Tensor:
        """
        Decompress a WebP residual payload and upsample to (target_h, target_w).

        Parameters
        ----------
        payload  : bytes produced by compress()
        target_h, target_w : spatial size to upsample to
        strength : scaling factor in [0, 1] to modulate residual intensity

        Returns
        -------
        residual : (1, 3, target_h, target_w) float32
        """
        offset = 0
        small_h, small_w, webp_len = struct.unpack_from("<HHI", payload, offset)
        offset += struct.calcsize("<HHI")
        webp_bytes = payload[offset: offset + webp_len]

        img = Image.open(io.BytesIO(webp_bytes)).convert("RGB")
        arr = np.array(img, dtype=np.float32)
        # Undo the [0,255] → [-1,1] shift
        small = torch.from_numpy(arr).permute(2, 0, 1).unsqueeze(0) / 127.5 - 1.0

        # Upsample to target size
        residual = F.interpolate(
            small, size=(target_h, target_w), mode="bilinear", align_corners=False
        )
        return residual * strength

    @staticmethod
    def apply(ai_base: torch.Tensor, residual: torch.Tensor) -> torch.Tensor:
        """Add residual to AI base and clamp to [0, 1]."""
        return (ai_base + residual).clamp(0.0, 1.0)
