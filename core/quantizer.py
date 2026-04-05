import torch


class TurboQuantizer:
    """TurboQuant for latent tensors with per-channel scaling and quality-based bit depth."""

    # Monotonic bit-depth ladder: higher quality = more bits = larger file = better fidelity
    _QUALITY_BITS = {
        1: 4,
        2: 4,
        3: 4,
        4: 5,
        5: 6,
        6: 7,
        7: 8,
        8: 8,
    }

    @staticmethod
    def _bits_for_quality(quality: int) -> int:
        return TurboQuantizer._QUALITY_BITS.get(quality, 4)

    @staticmethod
    def quantize(tensor: torch.Tensor, bits: int = 4) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Quantize a latent tensor using per-channel scaling.

        Parameters
        ----------
        tensor : torch.Tensor, shape (N, C, H, W)
        bits   : quantization bit-width (2-8)

        Returns
        -------
        quantized : torch.int8 tensor, shape (N, C, H, W)
        scales    : torch.float32 tensor, shape (C,) — one scale per channel
        """
        q_min, q_max = -(2 ** (bits - 1)), 2 ** (bits - 1) - 1

        C = tensor.shape[1]
        ch_view = tensor.permute(1, 0, 2, 3).reshape(C, -1)  # (C, N*H*W)
        absmax = ch_view.abs().max(dim=1).values  # (C,)
        absmax = absmax.clamp(min=1e-8)
        scales = absmax / q_max  # (C,)

        scales_4d = scales.view(1, C, 1, 1)
        quantized = torch.round(tensor / scales_4d).clamp(q_min, q_max)
        return quantized.to(torch.int8), scales

    @staticmethod
    def quantize_at_quality(tensor: torch.Tensor, quality: int) -> tuple[torch.Tensor, torch.Tensor]:
        """Quantize using bit depth determined by quality level."""
        bits = TurboQuantizer._bits_for_quality(quality)
        return TurboQuantizer.quantize(tensor, bits=bits)

    @staticmethod
    def dequantize(quantized: torch.Tensor, scale: torch.Tensor) -> torch.Tensor:
        """
        Dequantize a latent tensor.

        Parameters
        ----------
        quantized : torch.int8 tensor, shape (N, C, H, W)
        scale     : scalar tensor (global) OR 1-D tensor of shape (C,)

        Returns
        -------
        torch.float32 tensor, shape (N, C, H, W)
        """
        q_f = quantized.to(torch.float32)
        if scale.dim() == 0:
            return q_f * scale
        C = q_f.shape[1]
        scales_4d = scale.view(1, C, 1, 1)
        return q_f * scales_4d
