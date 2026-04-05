import torch
import torch.nn.functional as F

class ResidualEngine:
    @staticmethod
    def compute_residual(original, reconstruction):
        # Calculate the difference (signed)
        return original - reconstruction

    @staticmethod
    def apply_residual(reconstruction, residual):
        # Add the "detail patch" back to the AI base
        return torch.clamp(reconstruction + residual, 0, 1)

    @staticmethod
    def compress_residual(residual, scale_factor=0.5):
        # Downsample to save massive space
        small_res = F.interpolate(residual, scale_factor=scale_factor, mode='bilinear')
        return small_res
