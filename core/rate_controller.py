"""
RateController: dynamic bit-budgeting and rate control for .tiny compression.

Maps a target size percentage to engine parameters (quality, turbo, bit depths, etc.)
so that callers only need to specify a single target-size value.
"""

from dataclasses import dataclass


@dataclass
class RateParams:
    quality: int
    use_turbo: bool
    norm_bits: int
    dir_bits: int
    with_residual: bool
    residual_downsample: int
    audio_bandwidth: float
    lossless_bypass: bool = False

    def describe(self) -> str:
        parts = [f"quality={self.quality}"]
        if self.use_turbo:
            parts.append("turbo")
        parts.append(f"norm_bits={self.norm_bits}")
        parts.append(f"dir_bits={self.dir_bits}")
        if self.with_residual:
            parts.append(f"residual(downsample={self.residual_downsample})")
        else:
            parts.append("no_residual")
        parts.append(f"audio={self.audio_bandwidth}kbps")
        if self.lossless_bypass:
            parts.append("lossless")
        return ", ".join(parts)


class RateController:

    @classmethod
    def get_params(cls, target_percent: float, base_audio_bandwidth: float = 3.0) -> RateParams:
        """
        Map a target size percentage to a RateParams instance.

        Tiers:
          < 1%:    quality=1, turbo, norm_bits=2, dir_bits=2, no residuals, audio=1.5
          1–3%:    quality=2, turbo, norm_bits=3, dir_bits=3, residual ds=4, audio=3.0
          3–8%:    quality=3, turbo, norm_bits=4, dir_bits=4, residual ds=2, audio=6.0
          8–15%:   quality=5, no turbo, norm_bits=4, dir_bits=4, residual ds=1, audio=12.0
          15–89%:  quality=8, no turbo, norm_bits=8, dir_bits=8, residual ds=1, audio=24.0
          >= 90%:  quality=8, no turbo, lossless bypass, residual ds=1, audio=24.0
        """
        if target_percent < 1.0:
            return RateParams(
                quality=1,
                use_turbo=False,
                norm_bits=3,
                dir_bits=3,
                with_residual=False,
                residual_downsample=4,
                audio_bandwidth=1.5,
            )
        elif target_percent < 3.0:
            return RateParams(
                quality=2,
                use_turbo=False,
                norm_bits=3,
                dir_bits=3,
                with_residual=True,
                residual_downsample=4,
                audio_bandwidth=3.0,
            )
        elif target_percent < 8.0:
            return RateParams(
                quality=3,
                use_turbo=False,
                norm_bits=4,
                dir_bits=4,
                with_residual=True,
                residual_downsample=2,
                audio_bandwidth=6.0,
            )
        elif target_percent < 15.0:
            return RateParams(
                quality=5,
                use_turbo=False,
                norm_bits=6,
                dir_bits=6,
                with_residual=True,
                residual_downsample=1,
                audio_bandwidth=12.0,
            )
        elif target_percent < 90.0:
            return RateParams(
                quality=5,
                use_turbo=False,
                norm_bits=8,
                dir_bits=8,
                with_residual=True,
                residual_downsample=1,
                audio_bandwidth=24.0,
            )
        else:
            return RateParams(
                quality=5,
                use_turbo=False,
                norm_bits=8,
                dir_bits=8,
                with_residual=True,
                residual_downsample=1,
                audio_bandwidth=24.0,
                lossless_bypass=True,
            )

    @classmethod
    def target_bytes(cls, original_bytes: int, target_percent: float) -> int:
        """Compute the target byte count given original size and target percentage."""
        return int(original_bytes * target_percent / 100.0)

    @classmethod
    def describe_params(cls, params: RateParams) -> str:
        """Return a human-readable description of a RateParams instance."""
        return params.describe()
