"""
TurboMath: mathematical primitives for the Turbo compression pipeline.

Components:
  1. Seeded Householder Rotation  — random orthogonal rotation of channel vectors
  2. PolarQuant                   — polar-coordinate quantization (norm + direction)
  3. Lloyd-Max Codebook           — optimal quantizer for folded-Gaussian distribution
  4. QJL Correction               — 1-bit Johnson-Lindenstrauss residual correction
"""

import struct
import numpy as np
import torch
import torch.nn.functional as F
from scipy import stats


class TurboMath:

    # ── 1. Seeded Householder Rotation ────────────────────────────────────────

    @staticmethod
    def _get_rotation_matrix(C: int, seed: int) -> torch.Tensor:
        """Generate a deterministic C×C orthogonal matrix from seed via QR."""
        rng = np.random.RandomState(seed)
        A = rng.randn(C, C).astype(np.float32)
        Q, _ = np.linalg.qr(A)
        return torch.from_numpy(Q)  # shape (C, C), Q^T Q = I

    @staticmethod
    def rotate(tensor: torch.Tensor, seed: int) -> torch.Tensor:
        """
        Apply seeded orthogonal rotation to the channel dimension.

        Parameters
        ----------
        tensor : (1, C, H, W) float32
        seed   : int — determines the rotation matrix

        Returns
        -------
        (1, C, H, W) float32 — rotated tensor
        """
        C = tensor.shape[1]
        Q = TurboMath._get_rotation_matrix(C, seed).to(tensor.device)
        # Reshape to (N, C), apply Q^T (rotate), reshape back
        shape = tensor.shape
        x = tensor.permute(0, 2, 3, 1).reshape(-1, C)  # (H*W, C)
        x_rot = x @ Q.T                                  # (H*W, C)
        return x_rot.reshape(shape[0], shape[2], shape[3], C).permute(0, 3, 1, 2)

    @staticmethod
    def rotate_inverse(tensor_rot: torch.Tensor, seed: int) -> torch.Tensor:
        """
        Inverse of rotate() — apply Q (transpose of Q^T).

        Parameters
        ----------
        tensor_rot : (1, C, H, W) float32 — rotated tensor
        seed       : int — same seed used in rotate()

        Returns
        -------
        (1, C, H, W) float32 — unrotated tensor
        """
        C = tensor_rot.shape[1]
        Q = TurboMath._get_rotation_matrix(C, seed).to(tensor_rot.device)
        shape = tensor_rot.shape
        x = tensor_rot.permute(0, 2, 3, 1).reshape(-1, C)
        x_orig = x @ Q   # Q is its own inverse for orthogonal matrices: (Q^T)^T = Q
        return x_orig.reshape(shape[0], shape[2], shape[3], C).permute(0, 3, 1, 2)

    # ── 2. PolarQuant ─────────────────────────────────────────────────────────

    @staticmethod
    def lloyd_max_codebook(n_levels: int, dist: str = "halfnorm") -> tuple[np.ndarray, np.ndarray]:
        """
        Compute Lloyd-Max optimal quantizer thresholds and centroids
        for a half-normal distribution (distribution of L2 norms after rotation).

        Parameters
        ----------
        n_levels : number of quantization levels (power of 2)
        dist     : 'halfnorm' for folded Gaussian

        Returns
        -------
        (thresholds, centroids) — arrays of length n_levels-1 and n_levels
        """
        if dist == "halfnorm":
            pdf = stats.halfnorm.pdf
            cdf = stats.halfnorm.cdf
            ppf = stats.halfnorm.ppf
        else:
            raise ValueError(f"Unknown dist: {dist!r}")

        # Initialize centroids via quantile spacing
        quantiles = np.linspace(0.01, 0.99, n_levels)
        centroids = ppf(quantiles)

        # Iterate Lloyd-Max: alternate between optimal thresholds and centroids
        for _ in range(100):
            # Thresholds = midpoints between centroids
            thresholds = (centroids[:-1] + centroids[1:]) / 2
            thresholds = np.concatenate([[0.0], thresholds, [np.inf]])

            # Centroids = E[x | threshold_i < x <= threshold_{i+1}]
            new_centroids = np.zeros(n_levels)
            for k in range(n_levels):
                lo, hi = thresholds[k], thresholds[k + 1]
                # E[x | lo < x < hi] = integral(x * pdf(x)) / integral(pdf(x))
                hi_clip = min(hi, 20.0)  # avoid numerical issues at inf
                if pdf(lo) < 1e-10 and pdf(hi_clip) < 1e-10:
                    new_centroids[k] = centroids[k]
                    continue
                # Approximate via midpoint quadrature
                xs = np.linspace(lo, hi_clip, 200)
                weights = pdf(xs)
                if weights.sum() < 1e-12:
                    new_centroids[k] = centroids[k]
                else:
                    new_centroids[k] = np.dot(xs, weights) / weights.sum()

            if np.max(np.abs(new_centroids - centroids)) < 1e-6:
                break
            centroids = new_centroids

        thresholds = (centroids[:-1] + centroids[1:]) / 2
        return thresholds, centroids

    @staticmethod
    def polar_quantize(
        tensor: torch.Tensor,
        norm_bits: int = 3,
        dir_bits: int = 3,
    ) -> tuple[torch.Tensor, dict]:
        """
        Quantize a rotated latent tensor using polar decomposition.

        Strategy:
          - Compute per-spatial-position L2 norm of the channel vector
          - Normalize to unit sphere
          - Quantize norms with Lloyd-Max (3 bits)
          - Quantize normalized direction with uniform quant (3 bits per component)

        Parameters
        ----------
        tensor    : (1, C, H, W) float32 — rotated latent
        norm_bits : bits for norm quantization (Lloyd-Max)
        dir_bits  : bits for direction quantization (uniform, signed)

        Returns
        -------
        (quantized_tensor, codebook_dict)
          quantized_tensor : (1, C, H, W) float32 — dequantized reconstruction
          codebook_dict    : {'norm_centroids': array, 'dir_scale': float, 'norm_bits': int, 'dir_bits': int}
        """
        C = tensor.shape[1]
        shape = tensor.shape
        # Reshape to (N, C)
        x = tensor.permute(0, 2, 3, 1).reshape(-1, C).float()
        N = x.shape[0]

        # --- Norm quantization ---
        norms = x.norm(dim=1, keepdim=True)  # (N, 1)
        norms_np = norms.squeeze(1).cpu().numpy()

        n_levels = 2 ** norm_bits
        thresholds, centroids = TurboMath.lloyd_max_codebook(n_levels, dist="halfnorm")

        # Assign each norm to nearest centroid
        norm_indices = np.searchsorted(thresholds, norms_np)  # (N,)
        norm_q = torch.from_numpy(centroids[norm_indices]).float().unsqueeze(1)  # (N, 1)

        # --- Direction quantization (uniform on [-1, 1]) ---
        eps = 1e-8
        direction = x / (norms + eps)  # unit vectors (N, C)

        # Uniform symmetric quantization for direction
        dir_levels = 2 ** dir_bits
        dir_half = dir_levels // 2
        dir_indices = (direction.clamp(-1, 1) * dir_half).round().clamp(-dir_half, dir_half - 1).to(torch.int8)
        dir_q = dir_indices.float() / dir_half  # dequantize

        # --- Reconstruct ---
        x_q = dir_q * norm_q  # (N, C)
        quantized = x_q.reshape(shape[0], shape[2], shape[3], C).permute(0, 3, 1, 2)

        codebook = {
            "norm_centroids": centroids,
            "norm_thresholds": thresholds,
            "dir_scale": dir_half,
            "norm_bits": norm_bits,
            "dir_bits": dir_bits,
        }
        return quantized, codebook

    # Number of "anchor" channels kept at 8-bit int8 precision (immune to PolarQuant)
    ANCHOR_CHANNELS = 24

    @staticmethod
    def polar_quantize_encode(
        tensor: torch.Tensor,
        norm_bits: int = 3,
        dir_bits: int = 3,
    ) -> tuple[bytes, torch.Tensor]:
        """
        Encode tensor to compact bytes using PolarQuant with Anchor Channels.

        The first ANCHOR_CHANNELS channels are preserved at 8-bit int8 precision
        (standard per-channel quantization). Only the remaining channels go through
        aggressive PolarQuant. This protects the most information-dense channels
        from quantization collapse.

        Returns
        -------
        (payload_bytes, reconstructed_tensor)
        """
        import zstandard as zstd
        C = tensor.shape[1]
        shape = tensor.shape
        anchor_c = min(TurboMath.ANCHOR_CHANNELS, C)

        # --- Anchor channels: 8-bit int8 per-channel quantization ---
        anchor_tensor = tensor[:, :anchor_c, :, :]  # (1, anchor_c, H, W)
        anchor_flat = anchor_tensor.permute(0, 2, 3, 1).reshape(-1, anchor_c).float()
        # Per-channel absmax scaling to int8 range [-127, 127]
        absmax = anchor_flat.abs().max(dim=0).values.clamp(min=1e-8)  # (anchor_c,)
        anchor_scales = absmax / 127.0  # (anchor_c,)
        anchor_q = (anchor_flat / anchor_scales.unsqueeze(0)).round().clamp(-127, 127).to(torch.int8)
        anchor_deq = anchor_q.float() * anchor_scales.unsqueeze(0)
        anchor_rec = anchor_deq.reshape(shape[0], shape[2], shape[3], anchor_c).permute(0, 3, 1, 2)

        # --- Remaining channels: PolarQuant ---
        polar_c = C - anchor_c
        if polar_c > 0:
            polar_tensor = tensor[:, anchor_c:, :, :]
            x = polar_tensor.permute(0, 2, 3, 1).reshape(-1, polar_c).float()
            N = x.shape[0]

            norms = x.norm(dim=1)
            norms_np = norms.cpu().numpy()

            n_levels = 2 ** norm_bits
            thresholds, centroids = TurboMath.lloyd_max_codebook(n_levels)
            norm_indices = np.searchsorted(thresholds, norms_np).astype(np.uint8)
            norm_q = torch.from_numpy(centroids[norm_indices]).float()

            eps = 1e-8
            direction = x / (norms.unsqueeze(1) + eps)
            dir_half = 2 ** (dir_bits - 1)
            dir_indices = (direction.clamp(-1, 1) * dir_half).round().clamp(-dir_half, dir_half - 1).to(torch.int8)
            dir_q = dir_indices.float() / dir_half

            x_q = dir_q * norm_q.unsqueeze(1)
            polar_rec = x_q.reshape(shape[0], shape[2], shape[3], polar_c).permute(0, 3, 1, 2)
        else:
            N = anchor_flat.shape[0]
            polar_rec = tensor[:, anchor_c:, :, :]
            norm_indices = np.array([], dtype=np.uint8)
            dir_indices = torch.tensor([], dtype=torch.int8)

        # Concatenate reconstructed anchor + polar
        reconstructed = torch.cat([anchor_rec, polar_rec], dim=1)

        # --- Pack payload ---
        # Anchor data: scales (anchor_c * float32) + quantized int8 (N * anchor_c)
        anchor_scales_bytes = anchor_scales.cpu().numpy().astype(np.float32).tobytes()
        anchor_q_bytes = anchor_q.cpu().numpy().tobytes()

        # Polar data: norm indices + dir indices (compressed together)
        if polar_c > 0:
            polar_raw = norm_indices.tobytes() + dir_indices.cpu().numpy().tobytes()
        else:
            polar_raw = b""
        compressed = zstd.ZstdCompressor(level=19).compress(
            anchor_scales_bytes + anchor_q_bytes + polar_raw
        )

        # Header: [B norm_bits][B dir_bits][I N][H C][H anchor_c][I compressed_len]
        payload = struct.pack("<BBIHHI", norm_bits, dir_bits, N, C, anchor_c, len(compressed)) + compressed
        return payload, reconstructed

    @staticmethod
    def polar_quantize_decode(payload: bytes, shape: tuple) -> torch.Tensor:
        """Decode PolarQuant payload back to tensor (with anchor channel support)."""
        import zstandard as zstd
        offset = 0

        # Detect format: new format has 6-field header (with anchor_c), old has 5
        # New: [B norm_bits][B dir_bits][I N][H C][H anchor_c][I comp_len]
        # Old: [B norm_bits][B dir_bits][I N][H C][H comp_len_as_H]  — but comp_len was packed as H (2 bytes)
        # We distinguish by checking if the header size matches expectations
        # Old format: struct "<BBIHH" = 1+1+4+2+2 = 10 bytes
        # New format: struct "<BBIHHI" = 1+1+4+2+2+4 = 14 bytes
        # Try to detect: in old format, bytes 8-10 are comp_len (uint16).
        # In new format, bytes 8-10 are anchor_c (uint16) and bytes 10-14 are comp_len (uint32).
        # If anchor_c == 0 or > C, it's likely old format.
        norm_bits, dir_bits, N, C = struct.unpack_from("<BBIH", payload, offset)
        offset_test = 1 + 1 + 4 + 2
        (test_val,) = struct.unpack_from("<H", payload, offset_test)

        # Heuristic: if test_val <= C and test_val > 0, it's likely anchor_c (new format)
        # Old format has comp_len as uint16 which is typically large (>100)
        # New format has anchor_c which is 0-24
        # More reliable: try new format first, check if comp_len makes sense
        if test_val <= TurboMath.ANCHOR_CHANNELS:
            # New format with anchor channels
            norm_bits, dir_bits, N, C, anchor_c, comp_len = struct.unpack_from("<BBIHHI", payload, 0)
            offset = struct.calcsize("<BBIHHI")
            compressed = payload[offset: offset + comp_len]
            raw = zstd.ZstdDecompressor().decompress(compressed)

            polar_c = C - anchor_c
            raw_offset = 0

            # Anchor: scales (anchor_c * float32) + int8 (N * anchor_c)
            anchor_scales = np.frombuffer(raw[raw_offset: raw_offset + anchor_c * 4], dtype=np.float32).copy()
            raw_offset += anchor_c * 4
            anchor_q = np.frombuffer(raw[raw_offset: raw_offset + N * anchor_c], dtype=np.int8).reshape(N, anchor_c).copy()
            raw_offset += N * anchor_c
            anchor_deq = torch.from_numpy(anchor_q).float() * torch.from_numpy(anchor_scales).unsqueeze(0)
            anchor_rec = anchor_deq.reshape(shape[0], shape[2], shape[3], anchor_c).permute(0, 3, 1, 2)

            if polar_c > 0:
                # Polar: norm indices (N) + dir indices (N * polar_c)
                norm_indices = np.frombuffer(raw[raw_offset: raw_offset + N], dtype=np.uint8).copy()
                raw_offset += N
                dir_indices = np.frombuffer(raw[raw_offset:], dtype=np.int8).reshape(N, polar_c).copy()

                n_levels = 2 ** norm_bits
                thresholds, centroids = TurboMath.lloyd_max_codebook(n_levels)
                norm_q = torch.from_numpy(centroids[norm_indices]).float()
                dir_half = 2 ** (dir_bits - 1)
                dir_q = torch.from_numpy(dir_indices).float() / dir_half
                x_q = dir_q * norm_q.unsqueeze(1)
                polar_rec = x_q.reshape(shape[0], shape[2], shape[3], polar_c).permute(0, 3, 1, 2)
            else:
                polar_rec = torch.zeros(shape[0], 0, shape[2], shape[3])

            return torch.cat([anchor_rec, polar_rec], dim=1)
        else:
            # Old format (no anchor channels)
            norm_bits, dir_bits, N, C, comp_len = struct.unpack_from("<BBIHH", payload, 0)
            offset = struct.calcsize("<BBIHH")
            compressed = payload[offset: offset + comp_len]

            raw = zstd.ZstdDecompressor().decompress(compressed)
            n_levels = 2 ** norm_bits
            thresholds, centroids = TurboMath.lloyd_max_codebook(n_levels)

            norm_indices = np.frombuffer(raw[:N], dtype=np.uint8).copy()
            dir_indices = np.frombuffer(raw[N:], dtype=np.int8).reshape(N, C).copy()

            norm_q = torch.from_numpy(centroids[norm_indices]).float()
            dir_half = 2 ** (dir_bits - 1)
            dir_q = torch.from_numpy(dir_indices).float() / dir_half

            x_q = dir_q * norm_q.unsqueeze(1)
            return x_q.reshape(shape[0], shape[2], shape[3], C).permute(0, 3, 1, 2)

    # ── 4. QJL 1-bit Residual Correction ─────────────────────────────────────

    @staticmethod
    def qjl_project(residual: torch.Tensor, seed: int, n_projections: int) -> bytes:
        """
        Compute 1-bit QJL projection of the residual error.

        Projects residual onto n_projections random ±1 vectors (seeded),
        then stores only the sign of each dot product (1 bit each).

        Parameters
        ----------
        residual      : (1, C, H, W) float32 — quantization error
        seed          : int
        n_projections : int — number of 1-bit measurements

        Returns
        -------
        bytes — packed sign bits (np.packbits output)
        """
        C = residual.shape[1]
        shape = residual.shape
        x = residual.permute(0, 2, 3, 1).reshape(-1, C).float()  # (N, C)
        N = x.shape[0]

        rng = np.random.RandomState(seed ^ 0xDEADBEEF)
        # Random ±1 projection matrix: (C, n_projections)
        P = (rng.randint(0, 2, size=(C, n_projections)) * 2 - 1).astype(np.float32)
        P_t = torch.from_numpy(P).to(residual.device)

        projections = x @ P_t  # (N, n_projections)
        signs = (projections >= 0).cpu().numpy().reshape(-1)
        return np.packbits(signs).tobytes()

    @staticmethod
    def qjl_reconstruct(
        signs_bytes: bytes,
        seed: int,
        n_projections: int,
        shape: tuple,
        scale: float = 0.1,
    ) -> torch.Tensor:
        """
        Reconstruct a correction tensor from 1-bit QJL signs.

        Parameters
        ----------
        signs_bytes   : bytes from qjl_project()
        seed          : int — same seed as qjl_project()
        n_projections : int — same as qjl_project()
        shape         : (1, C, H, W) target shape
        scale         : amplitude of correction (tuned empirically)

        Returns
        -------
        (1, C, H, W) float32 — correction to add to quantized reconstruction
        """
        C = shape[1]
        N = shape[0] * shape[2] * shape[3]

        signs_np = np.unpackbits(np.frombuffer(signs_bytes, dtype=np.uint8))
        signs_np = signs_np[:N * n_projections].reshape(N, n_projections).astype(np.float32)
        signs_np = signs_np * 2 - 1  # {0,1} → {-1,+1}

        rng = np.random.RandomState(seed ^ 0xDEADBEEF)
        P = (rng.randint(0, 2, size=(C, n_projections)) * 2 - 1).astype(np.float32)

        signs_t = torch.from_numpy(signs_np)
        P_t = torch.from_numpy(P)

        # Correction = signs @ P^T / n_projections * scale
        correction = signs_t @ P_t.T / n_projections * scale  # (N, C)
        return correction.reshape(shape[0], shape[2], shape[3], C).permute(0, 3, 1, 2)
