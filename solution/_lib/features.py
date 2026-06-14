"""
101-dim engineered feature extraction for the real vs AI-generated classifier.

Feature layout (N x 101):
  RGB mean (3), RGB std (3), RGB histogram 8-bin (24),
  Laplacian variance (1), Sobel std (1), FFT HF ratio (1),
  FFT band ratios (4), patch stats 3x3 grid mean+std (54),
  noise std (3), channel skewness (3), chroma noise Cb/Cr (2),
  JPEG block score h+v (2)

Performance:
  Chunks are processed in parallel (joblib threads, n_jobs=4). FFT is computed
  once per chunk and reused for both the HF ratio and band ratio features,
  eliminating the duplicate fft2 call from the development version.
  Images are always 224x224 (output of clean_image), so the frequency grid is
  precomputed once at module load time.
"""

from __future__ import annotations

import os

import numpy as np

# Use all available cores up to the grader's --cpus 8 for the parallel chunk workers.
N_JOBS = min(8, os.cpu_count() or 1)

# Frequency grid for 224x224 images -- precomputed once, reused by all chunks.
_IMG = 224
_fft_yy, _fft_xx = np.meshgrid(
    np.fft.fftfreq(_IMG), np.fft.fftfreq(_IMG), indexing="ij"
)
_fft_r      = np.sqrt(_fft_yy ** 2 + _fft_xx ** 2)
_HF_MASK    = _fft_r > 0.25
_BAND_MASKS = [
    (_fft_r >= lo) & (_fft_r < hi)
    for lo, hi in ((0.0, 0.05), (0.05, 0.15), (0.15, 0.30), (0.30, 0.50))
]


# ---------------------------------------------------------------------------
# Helper functions (operate on float32 arrays)
# ---------------------------------------------------------------------------

def _gray(X_f: np.ndarray) -> np.ndarray:
    return 0.299 * X_f[..., 0] + 0.587 * X_f[..., 1] + 0.114 * X_f[..., 2]


def _lap_var(g: np.ndarray) -> np.ndarray:
    lap = (-4 * g[:, 1:-1, 1:-1]
           + g[:, :-2, 1:-1] + g[:, 2:, 1:-1]
           + g[:, 1:-1, :-2] + g[:, 1:-1, 2:])
    return lap.reshape(len(g), -1).var(axis=1)


def _sobel_std(g: np.ndarray) -> np.ndarray:
    gx = g[:, 1:-1, 2:] - g[:, 1:-1, :-2]
    gy = g[:, 2:, 1:-1] - g[:, :-2, 1:-1]
    return np.sqrt(gx * gx + gy * gy).reshape(len(g), -1).std(axis=1)


def _patch_stats(X: np.ndarray, grid: int = 3) -> np.ndarray:
    n, h, w, c = X.shape
    ph, pw = h // grid, w // grid
    out = []
    for ri in range(grid):
        for ci in range(grid):
            patch = X[:, ri*ph:(ri+1)*ph, ci*pw:(ci+1)*pw, :]
            out.append(patch.mean(axis=(1, 2)))
            out.append(patch.std(axis=(1, 2)))
    return np.concatenate(out, axis=1)  # (N, 54)


def _noise_std(X: np.ndarray) -> np.ndarray:
    padded = np.pad(X, ((0, 0), (1, 1), (1, 1), (0, 0)), mode="edge")
    blur = (padded[:, :-2, :-2] + padded[:, :-2, 1:-1] + padded[:, :-2, 2:] +
            padded[:, 1:-1, :-2] + padded[:, 1:-1, 1:-1] + padded[:, 1:-1, 2:] +
            padded[:, 2:, :-2] + padded[:, 2:, 1:-1] + padded[:, 2:, 2:]) / 9.0
    return (X - blur).reshape(len(X), -1, 3).std(axis=1)  # (N, 3)


def _channel_skewness(X: np.ndarray) -> np.ndarray:
    """Per-channel skewness. AI images tend toward symmetric distributions."""
    mu = X.mean(axis=(1, 2), keepdims=True)
    sigma = X.std(axis=(1, 2), keepdims=True) + 1e-8
    return (((X - mu) / sigma) ** 3).mean(axis=(1, 2))  # (N, 3)


def _chroma_noise(X: np.ndarray) -> np.ndarray:
    """Box-blur residual std in Cb and Cr channels."""
    cb = -0.16874 * X[..., 0] - 0.33126 * X[..., 1] + 0.5     * X[..., 2]
    cr =  0.5     * X[..., 0] - 0.41869 * X[..., 1] - 0.08131 * X[..., 2]
    out = []
    for ch in (cb, cr):
        padded = np.pad(ch, ((0, 0), (1, 1), (1, 1)), mode="edge")
        blur = (padded[:, :-2, :-2] + padded[:, :-2, 1:-1] + padded[:, :-2, 2:] +
                padded[:, 1:-1, :-2] + padded[:, 1:-1, 1:-1] + padded[:, 1:-1, 2:] +
                padded[:, 2:, :-2] + padded[:, 2:, 1:-1] + padded[:, 2:, 2:]) / 9.0
        out.append((ch - blur).reshape(len(X), -1).std(axis=1))
    return np.stack(out, axis=1)  # (N, 2)


def _jpeg_block_score(g: np.ndarray, block_size: int = 8) -> np.ndarray:
    """Mean absolute discontinuity at JPEG DCT block boundaries."""
    n_h = (g.shape[2] // block_size) - 1
    right_cols = np.arange(block_size, block_size + n_h * block_size, block_size)
    h_diff = np.abs(g[:, :, right_cols] - g[:, :, right_cols - 1]).mean(axis=(1, 2))

    n_v = (g.shape[1] // block_size) - 1
    bottom_rows = np.arange(block_size, block_size + n_v * block_size, block_size)
    v_diff = np.abs(g[:, bottom_rows, :] - g[:, bottom_rows - 1, :]).mean(axis=(1, 2))

    return np.stack([h_diff, v_diff], axis=1)  # (N, 2)


# ---------------------------------------------------------------------------
# Chunk worker (called in parallel)
# ---------------------------------------------------------------------------

def _process_chunk(chunk_u8: np.ndarray) -> np.ndarray:
    """(B, 224, 224, 3) uint8 -> (B, 101) float32. Single FFT per chunk."""
    X = chunk_u8.astype(np.float32) / 255.0

    mean = X.mean(axis=(1, 2))
    std  = X.std(axis=(1, 2))
    hist = np.zeros((X.shape[0], 24), dtype=np.float32)
    edges = np.linspace(0, 1, 9)
    for c in range(3):
        for b in range(8):
            hist[:, c * 8 + b] = (
                (X[..., c] >= edges[b]) & (X[..., c] < edges[b + 1])
            ).mean(axis=(1, 2))

    g   = _gray(X)
    lap = _lap_var(g)
    sob = _sobel_std(g)

    # Compute FFT once, reuse for hf_ratio and band_ratios
    fft_p   = np.abs(np.fft.fft2(g)) ** 2
    fft_tot = fft_p.reshape(len(g), -1).sum(axis=1) + 1e-9
    hf    = (fft_p * _HF_MASK).reshape(len(g), -1).sum(axis=1) / fft_tot
    bands = np.stack(
        [(fft_p * m).reshape(len(g), -1).sum(axis=1) / fft_tot for m in _BAND_MASKS],
        axis=1,
    )

    patch  = _patch_stats(X)
    noise  = _noise_std(X)
    skew   = _channel_skewness(X)
    chroma = _chroma_noise(X)
    block  = _jpeg_block_score(g)

    return np.concatenate(
        [mean, std, hist, lap[:, None], sob[:, None], hf[:, None],
         bands, patch, noise, skew, chroma, block], axis=1
    )


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def features_from_uint8(X_u8: np.ndarray, chunk: int = 256, n_jobs: int | None = None) -> np.ndarray:
    """(N, H, W, 3) uint8 -> (N, 101) float32.

    Chunks are processed in parallel (threads) to reduce wall time. Defaults to all
    available cores (capped at the grader's --cpus 8). The numpy ufuncs + FFT release
    the GIL, so thread workers give near-linear speedup. Output is independent of n_jobs
    (chunks are concatenated in order), so this only changes wall time, not the features.
    """
    from joblib import Parallel, delayed

    if n_jobs is None:
        n_jobs = N_JOBS
    starts = list(range(0, len(X_u8), chunk))
    results = Parallel(n_jobs=n_jobs, prefer="threads")(
        delayed(_process_chunk)(np.array(X_u8[s:s + chunk])) for s in starts
    )
    return np.concatenate(results, axis=0)


def features_single(arr_uint8: np.ndarray) -> np.ndarray:
    """(H, W, 3) uint8 -> (101,) float32. For single-image inference in predict.py."""
    return _process_chunk(arr_uint8[None])[0]
