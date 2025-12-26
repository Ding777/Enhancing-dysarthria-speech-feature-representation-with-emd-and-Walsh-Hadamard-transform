"""
emd_fwht.py
Contains EMD-based IMF extraction (PyEMD optional fallback) and FWHT + vector statistics.
"""
import math
import numpy as np
from scipy import linalg, stats, signal
# Try to import PyEMD
try:
    from PyEMD import EMD
    HAVE_EMD = True
except Exception:
    HAVE_EMD = False

def compute_imfs(x):
    """
    Compute IMFs using PyEMD if available; otherwise fallback to coarse window decomposition.
    Returns array of IMFs (n_imfs, n_samples) - imfs may be zero-padded to same length.
    """
    x = np.asarray(x, dtype=float)
    n = len(x)
    if HAVE_EMD:
        emd = EMD()
        imfs = emd(x)
        # If only 1 IMF returned as 1D, ensure shape (k,n)
        if imfs.ndim == 1:
            imfs = imfs[np.newaxis, :]
        # Some IMFs might be shorter/padded already by PyEMD; ensure consistent shape
        # (PyEMD returns IMFs each same length as input typically)
        return imfs
    else:
        # fallback: split signal into 6 windows as pseudo-IMFs
        chunks = np.array_split(x, 6)
        # pad each chunk to full length (so downstream stats use same length)
        imfs = [np.pad(c, (0, n - len(c)), mode='constant') for c in chunks if len(c) > 0]
        return np.vstack(imfs)

def stats_of_vector(v):
    """
    Compute a small set of statistics for vector v.
    Returns list: [mean, std, skew, kurtosis, energy, zero_crossings, entropy]
    """
    v = np.asarray(v, dtype=float)
    if v.size == 0:
        return [0.0]*7
    mean = float(np.mean(v))
    std = float(np.std(v))
    sk = float(stats.skew(v)) if v.size > 2 else 0.0
    kt = float(stats.kurtosis(v)) if v.size > 3 else 0.0
    energy = float(np.sum(v**2))
    zc = float(((v[:-1] * v[1:]) < 0).sum()) if v.size > 1 else 0.0
    p = np.abs(v)
    psum = p.sum() + 1e-12
    p_norm = p / psum
    entropy = float(-np.sum(p_norm * np.log(p_norm + 1e-12)))
    return [mean, std, sk, kt, energy, zc, entropy]

def fwht(x):
    """
    Fast Walsh-Hadamard Transform via Hadamard matrix product.
    Pads vector to next power-of-two.
    Returns transformed 1D numpy array.
    """
    x = np.asarray(x, dtype=float)
    if x.size == 0:
        return x
    n = x.size
    m = 1 << (n - 1).bit_length()
    if m != n:
        x = np.concatenate([x, np.zeros(m - n)])
    H = linalg.hadamard(x.size)
    return (H @ x) / math.sqrt(x.size)

def psd_stats(y, sr, nperseg=1024):
    """
    Compute log-energy in predefined bands using Welch PSD.
    Returns list of 4 band log energies.
    """
    f, Pxx = signal.welch(y, fs=sr, nperseg=min(nperseg, len(y)))
    bands = [(0, 500), (500, 1500), (1500, 4000), (4000, sr//2)]
    feats = []
    for a, b in bands:
        mask = (f >= a) & (f < b)
        val = float(np.log1p(Pxx[mask].sum())) if mask.any() else 0.0
        feats.append(val)
    return feats
