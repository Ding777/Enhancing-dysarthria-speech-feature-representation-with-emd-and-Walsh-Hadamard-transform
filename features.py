"""
features.py
High-level feature extraction: WHFEMD / WHGFCC features built from emd_fwht + gfcc.
"""
import numpy as np
from emd_fwht import compute_imfs, stats_of_vector, fwht, psd_stats
from gfcc import extract_gfcc

def extract_whfemd_features(y, sr, n_gfcc_ceps=13):
    """
    Extract combined features for a single audio signal 'y' sampled at 'sr'.
    Returns 1D numpy feature vector.
    """
    feats = []

    # IMFs
    imfs = compute_imfs(y)
    for imf in imfs:
        feats.extend(stats_of_vector(imf))

    # FWHT stats of IMFs
    for imf in imfs:
        coeffs = fwht(imf)
        feats.extend(stats_of_vector(coeffs))

    # PSD band stats
    feats.extend(psd_stats(y, sr))

    # GFCC / MFCC stats (global)
    try:
        ceps, eng = extract_gfcc(y, sr, n_ceps=n_gfcc_ceps)
        ceps = np.asarray(ceps)
        if ceps.ndim == 1:
            ceps = ceps[np.newaxis, :]
        feats.extend(np.mean(ceps, axis=0).tolist())
        feats.extend(np.std(ceps, axis=0).tolist())
        feats.append(float(np.mean(eng)))
        feats.append(float(np.std(eng)))
    except Exception:
        # fallback simple stats if GFCC unavailable
        feats.append(float(np.mean(np.abs(y))))
        feats.append(float(np.var(y)))

    return np.array(feats, dtype=float)
