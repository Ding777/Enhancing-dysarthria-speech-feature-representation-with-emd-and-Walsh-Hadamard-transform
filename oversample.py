"""
oversample.py
SMOTE wrapper with fallback random oversampling.
"""
import numpy as np
try:
    from imblearn.over_sampling import SMOTE
    HAVE_SMOTE = True
except Exception:
    HAVE_SMOTE = False

def simple_random_oversample(X, y, seed=42):
    """
    Random oversample minority classes to match majority class count.
    Returns Xo, yo (shuffled).
    """
    rng = np.random.RandomState(seed)
    unique, counts = np.unique(y, return_counts=True)
    max_count = counts.max()
    X_parts = []
    y_parts = []
    for cls in unique:
        Xc = X[y == cls]
        nc = len(Xc)
        repeats = max_count // nc
        rem = max_count % nc
        parts = [Xc] * repeats
        if rem:
            idx = rng.choice(nc, rem, replace=True)
            parts.append(Xc[idx])
        X_rep = np.vstack(parts) if parts else np.empty((0, X.shape[1]))
        X_parts.append(X_rep)
        y_parts.append(np.full(X_rep.shape[0], cls))
    Xo = np.vstack(X_parts)
    yo = np.concatenate(y_parts)
    perm = rng.permutation(len(yo))
    return Xo[perm], yo[perm]

def smote_or_fallback(X, y, seed=42):
    """
    If imblearn.SMOTE is available, use it. Otherwise use simple_random_oversample.
    """
    if HAVE_SMOTE:
        sm = SMOTE(random_state=seed)
        Xo, yo = sm.fit_resample(X, y)
        used_smote = True
    else:
        Xo, yo = simple_random_oversample(X, y, seed=seed)
        used_smote = False
    return Xo, yo, used_smote
