"""
gfcc.py
GFCC extraction: uses gammatone.gtgram if available; otherwise falls back to librosa MFCC.
"""
import numpy as np
from scipy import signal
# try optional libs
try:
    from gammatone.gtgram import gtgram
    HAVE_GAMMATONE = True
except Exception:
    HAVE_GAMMATONE = False

try:
    import librosa
    HAVE_LIBROSA = True
except Exception:
    HAVE_LIBROSA = False

def extract_gfcc(y, sr, n_ceps=13, n_filters=40, win_len=0.025, hop=0.01):
    """
    Extract GFCC (approx) features.
    Returns (ceps_matrix: frames x n_ceps, frame_energies)
    """
    if HAVE_GAMMATONE:
        # gtgram(y, fs, window_time, hop_time, channels, fmin)
        gt = gtgram(y, sr, win_len, hop, n_filters, 50)
        logE = np.log1p(np.abs(gt) + 1e-12)
        # simple DCT across filterbank axis to form cepstral-like coefficients
        # Using scipy.signal.dct not imported here—use numpy.fft for simple transform or use librosa fallback
        # We'll compute simple SVD-based dimensionality reduction as approximation
        U, S, Vt = np.linalg.svd(logE, full_matrices=False)
        ceps = (U[:, :n_ceps] * S[:n_ceps]).astype(float)  # frames x n_ceps
        energies = logE.sum(axis=0)
        return ceps, energies
    else:
        if HAVE_LIBROSA:
            n_fft = int(win_len * sr)
            hop_len = int(hop * sr)
            ceps = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=n_ceps, n_mels=n_filters, n_fft=n_fft, hop_length=hop_len).T
            S = librosa.feature.rms(y=y, frame_length=n_fft, hop_length=hop_len).flatten()
            return ceps, S
        else:
            # fallback - return empty to signal caller to use alternative
            raise RuntimeError("No gammatone or librosa available for GFCC/MFCC extraction.")
