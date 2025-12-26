"""
audio_io.py
Audio loading and synthetic demo signal generation.
"""
import os
import numpy as np
import soundfile as sf
from scipy import signal as sp_signal

# Try to import librosa for better resampling (optional)
try:
    import librosa
    HAVE_LIBROSA = True
except Exception:
    HAVE_LIBROSA = False

def load_audio(path, sr=None):
    """
    Load audio with soundfile; optionally resample to sr (Hz).
    Returns: y (1D float numpy), fs (int)
    """
    if not os.path.exists(path):
        raise FileNotFoundError(f"File not found: {path}")
    x, fs = sf.read(path)
    if x.ndim > 1:
        x = x.mean(axis=1)
    if sr is not None and fs != sr:
        if HAVE_LIBROSA:
            x = librosa.resample(x.astype(float), orig_sr=fs, target_sr=sr)
            fs = sr
        else:
            num = int(len(x) * sr / fs)
            x = sp_signal.resample(x, num)
            fs = sr
    return x.astype(float), fs

def demo_synthetic(n_samples=60, sr=16000, duration=1.0):
    """
    Small synthetic dataset generator (sinusoids + noise) for quick demo/testing.
    Returns (X_features, y_labels, raw_signals (optional)).
    For pipeline we will call feature extraction externally.
    """
    X_signals = []
    y = []
    for i in range(n_samples):
        cls = i % 3
        base_freq = [120, 200, 300][cls]
        freq = base_freq + np.random.randn()*5 + np.random.randint(-10,10)
        t = np.linspace(0, duration, int(sr*duration), endpoint=False)
        sig = 0.5*np.sin(2*np.pi*freq*t) + 0.05*np.random.randn(len(t))
        X_signals.append(sig.astype(float))
        y.append(cls)
    return X_signals, np.array(y), sr
