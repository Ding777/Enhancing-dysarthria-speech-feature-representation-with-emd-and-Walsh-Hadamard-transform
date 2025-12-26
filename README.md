# Enhancing-dysarthria-speech-feature-representation-with-emd-and-Walsh-Hadamard-transform
Speech recognition Speech signal processing 


# WHFEMD Pipeline (modular project)

A modular implementation of the WHFEMD / WHGFCC-style pipeline (EMD + FWHT + GFCC/MFCC features) with PCA + SMOTE (or fallback) + GMM membership augmentation and classification.


## Repo layout

whfemd-pipeline/
├─ audio_io.py # audio loading + demo synthetic signal
├─ emd_fwht.py # EMD & FWHT + vector statistics
├─ gfcc.py # GFCC/MFCC extraction (optional libs)
├─ features.py # high-level feature extraction
├─ oversample.py # SMOTE wrapper with fallback random oversampling
├─ train.py # PCA+SMOTE+GMM pipeline + train/eval
├─ cli.py # command-line entry point
├─ requirements.txt
└─ README.md


## Install

Create virtual environment and install:

```bash
python -m venv venv
source venv/bin/activate   # or venv\Scripts\activate on Windows
pip install -r requirements.txt
# For full fidelity, also:
pip install PyEMD gammatone imbalanced-learn librosa

Run

Demo (no external data required):
python cli.py --mode demo

Run with a CSV of audio filepaths and labels (CSV must have columns filepath,label):
python cli.py --mode run --csv /path/to/labels.csv --outdir results

The script will:

extract WHFEMD/WHGFCC features,

run PCA,

oversample (SMOTE if installed else random replicate),

fit GMM to get membership probabilities,

append membership features and train classifiers,

print baseline and augmented model metrics and save results_summary.csv in --outdir.

Notes

Optional packages (PyEMD, gammatone, librosa, imbalanced-learn) give a more faithful reproduction. The code contains fallbacks so it still runs without them.

To adapt the feature extractor to exact paper settings (filterbanks, frame sizes), edit gfcc.py/features.py.

For large datasets, consider parallelizing feature extraction or saving intermediate feature matrices to disk.

License / Citation

This project reproduces algorithmic ideas from the WHFEMD/WHGFCC paper (arXiv); adapt and cite the original work if you publish results.





