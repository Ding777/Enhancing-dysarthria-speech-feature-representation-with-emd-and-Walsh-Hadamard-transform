# Enhancing-dysarthria-speech-feature-representation-with-emd-and-Walsh-Hadamard-transform
Speech recognition Speech signal processing 

Abstract: Dysarthria speech contains the pathological characteristics of vocal tract and vocal fold, but so far, they have not yet been
included in traditional acoustic feature sets. Moreover, the nonlinearity and non-stationarity of speech have been ignored. In this
paper, we propose a feature enhancement algorithm for dysarthria speech called WHFEMD. It combines empirical mode decomposition (EMD) and fast Walsh-Hadamard transform (FWHT) to enhance features. With the proposed algorithm, the fast Fourier transform of the dysarthria speech is first performed and then followed by EMD to get intrinsic mode functions (IMFs). After that, FWHT
is used to output new coefficients and to extract statistical features based on IMFs, power spectral density, and enhanced gammatone frequency cepstral coefficients. To evaluate the proposed approach, we conducted experiments on two public pathological
speech databases including UA Speech and TORGO. The results show that our algorithm performed better than traditional features
in classification. We achieved improvements of 13.8% (UA Speech) and 3.84% (TORGO), respectively. Furthermore, the incorporation
of an imbalanced classification algorithm to address data imbalance has resulted in a 12.18% increase in recognition accuracy. This
algorithm effectively addresses the challenges of the imbalanced dataset and non-linearity in dysarthric speech and simultaneously
provides a robust representation of the local pathological features of the vocal folds and tracts.

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





