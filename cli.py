"""
cli.py
Command-line entry point. Use this to run demo or full CSV-based processing.
"""
import os
import argparse
import pandas as pd
import numpy as np

from audio_io import load_audio, demo_synthetic
from features import extract_whfemd_features
from train import build_and_eval

def prepare_from_csv(csv_path, sr=16000, nmax=None):
    """
    Load audio file paths and labels from CSV and extract features.
    CSV must have columns: filepath,label
    """
    df = pd.read_csv(csv_path)
    if nmax is not None:
        df = df.head(nmax)
    feats = []
    labels = []
    for i, row in df.iterrows():
        fp = row['filepath']
        label = row['label']
        if not os.path.isabs(fp):
            fp = os.path.expanduser(fp)
        try:
            y, fs = load_audio(fp, sr=sr)
        except Exception as e:
            print(f"Skipping {fp}: {e}")
            continue
        fv = extract_whfemd_features(y, fs)
        feats.append(fv)
        labels.append(label)
    if len(feats) == 0:
        raise RuntimeError("No features extracted from CSV dataset.")
    X = np.vstack(feats)
    y = np.array(labels)
    return X, y, df

def main():
    p = argparse.ArgumentParser(description="WHFEMD pipeline CLI")
    p.add_argument("--mode", choices=["demo","run"], default="demo")
    p.add_argument("--csv", type=str, default=None, help="CSV file with columns: filepath,label")
    p.add_argument("--outdir", type=str, default="results")
    p.add_argument("--nmax", type=int, default=None)
    p.add_argument("--sr", type=int, default=16000)
    args = p.parse_args()

    os.makedirs(args.outdir, exist_ok=True)

    if args.mode == "demo" or args.csv is None:
        print("Running demo: generating synthetic signals and extracting features...")
        signals, labels, sr = demo_synthetic(n_samples=60, sr=args.sr)
        feats = [extract_whfemd_features(s, sr) for s in signals]
        X = np.vstack(feats)
        y = np.array(labels)
    else:
        print(f"Loading and extracting features from CSV: {args.csv}")
        X, y, meta = prepare_from_csv(args.csv, sr=args.sr, nmax=args.nmax)

    print("Feature matrix shape:", X.shape)
    print("Running pipeline (PCA -> SMOTE/fallback -> GMM membership -> train)...")
    results = build_and_eval(X, y, random_state=42, pca_n=20, use_smote=True)

    print("\nBaseline results:")
    for k, v in results["baseline"].items():
        print(f" {k}: {v}")
    print("\nAugmented results:")
    for k, v in results["augmented"].items():
        print(f" {k}: {v}")

    # save summary
    summary = []
    for name, v in results["augmented"].items():
        summary.append({"model": name, "accuracy": v["accuracy"], "macro_f1": v["macro_f1"], "macro_recall": v["macro_recall"]})
    out_csv = os.path.join(args.outdir, "results_summary.csv")
    pd.DataFrame(summary).to_csv(out_csv, index=False)
    print(f"\nSaved summary to {out_csv}")

if __name__ == "__main__":
    main()
