"""
train.py
Contains the main pipeline: split, scale, baseline training, PCA, oversample, GMM membership, augment, train + eval.
"""
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.mixture import GaussianMixture
from sklearn.metrics import accuracy_score, f1_score, recall_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.neural_network import MLPClassifier

from oversample import smote_or_fallback

def eval_metrics(y_true, y_pred):
    return {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "macro_f1": float(f1_score(y_true, y_pred, average="macro")),
        "macro_recall": float(recall_score(y_true, y_pred, average="macro"))
    }

def build_and_eval(X, y, random_state=42, pca_n=20, use_smote=True):
    """
    Run pipeline on feature matrix X and label vector y.
    Returns dict with baseline and augmented results and some internals.
    """
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, stratify=y, random_state=random_state)

    scaler = StandardScaler().fit(X_train)
    X_train_s = scaler.transform(X_train)
    X_test_s = scaler.transform(X_test)

    # Baseline models
    base_models = {
        "RF_base": RandomForestClassifier(n_estimators=200, random_state=random_state),
        "DT_base": DecisionTreeClassifier(random_state=random_state),
        "MLP_base": MLPClassifier(hidden_layer_sizes=(128,), max_iter=200, random_state=random_state)
    }
    for m in base_models.values():
        m.fit(X_train_s, y_train)

    baseline_res = {}
    for name, m in base_models.items():
        baseline_res[name] = eval_metrics(y_test, m.predict(X_test_s))

    # PCA
    pca = PCA(n_components=min(pca_n, X_train_s.shape[1]), random_state=random_state)
    X_train_p = pca.fit_transform(X_train_s)
    X_test_p = pca.transform(X_test_s)

    # Oversample (SMOTE or fallback)
    X_res, y_res, used_smote = smote_or_fallback(X_train_p, y_train, seed=random_state) if use_smote else (X_train_p, y_train, False)

    # GMM (EM) to extract soft membership features
    gmm = GaussianMixture(n_components=len(np.unique(y_res)), covariance_type="full", random_state=random_state, max_iter=300)
    gmm.fit(X_res)
    mem_train = gmm.predict_proba(X_res)
    mem_test = gmm.predict_proba(X_test_p)

    # Augment PCA features with membership probabilities
    X_res_aug = np.hstack([X_res, mem_train])
    X_test_aug = np.hstack([X_test_p, mem_test])

    scaler_aug = StandardScaler().fit(X_res_aug)
    X_res_aug_s = scaler_aug.transform(X_res_aug)
    X_test_aug_s = scaler_aug.transform(X_test_aug)

    # Train same model classes on augmented data
    aug_models = {
        "RF_aug": RandomForestClassifier(n_estimators=200, random_state=random_state),
        "DT_aug": DecisionTreeClassifier(random_state=random_state),
        "MLP_aug": MLPClassifier(hidden_layer_sizes=(128,), max_iter=200, random_state=random_state)
    }
    for m in aug_models.values():
        m.fit(X_res_aug_s, y_res)

    aug_res = {}
    for name, m in aug_models.items():
        aug_res[name] = eval_metrics(y_test, m.predict(X_test_aug_s))

    return {
        "baseline": baseline_res,
        "augmented": aug_res,
        "used_smote": used_smote,
        "internals": {
            "scaler": scaler,
            "pca": pca,
            "gmm": gmm,
            "scaler_aug": scaler_aug
        }
    }
