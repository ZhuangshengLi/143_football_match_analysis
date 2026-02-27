"""
Ablation experiments: Model A/B/C/D

Model A: L1 static ability only
Model B: L1 + L2 tactical style
Model C: L1 + L2 + L3 dynamic state
Model D: C + odds
"""
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import roc_auc_score, log_loss, brier_score_loss


def _optimal_threshold(y_true: np.ndarray, proba: np.ndarray) -> float:
    """Step size 0.005 for threshold stability."""
    from sklearn.metrics import balanced_accuracy_score
    best_t, best_v = 0.5, 0.0
    for t in np.arange(0.3, 0.71, 0.005):
        pred = (proba >= t).astype(int)
        v = balanced_accuracy_score(y_true, pred)
        if v > best_v:
            best_v, best_t = v, t
    return best_t


def _prepare_X(
    df: pd.DataFrame,
    cols: list,
    fill_means: dict[str, float] | None = None,
) -> np.ndarray:
    """
    Select columns, fill NaN.
    If fill_means (from train) is provided, use it for val/test to avoid leakage.
    """
    avail = [c for c in cols if c in df.columns]
    X = df[avail].copy()
    for c in avail:
        if X[c].isna().any():
            fill = fill_means.get(c, X[c].mean()) if fill_means is not None else X[c].mean()
            X[c] = X[c].fillna(fill)
    return X.values


def run_ablation(
    df: pd.DataFrame,
    feature_groups: dict[str, list[str]] | None = None,
    test_size: float = 0.2,
    val_size: float = 0.2,
    class_weight: str | None = "balanced",
    random_state: int = 42,
) -> pd.DataFrame:
    """
    Run Model A/B/C/D ablation experiments.
    Split by time: first 80% train, last 20% test (avoid information leakage).
    NaN filled with train mean for val/test; class_weight defaults to "balanced", pass None for weighted ablation.
    """
    from inputs.match_features import get_feature_groups
    if feature_groups is None:
        feature_groups = get_feature_groups(df)
    df = df.sort_values("date").reset_index(drop=True)
    n = len(df)
    n_test = int(n * test_size)
    n_val = int((n - n_test) * val_size)
    n_train = n - n_test - n_val

    train_df = df.iloc[:n_train]
    val_df = df.iloc[n_train : n_train + n_val]
    test_df = df.iloc[n_train + n_val :]

    y_train = train_df["home_win"].values
    y_val = val_df["home_win"].values
    y_test = test_df["home_win"].values

    models = {
        "A": feature_groups["L1"],
        "B": feature_groups["L1"] + feature_groups["L2"],
        "C": feature_groups["L1"] + feature_groups["L2"] + feature_groups["L3"],
        "D": feature_groups["L1"] + feature_groups["L2"] + feature_groups["L3"] + feature_groups["odds"],
    }

    results = []
    for name, cols in models.items():
        avail = [c for c in cols if c in df.columns]
        if not avail:
            continue
        # Fill val/test with train mean to avoid leakage
        train_means = {c: train_df[c].mean() for c in avail}
        X_train = _prepare_X(train_df, avail, fill_means=None)
        X_val = _prepare_X(val_df, avail, fill_means=train_means)
        X_test = _prepare_X(test_df, avail, fill_means=train_means)

        scaler = StandardScaler()
        X_train_s = scaler.fit_transform(X_train)
        X_val_s = scaler.transform(X_val)
        X_test_s = scaler.transform(X_test)

        clf = LogisticRegression(C=1.0, max_iter=1000, class_weight=class_weight, random_state=random_state)
        clf.fit(X_train_s, y_train)

        proba_val = clf.predict_proba(X_val_s)[:, 1]
        proba_test = clf.predict_proba(X_test_s)[:, 1]
        threshold = _optimal_threshold(y_val, proba_val)
        pred_test = (proba_test >= threshold).astype(int)

        auc = roc_auc_score(y_test, proba_test)
        logloss = log_loss(y_test, proba_test)
        brier = brier_score_loss(y_test, proba_test)
        acc = (pred_test == y_test).mean()

        # Home/away coefficients: interpretable if model has "home" related features
        coef = dict(zip(avail, clf.coef_[0]))
        home_coef = [v for k, v in coef.items() if "home" in k.lower() or "diff" in k.lower()]
        home_coef_mean = np.mean(home_coef) if home_coef else np.nan

        results.append({
            "model": name,
            "n_features": len(avail),
            "test_auc": auc,
            "test_logloss": logloss,
            "test_brier": brier,
            "test_acc": acc,
            "threshold": threshold,
        })

    return pd.DataFrame(results)
