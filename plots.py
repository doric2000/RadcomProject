from __future__ import annotations

import os
from pathlib import Path
from typing import Iterable, List, Optional, Tuple

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.metrics import ConfusionMatrixDisplay, accuracy_score, confusion_matrix


# =============================================================================
# PRESENTATION PLOT GENERATOR
#
# Generates:
# - Class balance charts
# - Protocol share + down/up ratio histogram (APP)
# - Correlation heatmap (sample)
# - Confusion matrices (ATT full, APP top-K)
# - Feature importance (tree models)
#
# All figures are saved under: result/plots/
# =============================================================================


ROOT = Path(__file__).resolve().parent
OUT_DIR = ROOT / "result" / "plots"


def _ensure_out_dir() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)


def _save_fig(fig: plt.Figure, name: str) -> Path:
    """Save figure as PNG to the output directory."""
    _ensure_out_dir()
    path = OUT_DIR / f"{name}.png"
    fig.tight_layout()
    fig.savefig(path, dpi=180)
    plt.close(fig)
    return path


def _top_k_counts(s: pd.Series, k: int) -> pd.Series:
    return s.value_counts().head(k)


def _plot_top_k_bar(counts: pd.Series, title: str, xlabel: str, name: str) -> None:
    fig, ax = plt.subplots(figsize=(11, 6))
    sns.barplot(x=counts.values, y=counts.index, ax=ax, palette="viridis")
    ax.set_title(title)
    ax.set_xlabel(xlabel)
    ax.set_ylabel("Class")
    _save_fig(fig, name)


def _plot_protocol_share(df: pd.DataFrame, title: str, name: str) -> None:
    if "Protocol" not in df.columns:
        return
    p = df["Protocol"].astype(str).str.upper().value_counts(normalize=True)
    # Keep TCP/UDP explicitly if present
    labels = [x for x in ["TCP", "UDP"] if x in p.index]
    values = [p.get(x, 0.0) for x in labels]
    if not labels:
        return

    fig, ax = plt.subplots(figsize=(6.5, 4))
    ax.bar(labels, values)
    ax.set_ylim(0, 1)
    ax.set_ylabel("Fraction")
    ax.set_title(title)
    _save_fig(fig, name)


def _plot_down_up_ratio_hist(df: pd.DataFrame, title: str, name: str) -> None:
    if not {"bwd_packets_length", "fwd_packets_length"}.issubset(df.columns):
        return
    down = df["bwd_packets_length"].astype(float) + 1.0
    up = df["fwd_packets_length"].astype(float) + 1.0
    ratio = np.log10(down / up)
    fig, ax = plt.subplots(figsize=(7.5, 4.5))
    ax.hist(ratio, bins=50)
    ax.set_title(title)
    ax.set_xlabel("log10(down/up)")
    ax.set_ylabel("Count")
    _save_fig(fig, name)


def _plot_corr_heatmap(df: pd.DataFrame, cols: List[str], title: str, name: str) -> None:
    numeric_df = df.select_dtypes(include=[np.number]).copy()
    corr_cols = [c for c in cols if c in numeric_df.columns]
    if len(corr_cols) < 2:
        return
    corr = numeric_df[corr_cols].corr()
    fig, ax = plt.subplots(figsize=(9, 7))
    sns.heatmap(corr, annot=True, cmap="coolwarm", fmt=".2f", ax=ax)
    ax.set_title(title)
    _save_fig(fig, name)


def _plot_confusion_matrix(
    y_true: Iterable[str],
    y_pred: Iterable[str],
    labels: List[str],
    title: str,
    name: str,
    normalize: Optional[str] = "true",
    figsize: Tuple[int, int] = (10, 8),
) -> None:
    cm = confusion_matrix(y_true, y_pred, labels=labels, normalize=normalize)
    fig, ax = plt.subplots(figsize=figsize)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=labels)
    disp.plot(ax=ax, cmap="Blues", colorbar=True, values_format=".2f" if normalize else "d")
    ax.set_title(title)
    plt.xticks(rotation=45, ha="right")
    _save_fig(fig, name)


def _plot_feature_importance(model, feature_names: List[str], title: str, name: str, top_k: int = 20) -> None:
    # Works for tree models with feature_importances_
    if not hasattr(model, "feature_importances_"):
        return
    imp = np.asarray(model.feature_importances_, dtype=float)
    if imp.size != len(feature_names):
        return
    idx = np.argsort(imp)[::-1][:top_k]
    top_features = [feature_names[i] for i in idx]
    top_vals = imp[idx]

    fig, ax = plt.subplots(figsize=(11, 6))
    sns.barplot(x=top_vals, y=top_features, ax=ax, palette="mako")
    ax.set_title(title)
    ax.set_xlabel("Importance")
    ax.set_ylabel("Feature")
    _save_fig(fig, name)


# =============================================================================
# APP: Train/load model, evaluate, and plot
# =============================================================================


def run_app_section(top_k_cm: int = 15) -> None:
    from app_model import PATHS as APP_PATHS
    from app_model import TARGET_COL as APP_TARGET
    from app_model import build_feature_matrix
    from app_model import train_app_model

    print("[APP] Loading datasets...")
    train_df = pd.read_csv(ROOT / APP_PATHS.train)
    test_df = pd.read_csv(ROOT / APP_PATHS.test)

    # EDA plots
    _plot_top_k_bar(
        _top_k_counts(train_df[APP_TARGET], 20),
        title="APP Train: Top 20 Most Frequent Applications",
        xlabel="Number of Samples",
        name="app_train_top20_classes",
    )
    _plot_protocol_share(train_df, title="APP Train: Protocol Share", name="app_train_protocol_share")
    _plot_down_up_ratio_hist(train_df, title="APP Train: log10(Down/Up Bytes Ratio)", name="app_train_down_up_ratio")
    _plot_corr_heatmap(
        train_df,
        cols=[
            "fwd_packets_amount",
            "bwd_packets_amount",
            "fwd_packets_length",
            "bwd_packets_length",
            "min_fwd_inter_arrival_time",
            "mean_fwd_inter_arrival_time",
            "max_fwd_inter_arrival_time",
        ],
        title="APP Train: Feature Correlation (Selected)",
        name="app_train_corr_selected",
    )

    # Train (or load) model artifacts
    model_path = ROOT / APP_PATHS.model_dir / "app_model.pkl"
    le_path = ROOT / APP_PATHS.model_dir / "app_label_encoder.pkl"
    cols_path = ROOT / APP_PATHS.model_dir / "app_columns.pkl"

    if model_path.exists() and le_path.exists() and cols_path.exists():
        print("[APP] Loading existing model artifacts from models/...")
        model = joblib.load(model_path)
        le = joblib.load(le_path)
        features = joblib.load(cols_path)
    else:
        print("[APP] Training model (artifacts not found)...")
        model, le, features = train_app_model()

    # Build test features & predict
    X_test, y_test, _ = build_feature_matrix(test_df, target_col=APP_TARGET)
    X_test = X_test.reindex(columns=features, fill_value=0.0)
    y_true = y_test.astype(str).values
    y_true_enc = le.transform(y_test.astype(str))
    y_pred_enc = model.predict(X_test)
    y_pred = le.inverse_transform(y_pred_enc)
    acc = accuracy_score(y_true_enc, y_pred_enc)
    print(f"[APP] Test accuracy: {acc:.4f}")

    # Feature importance (from sub-models if available)
    if hasattr(model, "named_estimators_"):
        rf = model.named_estimators_.get("rf")
        et = model.named_estimators_.get("et")
        if rf is not None:
            _plot_feature_importance(rf, features, "APP: RandomForest Feature Importance (Top 20)", "app_rf_feature_importance", 20)
        if et is not None:
            _plot_feature_importance(et, features, "APP: ExtraTrees Feature Importance (Top 20)", "app_et_feature_importance", 20)

    # Confusion matrix is too large for full APP → focus on top-K most frequent classes in test
    top_labels = test_df[APP_TARGET].astype(str).value_counts().head(top_k_cm).index.tolist()
    mask = pd.Series(y_true).isin(top_labels).values
    _plot_confusion_matrix(
        y_true=pd.Series(y_true)[mask],
        y_pred=pd.Series(y_pred)[mask],
        labels=top_labels,
        title=f"APP Test: Confusion Matrix (Top {top_k_cm} Classes, Normalized)",
        name=f"app_test_confusion_top{top_k_cm}",
        normalize="true",
        figsize=(12, 10),
    )


# =============================================================================
# ATT: Train/load model, evaluate, and plot
# =============================================================================


def run_att_section() -> None:
    from att_model import MODEL_DIR as ATT_MODEL_DIR
    from att_model import TEST_PATH as ATT_TEST_PATH
    from att_model import TRAIN_PATH as ATT_TRAIN_PATH
    from att_model import extract_isolation_features
    from att_model import train_att_isolation

    print("[ATT] Loading datasets...")
    train_df = pd.read_csv(ROOT / ATT_TRAIN_PATH)
    test_df = pd.read_csv(ROOT / ATT_TEST_PATH)

    # EDA plots
    _plot_top_k_bar(
        _top_k_counts(train_df["attribution"].astype(str), 20),
        title="ATT Train: Top 20 Most Frequent Classes",
        xlabel="Number of Samples",
        name="att_train_top20_classes",
    )
    _plot_protocol_share(train_df, title="ATT Train: Protocol Share", name="att_train_protocol_share")
    _plot_down_up_ratio_hist(train_df, title="ATT Train: log10(Down/Up Bytes Ratio)", name="att_train_down_up_ratio")

    model_path = ROOT / ATT_MODEL_DIR / "att_model.pkl"
    scaler_path = ROOT / ATT_MODEL_DIR / "att_scaler.pkl"
    le_path = ROOT / ATT_MODEL_DIR / "att_label_encoder.pkl"
    cols_path = ROOT / ATT_MODEL_DIR / "att_columns.pkl"

    if model_path.exists() and scaler_path.exists() and le_path.exists() and cols_path.exists():
        print("[ATT] Loading existing model artifacts from models/...")
        model = joblib.load(model_path)
        scaler = joblib.load(scaler_path)
        le = joblib.load(le_path)
        features = joblib.load(cols_path)
    else:
        print("[ATT] Training model (artifacts not found)...")
        model, scaler, le, features = train_att_isolation()

    # Feature engineering
    test_eng = extract_isolation_features(test_df, q_large=0.90, n_packets=20)
    X_test = test_eng[features].fillna(0.0).reindex(columns=features, fill_value=0.0)
    X_test_scaled = scaler.transform(X_test)
    y_test = test_df["attribution"].astype(str)
    y_true = y_test.values
    y_true_enc = le.transform(y_test)
    y_pred_enc = model.predict(X_test_scaled)
    y_pred = le.inverse_transform(y_pred_enc)
    acc = accuracy_score(y_true_enc, y_pred_enc)
    print(f"[ATT] Test accuracy: {acc:.4f}")

    # Feature importance (RF inside VotingClassifier)
    if hasattr(model, "named_estimators_") and "rf" in model.named_estimators_:
        rf = model.named_estimators_["rf"]
        _plot_feature_importance(rf, features, "ATT: RandomForest Feature Importance (Top 20)", "att_rf_feature_importance", 20)

    # Confusion matrix (usually manageable size)
    labels = le.classes_.tolist()
    _plot_confusion_matrix(
        y_true=y_true,
        y_pred=y_pred,
        labels=labels,
        title="ATT Test: Confusion Matrix (Normalized)",
        name="att_test_confusion",
        normalize="true",
        figsize=(10, 8),
    )


def main() -> None:
    sns.set_theme(style="whitegrid")
    print(f"Saving plots to: {OUT_DIR}")
    run_app_section(top_k_cm=15)
    run_att_section()
    print("Done. Generated presentation plots.")


if __name__ == "__main__":
    main()