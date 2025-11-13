import os
import re
import pickle
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from tensorflow.keras.models import load_model


def _clean_columns(df: pd.DataFrame) -> pd.DataFrame:
    rename = {}
    for col in df.columns:
        key = re.sub(r"[^A-Za-z0-9]+", "_", col.strip()).strip("_").lower()
        rename[col] = key
    return df.rename(columns=rename)


def _ensure_dir(path: str) -> None:
    os.makedirs(path, exist_ok=True)


def load_data(csv_path: str) -> pd.DataFrame:
    df = pd.read_csv(csv_path)
    df = _clean_columns(df)
    # Expected columns after cleaning
    # name, sex, red_pixel, green_pixel, blue_pixel, hb, anaemic
    # Some CSV headers may map to e.g. _red_pixel; harmonize
    for candidate in ["red_pixel", "_red_pixel", "percent_red_pixel", "red"]:
        if candidate in df.columns:
            df["red_pixel"] = df[candidate]
            break
    for candidate in ["green_pixel", "_green_pixel", "percent_green_pixel", "green"]:
        if candidate in df.columns:
            df["green_pixel"] = df[candidate]
            break
    for candidate in ["blue_pixel", "_blue_pixel", "percent_blue_pixel", "blue"]:
        if candidate in df.columns:
            df["blue_pixel"] = df[candidate]
            break
    # Standardize target name
    if "hb" not in df.columns:
        raise ValueError("'Hb' column not found after cleaning. Check CSV headers.")
    # Normalize label field
    if "anaemic" in df.columns:
        df["anaemic"] = df["anaemic"].astype(str).str.strip().str.lower()
    return df


def load_model_and_scalers(models_dir: str):
    model_path = os.path.join(models_dir, "Hemoglobin_predictor.h5")
    in_scaler_path = os.path.join(models_dir, "input_scaler.pkl")
    out_scaler_path = os.path.join(models_dir, "output_scaler.pkl")
    model = load_model(model_path)
    with open(in_scaler_path, "rb") as f:
        in_scaler = pickle.load(f)
    with open(out_scaler_path, "rb") as f:
        out_scaler = pickle.load(f)
    return model, in_scaler, out_scaler


def predict_hb(df: pd.DataFrame, model, in_scaler, out_scaler) -> np.ndarray:
    X = df[["red_pixel", "green_pixel", "blue_pixel"]].to_numpy(dtype=np.float32)
    Xs = in_scaler.transform(X)
    y_pred_s = model.predict(Xs, verbose=0)
    y_pred = out_scaler.inverse_transform(y_pred_s)
    return y_pred.ravel()


def plot_basic_distributions(df: pd.DataFrame, out_dir: str) -> None:
    plt.figure(figsize=(6,4))
    sns.histplot(df["hb"], bins=20, kde=True, color="#ff7a18")
    plt.xlabel("Hemoglobin (g/dL)")
    plt.title("Hemoglobin Distribution")
    plt.tight_layout(); plt.savefig(os.path.join(out_dir, "hb_hist.png"), dpi=220)

    if "sex" in df.columns:
        plt.figure(figsize=(6,4))
        sns.boxplot(x=df["sex"], y=df["hb"], palette="Oranges")
        plt.xlabel("Sex"); plt.ylabel("Hemoglobin (g/dL)")
        plt.title("Hemoglobin by Sex")
        plt.tight_layout(); plt.savefig(os.path.join(out_dir, "hb_by_sex_box.png"), dpi=220)

    plt.figure(figsize=(10,3))
    for i, c in enumerate(["red_pixel", "green_pixel", "blue_pixel"]):
        ax = plt.subplot(1,3,i+1)
        sns.scatterplot(x=df[c], y=df["hb"], s=18, color="#ff7a18", edgecolor=None)
        ax.set_xlabel(c.replace("_", " ").title()); ax.set_ylabel("Hb (g/dL)" if i==0 else "")
    plt.suptitle("Color Channel vs Hemoglobin"); plt.tight_layout(rect=[0,0,1,0.95])
    plt.savefig(os.path.join(out_dir, "rgb_vs_hb_scatter.png"), dpi=220)

    plt.figure(figsize=(5,4))
    corr = df[["red_pixel","green_pixel","blue_pixel","hb"]].corr()
    sns.heatmap(corr, cmap="Oranges", annot=True, fmt=".2f")
    plt.title("Correlation Heatmap")
    plt.tight_layout(); plt.savefig(os.path.join(out_dir, "correlation_heatmap.png"), dpi=220)

    if "anaemic" in df.columns:
        plt.figure(figsize=(5,4))
        sns.countplot(x=df["anaemic"], palette="Oranges")
        plt.xlabel("Anaemic"); plt.ylabel("Count"); plt.title("Class Distribution")
        plt.tight_layout(); plt.savefig(os.path.join(out_dir, "class_counts.png"), dpi=220)


def plot_regression_quality(df: pd.DataFrame, y_pred: np.ndarray, out_dir: str) -> None:
    y_true = df["hb"].to_numpy()
    lims = [float(min(y_true.min(), y_pred.min())), float(max(y_true.max(), y_pred.max()))]
    # Pred vs True
    plt.figure(figsize=(5,5))
    plt.scatter(y_true, y_pred, s=18, alpha=0.7)
    plt.plot(lims, lims, "r--", lw=2)
    plt.xlabel("True Hb (g/dL)"); plt.ylabel("Predicted Hb (g/dL)")
    plt.title("Predicted vs True")
    plt.xlim(lims); plt.ylim(lims); plt.tight_layout()
    plt.savefig(os.path.join(out_dir, "pred_vs_true.png"), dpi=220)

    # Residuals
    res = y_pred - y_true
    plt.figure(figsize=(6,4))
    sns.histplot(res, bins=20, kde=True, color="#ffb347")
    plt.xlabel("Residual (Pred − True) g/dL"); plt.title("Residuals Histogram")
    plt.tight_layout(); plt.savefig(os.path.join(out_dir, "residuals_hist.png"), dpi=220)

    # Bland–Altman
    mean = (y_true + y_pred) / 2
    md = res.mean(); sd = res.std(ddof=1)
    loa = (md - 1.96*sd, md + 1.96*sd)
    plt.figure(figsize=(6,4))
    plt.scatter(mean, res, s=18, alpha=0.7)
    for y, ls in [(md, "--"), (loa[0], ":"), (loa[1], ":")]:
        plt.axhline(y, color="r", linestyle=ls, alpha=0.8)
    plt.xlabel("Mean of True & Pred (g/dL)"); plt.ylabel("Pred − True (g/dL)")
    plt.title("Bland–Altman")
    plt.tight_layout(); plt.savefig(os.path.join(out_dir, "bland_altman.png"), dpi=220)


def main():
    base_dir = os.path.dirname(__file__)
    out_dir = os.path.join(base_dir, "presentation_figs")
    _ensure_dir(out_dir)

    csv_path = os.path.join(base_dir, "Anemia_Dataset.csv")
    models_dir = os.path.join(base_dir, "models")

    df = load_data(csv_path)
    plot_basic_distributions(df, out_dir)

    try:
        model, in_scaler, out_scaler = load_model_and_scalers(models_dir)
        y_pred = predict_hb(df, model, in_scaler, out_scaler)
        plot_regression_quality(df, y_pred, out_dir)
    except Exception as e:
        # Fallback: skip model-based plots if weights/scalers are unavailable
        print(f"Warning: Skipping model-based plots due to: {e}")

    print(f"Saved figures to: {out_dir}")


if __name__ == "__main__":
    main()


