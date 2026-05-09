"""
evaluator.py
============
Evaluates trained models on the held-out test set and produces:

  1. Classification metrics: Accuracy, Precision, Recall, F1-Score
  2. Regression metric: RMSE (Root Mean Squared Error) of return magnitude
  3. Comparison table printed to console and saved to CSV
  4. Training loss curves (train vs validation) for each model
  5. Predicted vs actual market direction plot over time

All plots are saved as PNG files in the `plots/` directory.
"""

import logging
import os

import matplotlib
matplotlib.use("Agg")  # non-interactive backend - works on headless servers

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    f1_score,
    precision_score,
    recall_score,
)

logger = logging.getLogger(__name__)

# Directory for saving plots
PLOTS_DIR   = os.path.join(os.path.dirname(__file__), "plots")
RESULTS_DIR = os.path.join(os.path.dirname(__file__), "results")


# ─────────────────────────────────────────────────────────────────────────────
# 1.  TEST-SET EVALUATION
# ─────────────────────────────────────────────────────────────────────────────

def _evaluate_model(
    model:       torch.nn.Module,
    test_loader: torch.utils.data.DataLoader,
    device:      torch.device,
) -> dict:
    """
    Run the model on the entire test set and compute metrics.

    Returns
    -------
    dict with keys:
        accuracy, precision, recall, f1, rmse,
        y_true (list), y_pred (list), y_reg_true, y_reg_pred
    """
    model.to(device)
    model.eval()

    all_y_true    = []
    all_y_pred    = []
    all_reg_true  = []
    all_reg_pred  = []

    with torch.no_grad():
        for X_batch, y_cls, y_reg in test_loader:
            X_batch = X_batch.to(device)
            logits, reg_pred = model(X_batch)

            preds = torch.argmax(logits, dim=1).cpu().numpy()
            all_y_true.extend(y_cls.numpy().tolist())
            all_y_pred.extend(preds.tolist())
            all_reg_true.extend(y_reg.numpy().tolist())
            all_reg_pred.extend(reg_pred.cpu().numpy().tolist())

    y_true = np.array(all_y_true)
    y_pred = np.array(all_y_pred)
    reg_true = np.array(all_reg_true)
    reg_pred = np.array(all_reg_pred)

    # Classification metrics
    acc  = accuracy_score(y_true, y_pred)
    prec = precision_score(y_true, y_pred, zero_division=0)
    rec  = recall_score(y_true, y_pred, zero_division=0)
    f1   = f1_score(y_true, y_pred, zero_division=0)

    # Regression metric
    rmse = np.sqrt(np.mean((reg_true - reg_pred) ** 2))

    return {
        "accuracy":    acc,
        "precision":   prec,
        "recall":      rec,
        "f1":          f1,
        "rmse":        rmse,
        "y_true":      y_true,
        "y_pred":      y_pred,
        "reg_true":    reg_true,
        "reg_pred":    reg_pred,
    }


# ─────────────────────────────────────────────────────────────────────────────
# 2.  COMPARISON TABLE
# ─────────────────────────────────────────────────────────────────────────────

def _build_comparison_table(eval_results: dict) -> pd.DataFrame:
    """
    Build a neat DataFrame comparing all models across all tickers.

    Parameters
    ----------
    eval_results : dict[ticker][model_name] -> metrics dict

    Returns
    -------
    DataFrame with columns: Ticker, Model, Accuracy, Precision, Recall, F1, RMSE
    """
    rows = []
    for ticker in eval_results:
        for model_name in eval_results[ticker]:
            m = eval_results[ticker][model_name]
            rows.append({
                "Ticker":    ticker,
                "Model":     model_name,
                "Accuracy":  round(m["accuracy"],  4),
                "Precision": round(m["precision"], 4),
                "Recall":    round(m["recall"],    4),
                "F1-Score":  round(m["f1"],        4),
                "RMSE":      round(m["rmse"],      6),
            })
    return pd.DataFrame(rows)


# ─────────────────────────────────────────────────────────────────────────────
# 3.  TRAINING LOSS CURVES
# ─────────────────────────────────────────────────────────────────────────────

def _plot_training_curves(train_results: dict):
    """
    Plot train vs validation loss for every (ticker, model) pair.
    Each ticker gets its own figure with subplots for RNN / LSTM / GRU.
    """
    os.makedirs(PLOTS_DIR, exist_ok=True)

    for ticker in train_results:
        model_names = list(train_results[ticker].keys())
        n_models    = len(model_names)

        fig, axes = plt.subplots(1, n_models, figsize=(6 * n_models, 5), squeeze=False)
        fig.suptitle(f"Training & Validation Loss - {ticker}", fontsize=14, fontweight="bold")

        for idx, model_name in enumerate(model_names):
            ax      = axes[0][idx]
            history = train_results[ticker][model_name]
            epochs  = range(1, len(history["train_losses"]) + 1)

            ax.plot(epochs, history["train_losses"], label="Train Loss", linewidth=2, color="#2196F3")
            ax.plot(epochs, history["val_losses"],   label="Val Loss",   linewidth=2, color="#FF5722", linestyle="--")
            ax.axvline(history["best_epoch"], color="#4CAF50", linestyle=":", alpha=0.7, label=f"Best epoch={history['best_epoch']}")

            ax.set_title(f"{model_name}", fontsize=12, fontweight="bold")
            ax.set_xlabel("Epoch")
            ax.set_ylabel("Loss")
            ax.legend(fontsize=9)
            ax.grid(True, alpha=0.3)

        plt.tight_layout()
        path = os.path.join(PLOTS_DIR, f"{ticker}_training_curves.png")
        plt.savefig(path, dpi=150, bbox_inches="tight")
        plt.close()
        logger.info("  Saved training curves -> %s", path)


# ─────────────────────────────────────────────────────────────────────────────
# 4.  PREDICTED vs ACTUAL DIRECTION PLOT
# ─────────────────────────────────────────────────────────────────────────────

def _plot_predictions(eval_results: dict):
    """
    For each ticker, overlay predicted vs actual directions on a timeline.
    Also shows model agreement regions and correct / incorrect markers.
    """
    os.makedirs(PLOTS_DIR, exist_ok=True)

    for ticker in eval_results:
        model_names = list(eval_results[ticker].keys())
        n_models    = len(model_names)

        fig, axes = plt.subplots(n_models, 1, figsize=(14, 4 * n_models), squeeze=False)
        fig.suptitle(f"Predicted vs Actual Direction - {ticker}", fontsize=14, fontweight="bold")

        for idx, model_name in enumerate(model_names):
            ax    = axes[idx][0]
            m     = eval_results[ticker][model_name]
            t     = np.arange(len(m["y_true"]))

            # Colour correct predictions green, incorrect red
            correct = m["y_true"] == m["y_pred"]

            ax.fill_between(t, 0, 1, where=correct,  alpha=0.15, color="green", label="Correct", step="mid")
            ax.fill_between(t, 0, 1, where=~correct, alpha=0.15, color="red",   label="Wrong",   step="mid")

            ax.step(t, m["y_true"], where="mid", label="Actual",    linewidth=2,   color="#1565C0")
            ax.step(t, m["y_pred"], where="mid", label="Predicted", linewidth=1.5, color="#E65100", linestyle="--")

            ax.set_title(f"{model_name}  (Acc={m['accuracy']:.2%})", fontsize=11, fontweight="bold")
            ax.set_ylabel("Direction (1=Up, 0=Down)")
            ax.set_xlabel("Test Sample Index")
            ax.set_ylim(-0.1, 1.1)
            ax.set_yticks([0, 1])
            ax.set_yticklabels(["Down", "Up"])
            ax.legend(fontsize=9, loc="upper right")
            ax.grid(True, alpha=0.3)

        plt.tight_layout()
        path = os.path.join(PLOTS_DIR, f"{ticker}_predictions.png")
        plt.savefig(path, dpi=150, bbox_inches="tight")
        plt.close()
        logger.info("  Saved prediction plot -> %s", path)


# ─────────────────────────────────────────────────────────────────────────────
# 5.  AGGREGATED MODEL COMPARISON BAR CHART
# ─────────────────────────────────────────────────────────────────────────────

def _plot_model_comparison(comparison_df: pd.DataFrame):
    """
    Bar chart comparing average Accuracy & F1 across tickers for each model.
    """
    os.makedirs(PLOTS_DIR, exist_ok=True)

    avg = comparison_df.groupby("Model")[["Accuracy", "F1-Score", "RMSE"]].mean()

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
    fig.suptitle("Model Comparison (averaged across tickers)", fontsize=14, fontweight="bold")

    # ── Classification metrics ────────────────────────────────────────────
    colors = {"RNN": "#2196F3", "LSTM": "#4CAF50", "GRU": "#FF9800"}
    x      = np.arange(len(avg))
    width  = 0.35

    bars1 = ax1.bar(x - width/2, avg["Accuracy"],  width, label="Accuracy",  color=[colors.get(m, "#999") for m in avg.index], alpha=0.85)
    bars2 = ax1.bar(x + width/2, avg["F1-Score"],  width, label="F1-Score",  color=[colors.get(m, "#999") for m in avg.index], alpha=0.55)

    ax1.set_xticks(x)
    ax1.set_xticklabels(avg.index, fontsize=12, fontweight="bold")
    ax1.set_ylabel("Score")
    ax1.set_title("Accuracy & F1-Score")
    ax1.legend()
    ax1.set_ylim(0, 1)
    ax1.grid(True, axis="y", alpha=0.3)

    # Add value labels
    for bar in list(bars1) + list(bars2):
        h = bar.get_height()
        ax1.annotate(f"{h:.2f}", xy=(bar.get_x() + bar.get_width() / 2, h),
                     xytext=(0, 3), textcoords="offset points", ha="center", fontsize=9)

    # ── RMSE ──────────────────────────────────────────────────────────────
    bars3 = ax2.bar(avg.index, avg["RMSE"], color=[colors.get(m, "#999") for m in avg.index], alpha=0.85)
    ax2.set_ylabel("RMSE")
    ax2.set_title("Return Magnitude RMSE (lower is better)")
    ax2.grid(True, axis="y", alpha=0.3)

    for bar in bars3:
        h = bar.get_height()
        ax2.annotate(f"{h:.4f}", xy=(bar.get_x() + bar.get_width() / 2, h),
                     xytext=(0, 3), textcoords="offset points", ha="center", fontsize=9)

    plt.tight_layout()
    path = os.path.join(PLOTS_DIR, "model_comparison.png")
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    logger.info("  Saved model comparison chart -> %s", path)


# ─────────────────────────────────────────────────────────────────────────────
# 6.  MASTER EVALUATOR  (called by main.py)
# ─────────────────────────────────────────────────────────────────────────────

def evaluate_all(
    train_results: dict,
    loaders:       dict,
) -> pd.DataFrame:
    """
    Evaluate every model on every ticker's test set and generate all outputs.

    Parameters
    ----------
    train_results : output of trainer.train_all_models()
    loaders       : output of dataset_builder.build_dataset()

    Returns
    -------
    comparison_df : DataFrame with per-(ticker, model) metrics - also saved to CSV.
    """
    logger.info("=" * 60)
    logger.info("STEP 5 - EVALUATION & RESULTS")
    logger.info("=" * 60)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    eval_results = {}

    for ticker in train_results:
        eval_results[ticker] = {}
        test_loader = loaders[ticker]["test"]

        for model_name in train_results[ticker]:
            model = train_results[ticker][model_name]["model"]
            metrics = _evaluate_model(model, test_loader, device)
            eval_results[ticker][model_name] = metrics

    # ── Build comparison table ────────────────────────────────────────────
    comparison_df = _build_comparison_table(eval_results)

    # ── Print the table ───────────────────────────────────────────────────
    logger.info("\n" + "=" * 80)
    logger.info("FINAL MODEL COMPARISON")
    logger.info("=" * 80)
    print("\n" + comparison_df.to_string(index=False))
    print()

    # ── Print per-ticker classification reports ───────────────────────────
    for ticker in eval_results:
        for model_name in eval_results[ticker]:
            m = eval_results[ticker][model_name]
            logger.info("\n-- %s / %s --", ticker, model_name)
            report = classification_report(
                m["y_true"], m["y_pred"],
                target_names=["Down", "Up"],
                zero_division=0,
            )
            print(report)

    # ── Save CSV ──────────────────────────────────────────────────────────
    os.makedirs(RESULTS_DIR, exist_ok=True)
    csv_path = os.path.join(RESULTS_DIR, "model_comparison.csv")
    comparison_df.to_csv(csv_path, index=False)
    logger.info("Results saved -> %s", csv_path)

    # ── Generate all plots ────────────────────────────────────────────────
    logger.info("\nGenerating plots ...")
    _plot_training_curves(train_results)
    _plot_predictions(eval_results)
    _plot_model_comparison(comparison_df)
    logger.info("All plots saved to %s/", PLOTS_DIR)

    return comparison_df


# ─────────────────────────────────────────────────────────────────────────────
# QUICK TEST
# ─────────────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    from data_collector    import collect_all_data
    from sentiment_analyzer import run_sentiment_pipeline
    from dataset_builder   import build_dataset
    from models            import build_all_models
    from trainer           import train_all_models

    price_data, text_data = collect_all_data()
    _, sentiment_df       = run_sentiment_pipeline(text_data)
    loaders, n_features   = build_dataset(price_data, sentiment_df)
    models                = build_all_models(n_features)
    train_results         = train_all_models(models, loaders, epochs=5)
    comparison            = evaluate_all(train_results, loaders)
    print("\nDone!")
