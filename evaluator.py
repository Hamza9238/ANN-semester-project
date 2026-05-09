"""
evaluator.py
============
Evaluates trained models on the test set and computes comprehensive metrics.

Metrics computed
----------------
  • Accuracy         – % of correct predictions
  • Precision        – of positive predictions, how many were correct
  • Recall           – of actual positives, how many were found
  • F1 Score         – harmonic mean of precision and recall
  • Confusion Matrix – breakdown of predictions (TP, FP, TN, FN)

The evaluate_all function processes each model on each ticker,
generates individual model metrics, and returns a summary DataFrame.
"""

import logging
import os

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    classification_report,
)

from helpers import get_device, ensure_directory, print_step_header

logger = logging.getLogger(__name__)

# ─────────────────────────────────────────────────────────────────────────────
# CONFIGURATION
# ─────────────────────────────────────────────────────────────────────────────

# Directory to save results
RESULTS_DIR = os.path.join(os.path.dirname(__file__), "results")


# ─────────────────────────────────────────────────────────────────────────────
# SINGLE MODEL EVALUATION
# ─────────────────────────────────────────────────────────────────────────────

def _evaluate_one_model(
    model: nn.Module,
    model_name: str,
    ticker: str,
    test_loader: torch.utils.data.DataLoader,
    device: torch.device,
) -> dict:
    """
    Evaluate a single model on the test set and compute classification metrics.

    Parameters
    ----------
    model         : trained PyTorch model
    model_name    : name of model (e.g., "RNN", "LSTM", "GRU")
    ticker        : stock ticker symbol
    test_loader   : PyTorch DataLoader with test data
    device        : torch.device (cpu or cuda)

    Returns
    -------
    dict containing:
        - y_true  : ground truth binary labels (0/1)
        - y_pred  : predicted labels (0/1)
        - accuracy, precision, recall, f1_score
        - confusion_matrix : 2x2 array [[TN, FP], [FN, TP]]
        - report : classification report string
    """
    model.to(device)
    model.eval()

    all_preds = []
    all_labels = []

    with torch.no_grad():
        for X_batch, y_cls_batch, y_reg_batch in test_loader:
            X_batch = X_batch.to(device)
            y_cls_batch = y_cls_batch.to(device)

            # Forward pass: get logits for classification head
            logits, _ = model(X_batch)

            # Convert logits to predicted class (argmax)
            preds = torch.argmax(logits, dim=1).cpu().numpy()
            labels = y_cls_batch.cpu().numpy()

            all_preds.extend(preds)
            all_labels.extend(labels)

    y_true = np.array(all_labels)
    y_pred = np.array(all_preds)

    # Compute metrics
    acc = accuracy_score(y_true, y_pred)
    prec = precision_score(y_true, y_pred, zero_division=0)
    rec = recall_score(y_true, y_pred, zero_division=0)
    f1 = f1_score(y_true, y_pred, zero_division=0)
    cm = confusion_matrix(y_true, y_pred)
    report = classification_report(y_true, y_pred, output_dict=False)

    # Extract confusion matrix values
    # cm has shape (2, 2): [[TN, FP], [FN, TP]]
    tn = cm[0, 0] if cm.shape[0] > 0 else 0
    fp = cm[0, 1] if cm.shape[1] > 1 else 0
    fn = cm[1, 0] if cm.shape[0] > 1 else 0
    tp = cm[1, 1] if cm.shape == (2, 2) else 0

    logger.info(
        "  [%s] %s: Acc=%.4f  Prec=%.4f  Rec=%.4f  F1=%.4f",
        ticker, model_name, acc, prec, rec, f1,
    )

    return {
        "ticker": ticker,
        "model": model_name,
        "accuracy": acc,
        "precision": prec,
        "recall": rec,
        "f1_score": f1,
        "true_negatives": int(tn),
        "false_positives": int(fp),
        "false_negatives": int(fn),
        "true_positives": int(tp),
        "y_true": y_true,
        "y_pred": y_pred,
        "confusion_matrix": cm,
        "classification_report": report,
    }


# ─────────────────────────────────────────────────────────────────────────────
# MASTER EVALUATOR (called by main.py)
# ─────────────────────────────────────────────────────────────────────────────

def evaluate_all(
    train_results: dict,
    loaders: dict,
) -> pd.DataFrame:
    """
    Evaluate all trained models on their respective test sets.

    Parameters
    ----------
    train_results : output of trainer.train_all_models()
                    structure: train_results[ticker][model_name] -> training dict
    loaders       : output of dataset_builder.build_dataset()
                    structure: loaders[ticker] -> {"train", "val", "test", "n_samples"}

    Returns
    -------
    comparison_df : pandas DataFrame with columns:
                    - Ticker
                    - Model
                    - Accuracy
                    - Precision
                    - Recall
                    - F1 Score
    """
    print_step_header(5, "MODEL EVALUATION")

    device = get_device()

    all_results = []

    for ticker in train_results:
        logger.info("Evaluating ticker: %s", ticker)

        if ticker not in loaders:
            logger.warning("  No loader for %s – skipping", ticker)
            continue

        test_loader = loaders[ticker]["test"]

        for model_name, history in train_results[ticker].items():
            # Get the trained model
            model = history["model"]

            # Evaluate on test set
            eval_metrics = _evaluate_one_model(
                model=model,
                model_name=model_name,
                ticker=ticker,
                test_loader=test_loader,
                device=device,
            )

            all_results.append(eval_metrics)

            # Log detailed confusion matrix
            logger.info(
                "    Confusion Matrix (TN=%d, FP=%d, FN=%d, TP=%d)",
                eval_metrics["true_negatives"],
                eval_metrics["false_positives"],
                eval_metrics["false_negatives"],
                eval_metrics["true_positives"],
            )

    if not all_results:
        logger.error("No models were evaluated!")
        return pd.DataFrame()

    # Create comparison DataFrame
    comparison_df = pd.DataFrame([
        {
            "Ticker": r["ticker"],
            "Model": r["model"],
            "Accuracy": f"{r['accuracy']:.4f}",
            "Precision": f"{r['precision']:.4f}",
            "Recall": f"{r['recall']:.4f}",
            "F1 Score": f"{r['f1_score']:.4f}",
            "TP": r["true_positives"],
            "FP": r["false_positives"],
            "TN": r["true_negatives"],
            "FN": r["false_negatives"],
        }
        for r in all_results
    ])

    # Save results to CSV
    results_dir = ensure_directory(RESULTS_DIR)
    csv_path = os.path.join(results_dir, "model_comparison.csv")
    comparison_df.to_csv(csv_path, index=False)
    logger.info("\nResults saved to: %s", csv_path)

    # Save detailed metrics for each model
    for result in all_results:
        ticker = result["ticker"]
        model_name = result["model"]
        report_path = os.path.join(
            results_dir, f"{ticker}_{model_name}_classification_report.txt"
        )
        with open(report_path, "w") as f:
            f.write(f"Classification Report: {ticker} - {model_name}\n")
            f.write("=" * 60 + "\n\n")
            f.write(result["classification_report"])
            f.write(f"\n\nConfusion Matrix:\n")
            f.write(f"True Negatives:  {result['true_negatives']}\n")
            f.write(f"False Positives: {result['false_positives']}\n")
            f.write(f"False Negatives: {result['false_negatives']}\n")
            f.write(f"True Positives:  {result['true_positives']}\n")
        logger.info("  Saved report: %s", report_path)

    logger.info("\n" + "=" * 60)
    logger.info("EVALUATION COMPLETE")
    logger.info("=" * 60)

    return comparison_df


# ─────────────────────────────────────────────────────────────────────────────
# QUICK TEST
# ─────────────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    from data_collector import collect_all_data
    from sentiment_analyzer import run_sentiment_pipeline
    from dataset_builder import build_dataset
    from models import build_all_models
    from trainer import train_all_models

    price_data, text_data = collect_all_data()
    _, sentiment_df = run_sentiment_pipeline(text_data)
    loaders, n_features = build_dataset(price_data, sentiment_df)
    models = build_all_models(n_features)
    results = train_all_models(models, loaders, epochs=5)
    comparison_df = evaluate_all(results, loaders)

    print("\n" + "=" * 80)
    print("  FINAL EVALUATION RESULTS")
    print("=" * 80)
    print(comparison_df.to_string(index=False))
    print("=" * 80)
