"""
main.py
=======
End-to-end pipeline for the Real-Time Market Movement Prediction System.

Run with:
    python main.py

This single entry point orchestrates the full workflow:

    STEP 1 - Data Collection      (data_collector.py)
    STEP 2 - Sentiment Analysis   (sentiment_analyzer.py)
    STEP 3 - Dataset Building     (dataset_builder.py)
    STEP 4 - Model Training       (trainer.py  +  models.py)
    STEP 5 - Evaluation & Results  (evaluator.py)

Every data source has graceful fallback to synthetic data so the pipeline
never breaks even without network access.

Outputs:
    - checkpoints/<ticker>_<model>.pt   - saved model weights
    - results/model_comparison.csv      - metrics table
    - plots/*.png                       - training curves, predictions, comparisons
"""

import logging
import os
import sys
import time
import warnings

warnings.filterwarnings("ignore")

# -------------------------------------------------------------------------
# LOGGING SETUP
# -------------------------------------------------------------------------

logging.basicConfig(
    level   = logging.INFO,
    format  = "%(asctime)s [%(levelname)s] %(message)s",
    datefmt = "%H:%M:%S",
    handlers=[
        logging.StreamHandler(sys.stdout),
    ],
)
logger = logging.getLogger(__name__)


def main():
    """Run the complete market prediction pipeline."""
    start = time.time()

    logger.info("+----------------------------------------------------------+")
    logger.info("|   REAL-TIME MARKET MOVEMENT PREDICTION SYSTEM            |")
    logger.info("|   RNN  -  LSTM  -  GRU  with Sentiment Features         |")
    logger.info("+----------------------------------------------------------+")
    logger.info("")

    # =====================================================================
    # STEP 1 - DATA COLLECTION
    # =====================================================================
    from data_collector import collect_all_data
    price_data, text_data = collect_all_data()

    # =====================================================================
    # STEP 2 - SENTIMENT ANALYSIS
    # =====================================================================
    from sentiment_analyzer import run_sentiment_pipeline
    analyzed_items, sentiment_df = run_sentiment_pipeline(
        text_data,
        time_window="1D",    # daily aggregation to match daily price data
    )

    # =====================================================================
    # STEP 3 - DATASET BUILDING
    # =====================================================================
    from dataset_builder import build_dataset
    loaders, n_features = build_dataset(
        price_data   = price_data,
        sentiment_df = sentiment_df,
        seq_len      = 10,
        batch_size   = 32,
    )

    # =====================================================================
    # STEP 4 - MODEL BUILDING & TRAINING
    # =====================================================================
    from models  import build_all_models
    from trainer import train_all_models

    models = build_all_models(
        n_features  = n_features,
        hidden_size = 64,
        num_layers  = 2,
        dropout     = 0.3,
    )

    # Print model architectures
    logger.info("\nModel architectures:")
    for name, model in models.items():
        n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        logger.info("  %s: %s parameters = %s", name, model.extra_repr(), f"{n_params:,}")

    train_results = train_all_models(
        models  = models,
        loaders = loaders,
        epochs  = 50,
        lr      = 1e-3,
    )

    # =====================================================================
    # STEP 5 - EVALUATION & RESULTS
    # =====================================================================
    from evaluator import evaluate_all
    comparison_df = evaluate_all(train_results, loaders)

    # =====================================================================
    # SUMMARY
    # =====================================================================
    elapsed = time.time() - start
    logger.info("\n" + "=" * 60)
    logger.info("  PIPELINE COMPLETE  (%.1f seconds)", elapsed)
    logger.info("=" * 60)
    logger.info("")
    logger.info("  Saved artifacts:")
    logger.info("    - Model checkpoints : checkpoints/")
    logger.info("    - Results CSV       : results/model_comparison.csv")
    logger.info("    - Plots             : plots/")
    logger.info("")

    # Print the final comparison one more time for visibility
    print("\n" + "=" * 80)
    print("  FINAL MODEL COMPARISON TABLE")
    print("=" * 80)
    print(comparison_df.to_string(index=False))
    print("=" * 80)

    return comparison_df


if __name__ == "__main__":
    main()
