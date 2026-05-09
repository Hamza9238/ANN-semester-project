"""
trainer.py
==========
Trains all three sequence models (RNN, LSTM, GRU) on the prepared dataset.

Training loop features:
  - Combined loss = CrossEntropy (classification) + MSE (regression), weighted
  - Gradient clipping to prevent exploding gradients (critical for vanilla RNN)
  - Early stopping on validation loss with configurable patience
  - Per-epoch logging of train / val losses
  - Saves best model checkpoint to disk for each model
  - Returns training history for downstream plotting

The trainer is model-agnostic: it receives a dict of {name: model} and
trains each one with identical settings for a fair comparison.
"""

import copy
import logging
import os
import time

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

from helpers import get_device, ensure_directory, print_step_header

logger = logging.getLogger(__name__)

# ─────────────────────────────────────────────────────────────────────────────
# CONFIGURATION
# ─────────────────────────────────────────────────────────────────────────────

EPOCHS        = 50        # maximum epochs per model
LEARNING_RATE = 1e-3
WEIGHT_DECAY  = 1e-5      # L2 regularisation
GRAD_CLIP     = 1.0       # max gradient norm
PATIENCE      = 8         # early-stopping patience (epochs without improvement)

# How much weight to give the regression loss vs classification loss
# total_loss = cls_weight * CE_loss + reg_weight * MSE_loss
CLS_WEIGHT = 1.0
REG_WEIGHT = 0.3

# Directory to save model checkpoints
CHECKPOINT_DIR = os.path.join(os.path.dirname(__file__), "checkpoints")


# ─────────────────────────────────────────────────────────────────────────────
# SINGLE-MODEL TRAINING LOOP
# ─────────────────────────────────────────────────────────────────────────────

def _train_one_model(
    model:         nn.Module,
    model_name:    str,
    train_loader:  torch.utils.data.DataLoader,
    val_loader:    torch.utils.data.DataLoader,
    device:        torch.device,
    epochs:        int = EPOCHS,
    lr:            float = LEARNING_RATE,
    patience:      int = PATIENCE,
) -> dict:
    """
    Train a single model and return its history.

    Returns
    -------
    dict with keys:
        train_losses  : list[float]   - per-epoch average training loss
        val_losses    : list[float]   - per-epoch average validation loss
        best_epoch    : int           - epoch with lowest validation loss
        best_val_loss : float
        wall_time     : float         - total training wall-clock seconds
        state_dict    : OrderedDict   - best model weights
    """
    model.to(device)

    # ── Loss functions ────────────────────────────────────────────────────
    cls_criterion = nn.CrossEntropyLoss()
    reg_criterion = nn.MSELoss()

    # ── Optimiser with weight decay (AdamW) ───────────────────────────────
    optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=WEIGHT_DECAY)

    # ── Learning-rate scheduler: reduce on plateau ────────────────────────
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", factor=0.5, patience=3,
    )

    # ── History tracking ──────────────────────────────────────────────────
    train_losses = []
    val_losses   = []
    best_val     = float("inf")
    best_epoch   = 0
    best_state   = None
    patience_ctr = 0
    start_time   = time.time()

    for epoch in range(1, epochs + 1):
        # ──────────── TRAINING PHASE ──────────────────────────────────────
        model.train()
        batch_losses = []

        for X_batch, y_cls_batch, y_reg_batch in train_loader:
            X_batch     = X_batch.to(device)
            y_cls_batch = y_cls_batch.to(device)
            y_reg_batch = y_reg_batch.to(device)

            optimizer.zero_grad()

            logits, reg_pred = model(X_batch)

            loss_cls = cls_criterion(logits, y_cls_batch)
            loss_reg = reg_criterion(reg_pred, y_reg_batch)
            loss     = CLS_WEIGHT * loss_cls + REG_WEIGHT * loss_reg

            loss.backward()

            # Gradient clipping (essential for vanilla RNN stability)
            torch.nn.utils.clip_grad_norm_(model.parameters(), GRAD_CLIP)

            optimizer.step()
            batch_losses.append(loss.item())

        train_loss = np.mean(batch_losses)
        train_losses.append(train_loss)

        # ──────────── VALIDATION PHASE ────────────────────────────────────
        model.eval()
        val_batch_losses = []

        with torch.no_grad():
            for X_batch, y_cls_batch, y_reg_batch in val_loader:
                X_batch     = X_batch.to(device)
                y_cls_batch = y_cls_batch.to(device)
                y_reg_batch = y_reg_batch.to(device)

                logits, reg_pred = model(X_batch)

                loss_cls = cls_criterion(logits, y_cls_batch)
                loss_reg = reg_criterion(reg_pred, y_reg_batch)
                loss     = CLS_WEIGHT * loss_cls + REG_WEIGHT * loss_reg

                val_batch_losses.append(loss.item())

        val_loss = np.mean(val_batch_losses)
        val_losses.append(val_loss)

        # Step the LR scheduler
        scheduler.step(val_loss)

        # ──────────── EARLY STOPPING CHECK ────────────────────────────────
        if val_loss < best_val:
            best_val     = val_loss
            best_epoch   = epoch
            best_state   = copy.deepcopy(model.state_dict())
            patience_ctr = 0
        else:
            patience_ctr += 1

        # Log every 5 epochs or on improvement
        if epoch % 5 == 0 or patience_ctr == 0:
            lr_now = optimizer.param_groups[0]["lr"]
            logger.info(
                "  [%s] Epoch %3d/%d  train=%.4f  val=%.4f  lr=%.1e %s",
                model_name, epoch, epochs, train_loss, val_loss, lr_now,
                "*" if patience_ctr == 0 else "",
            )

        if patience_ctr >= patience:
            logger.info("  [%s] Early stopping at epoch %d (best=%d)", model_name, epoch, best_epoch)
            break

    wall_time = time.time() - start_time

    # Restore best weights
    if best_state is not None:
        model.load_state_dict(best_state)

    return {
        "train_losses":  train_losses,
        "val_losses":    val_losses,
        "best_epoch":    best_epoch,
        "best_val_loss": best_val,
        "wall_time":     wall_time,
        "state_dict":    best_state or model.state_dict(),
    }


# ─────────────────────────────────────────────────────────────────────────────
# SAVE MODEL CHECKPOINT
# ─────────────────────────────────────────────────────────────────────────────

def _save_checkpoint(model_name: str, state_dict: dict, ticker: str):
    """Save model weights to disk under checkpoints/<ticker>_<model>.pt."""
    path = os.path.join(
        ensure_directory(CHECKPOINT_DIR),
        f"{ticker}_{model_name}.pt"
    )
    torch.save(state_dict, path)
    logger.info("  Saved checkpoint -> %s", path)


# ─────────────────────────────────────────────────────────────────────────────
# MASTER TRAINER  (called by main.py)
# ─────────────────────────────────────────────────────────────────────────────

def train_all_models(
    models:     dict[str, nn.Module],
    loaders:    dict,
    epochs:     int = EPOCHS,
    lr:         float = LEARNING_RATE,
) -> dict:
    """
    Train every model on every ticker and return combined training histories.

    Parameters
    ----------
    models  : {"RNN": model, "LSTM": model, "GRU": model}
    loaders : output of dataset_builder.build_dataset()
    epochs  : max training epochs
    lr      : initial learning rate

    Returns
    -------
    results : dict[ticker][model_name] -> training history dict
    """
    print_step_header(4, "MODEL TRAINING")

    device = get_device()

    results = {}

    for ticker, ticker_loaders in loaders.items():
        logger.info("\n-- Ticker: %s --", ticker)
        results[ticker] = {}

        train_loader = ticker_loaders["train"]
        val_loader   = ticker_loaders["val"]

        for model_name, model_template in models.items():
            logger.info("Training %s on %s ...", model_name, ticker)

            # Deep-copy so each ticker gets a fresh model
            model = copy.deepcopy(model_template)

            history = _train_one_model(
                model        = model,
                model_name   = model_name,
                train_loader = train_loader,
                val_loader   = val_loader,
                device       = device,
                epochs       = epochs,
                lr           = lr,
            )

            # Save best weights
            _save_checkpoint(model_name, history["state_dict"], ticker)

            # Store model reference for evaluation
            history["model"] = model
            results[ticker][model_name] = history

            logger.info(
                "  %s %s -> best_val=%.4f @ epoch %d (%.1fs)",
                ticker, model_name,
                history["best_val_loss"], history["best_epoch"], history["wall_time"],
            )

    return results


# ─────────────────────────────────────────────────────────────────────────────
# QUICK TEST
# ─────────────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    from data_collector    import collect_all_data
    from sentiment_analyzer import run_sentiment_pipeline
    from dataset_builder   import build_dataset
    from models            import build_all_models

    price_data, text_data = collect_all_data()
    _, sentiment_df       = run_sentiment_pipeline(text_data)
    loaders, n_features   = build_dataset(price_data, sentiment_df)
    models                = build_all_models(n_features)
    results               = train_all_models(models, loaders, epochs=5)

    for ticker in results:
        for name in results[ticker]:
            h = results[ticker][name]
            print(f"{ticker} {name}: best_val={h['best_val_loss']:.4f} @ epoch {h['best_epoch']}")
