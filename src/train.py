"""Step 4 - Train the LSTM and 1D-CNN with early stopping on the validation set.

Usage:
    python -m src.train                       # train lstm + cnn with guide settings
    python -m src.train --models cnn --epochs 10
"""
from __future__ import annotations

import argparse
import json
import time

import numpy as np
import torch
from torch.utils.data import DataLoader

from .build_dataset import load_dataset
from .config import BATCH_SIZE, LR, MAX_EPOCHS, MODELS_DIR, PATIENCE, SEED
from .data_prep import WindowDataset, prepare
from .models import PinballLoss, build_model


def get_device() -> torch.device:
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


@torch.no_grad()
def predict(model: torch.nn.Module, ds: WindowDataset, device=None, batch_size: int = 256) -> np.ndarray:
    """Scaled quantile forecasts, shape (N, 24, 2, 3)."""
    device = device or next(model.parameters()).device
    model.eval()
    out = [model(x.to(device)).cpu().numpy() for x, _ in DataLoader(ds, batch_size=batch_size)]
    return np.concatenate(out) if out else np.empty((0, 24, 2, 3), dtype=np.float32)


def train_model(name: str, train_ds: WindowDataset, val_ds: WindowDataset, epochs: int = MAX_EPOCHS,
                patience: int = PATIENCE, lr: float = LR, batch_size: int = BATCH_SIZE,
                device=None, verbose: bool = True):
    torch.manual_seed(SEED)
    np.random.seed(SEED)
    device = device or get_device()
    model = build_model(name).to(device)
    loss_fn = PinballLoss().to(device)
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, drop_last=True)

    best, best_state, wait, history = float("inf"), None, 0, []
    for epoch in range(1, epochs + 1):
        t0 = time.time()
        model.train()
        losses = []
        for x, y in loader:
            x, y = x.to(device), y.to(device)
            opt.zero_grad()
            loss = loss_fn(model(x), y)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            losses.append(loss.item())
        val_pred = torch.from_numpy(predict(model, val_ds, device))
        val_y = torch.stack([val_ds[k][1] for k in range(len(val_ds))])
        val_loss = loss_fn.cpu()(val_pred, val_y).item()
        loss_fn.to(device)
        history.append({"epoch": epoch, "train": float(np.mean(losses)), "val": val_loss})
        if verbose:
            print(f"[{name}] epoch {epoch:2d}  train {np.mean(losses):.4f}  val {val_loss:.4f}  "
                  f"({time.time() - t0:.0f}s)")
        if val_loss < best - 1e-5:
            best, wait = val_loss, 0
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
        else:
            wait += 1
            if wait >= patience:
                if verbose:
                    print(f"[{name}] early stopping - best val pinball {best:.4f}")
                break
    model.load_state_dict(best_state)
    return model, history


def save_model(model: torch.nn.Module, name: str, history: list) -> None:
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    torch.save(model.cpu().state_dict(), MODELS_DIR / f"{name}.pt")
    (MODELS_DIR / f"{name}_history.json").write_text(json.dumps(history, indent=1))


def load_model(name: str) -> torch.nn.Module:
    model = build_model(name)
    model.load_state_dict(torch.load(MODELS_DIR / f"{name}.pt", map_location="cpu"))
    return model.eval()


def main(model_names, epochs):
    df = load_dataset()
    full, values, scaler, w = prepare(df, fit=True)
    train_ds = WindowDataset(values, w["train"].starts)
    val_ds = WindowDataset(values, w["val"].starts)
    print(f"windows: train {len(train_ds)}, val {len(val_ds)}, test {len(w['test'].starts)}  "
          f"device: {get_device()}")
    for name in model_names:
        model, hist = train_model(name, train_ds, val_ds, epochs=epochs)
        save_model(model, name, hist)
        print(f"saved models/{name}.pt")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--models", nargs="+", default=["lstm", "cnn"], choices=["lstm", "cnn"])
    p.add_argument("--epochs", type=int, default=MAX_EPOCHS)
    a = p.parse_args()
    main(a.models, a.epochs)
