import copy

import numpy as np
import torch
from scipy import sparse

from evaluate import timer
from mlp import build_loss

DEFAULT_TRAIN_SETTINGS = {
    "lr": 1e-3,
    "weight_decay": 0.0,
    "batch_size": 64,
    "max_epochs": 50,
    "patience": 5,
}


def get_device() -> torch.device:
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def _to_tensor(X, rows, device) -> torch.Tensor:
    batch = X[rows]
    batch = batch.toarray() if sparse.issparse(batch) else np.asarray(batch)
    return torch.as_tensor(batch, dtype=torch.float32, device=device)


def _batches(n_rows: int, batch_size: int, order=None):
    order = np.arange(n_rows) if order is None else order
    for start in range(0, n_rows, batch_size):
        yield order[start:start + batch_size]


def _check_inputs(model, X, y, name: str):
    if X.shape[0] != y.shape[0]:
        raise ValueError(f"{name}: {X.shape[0]} feature rows but {y.shape[0]} label rows")
    if X.shape[1] != model.input_dim:
        raise ValueError(f"{name}: {X.shape[1]} features but the model expects {model.input_dim}")
    if y.shape[1] != model.n_labels:
        raise ValueError(f"{name}: {y.shape[1]} labels but the model has {model.n_labels} outputs")


@torch.no_grad()
def _mean_loss(model, loss_fn, X, y, batch_size, device) -> float:
    model.eval()
    total = 0.0
    for rows in _batches(X.shape[0], batch_size):
        logits = model(_to_tensor(X, rows, device))
        total += loss_fn(logits, torch.as_tensor(y[rows], device=device)).item() * len(rows)
    return total / X.shape[0]


def train_mlp(model, X_train, y_train, X_val=None, y_val=None, loss_fn=None,
              seed: int | None = 42, verbose: bool = True, **settings) -> dict:
    unknown = set(settings) - set(DEFAULT_TRAIN_SETTINGS)
    if unknown:
        raise ValueError(f"Unknown training settings: {sorted(unknown)}")
    cfg = {**DEFAULT_TRAIN_SETTINGS, **settings}
    has_val = X_val is not None and y_val is not None

    y_train = np.asarray(y_train, dtype=np.float32)
    _check_inputs(model, X_train, y_train, "training data")
    if has_val:
        y_val = np.asarray(y_val, dtype=np.float32)
        _check_inputs(model, X_val, y_val, "validation data")

    if seed is not None:
        torch.manual_seed(seed)
    shuffler = np.random.default_rng(seed)

    device = get_device()
    model.to(device)
    loss_fn = (loss_fn if loss_fn is not None else build_loss()).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=cfg["lr"],
                                 weight_decay=cfg["weight_decay"])

    n_rows = X_train.shape[0]
    train_losses, val_losses = [], []
    best_loss, best_state, best_epoch, waited = float("inf"), None, 0, 0

    with timer() as t_train:
        for epoch in range(1, cfg["max_epochs"] + 1):
            model.train()
            running = 0.0
            for rows in _batches(n_rows, cfg["batch_size"], shuffler.permutation(n_rows)):
                xb = _to_tensor(X_train, rows, device)
                yb = torch.as_tensor(y_train[rows], device=device)
                optimizer.zero_grad()
                loss = loss_fn(model(xb), yb)
                loss.backward()
                optimizer.step()
                running += loss.item() * len(rows)
            train_losses.append(running / n_rows)

            msg = f"epoch {epoch:3d} | train loss {train_losses[-1]:.4f}"
            if has_val:
                val_losses.append(_mean_loss(model, loss_fn, X_val, y_val,
                                             cfg["batch_size"], device))
                msg += f" | val loss {val_losses[-1]:.4f}"
                if val_losses[-1] < best_loss:
                    best_loss, best_epoch, waited = val_losses[-1], epoch, 0
                    best_state = copy.deepcopy(model.state_dict())
                else:
                    waited += 1
            if verbose:
                print(msg)
            if has_val and waited >= cfg["patience"]:
                if verbose:
                    print(f"Early stop: no val improvement for {cfg['patience']} epochs.")
                break

    if best_state is not None:
        model.load_state_dict(best_state)
    else:
        best_epoch = len(train_losses)
    model.eval()
    if verbose:
        print(f"Best epoch: {best_epoch} | training time {t_train['seconds']:.1f} s on {device}")

    return {
        "train_settings": cfg,
        "device": str(device),
        "epochs_run": len(train_losses),
        "best_epoch": best_epoch,
        "stopped_early": len(train_losses) < cfg["max_epochs"],
        "train_seconds": t_train["seconds"],
        "train_loss": [round(v, 6) for v in train_losses],
        "val_loss": [round(v, 6) for v in val_losses],
    }


def predict_proba(model, X, batch_size: int = 1024) -> np.ndarray:
    device = next(model.parameters()).device
    out = [model.predict_proba(_to_tensor(X, rows, device)).cpu().numpy()
           for rows in _batches(X.shape[0], batch_size)]
    return np.vstack(out) if out else np.zeros((0, model.n_labels), dtype=np.float32)