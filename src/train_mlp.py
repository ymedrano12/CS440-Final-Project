import copy

import numpy as np
import torch
from scipy import sparse

from evaluate import timer
from mlp import build_loss

DEFAULT_TRAIN_SETTINGS = {
    "epochs": 50,
    "batch_size": 64,
    "lr": 1e-3,
    "weight_decay": 1e-5,
    "patience": 5,
}


def _get_device(device=None) -> torch.device:
    if device is not None:
        return torch.device(device)
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def _to_tensor(X, rows=None) -> torch.Tensor:
    if hasattr(X, "to_numpy"):
        X = X.to_numpy()
    if rows is not None:
        X = X[rows]
    if sparse.issparse(X):
        X = X.toarray()
    return torch.as_tensor(np.asarray(X, dtype=np.float32))


def _batches(n_rows: int, batch_size: int, generator=None):
    order = (torch.randperm(n_rows, generator=generator).numpy()
             if generator is not None else np.arange(n_rows))
    for start in range(0, n_rows, batch_size):
        yield order[start:start + batch_size]


@torch.no_grad()
def _mean_loss(model, loss_fn, X, y, batch_size, device) -> float:
    model.eval()
    total, n = 0.0, X.shape[0]
    for rows in _batches(n, batch_size):
        xb, yb = _to_tensor(X, rows).to(device), _to_tensor(y, rows).to(device)
        total += loss_fn(model(xb), yb).item() * len(rows)
    return total / n


@torch.no_grad()
def predict_proba(model, X, batch_size: int = 512, device=None) -> np.ndarray:
    device = _get_device(device)
    model.to(device)
    out = [model.predict_proba(_to_tensor(X, rows).to(device)).cpu().numpy()
           for rows in _batches(X.shape[0], batch_size)]
    return np.vstack(out)


def train_mlp(model, X_train, y_train, X_val=None, y_val=None, *,
              pos_weight=None, seed: int = 42, device=None, verbose: bool = True,
              **settings) -> dict:
    cfg = {**DEFAULT_TRAIN_SETTINGS, **settings}
    unknown = set(cfg) - set(DEFAULT_TRAIN_SETTINGS)
    if unknown:
        raise ValueError(f"Unknown training settings: {sorted(unknown)}")
    if X_train.shape[0] != y_train.shape[0]:
        raise ValueError(f"X_train has {X_train.shape[0]} rows, y_train {y_train.shape[0]}")
    if y_train.shape[1] != model.n_labels:
        raise ValueError(f"y_train has {y_train.shape[1]} labels, model expects {model.n_labels}")
    has_val = X_val is not None and y_val is not None

    device = _get_device(device)
    model.to(device)
    loss_fn = build_loss(pos_weight).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=cfg["lr"],
                                 weight_decay=cfg["weight_decay"])
    generator = torch.Generator().manual_seed(seed)
    torch.manual_seed(seed)

    history = {"train_loss": [], "val_loss": [], "best_epoch": None,
               "epochs_run": 0, "settings": cfg}
    best_val, best_state, bad_epochs = float("inf"), None, 0
    n_train = X_train.shape[0]

    with timer() as t:
        for epoch in range(1, cfg["epochs"] + 1):
            model.train()
            total = 0.0
            for rows in _batches(n_train, cfg["batch_size"], generator):
                xb = _to_tensor(X_train, rows).to(device)
                yb = _to_tensor(y_train, rows).to(device)
                optimizer.zero_grad()
                loss = loss_fn(model(xb), yb)
                loss.backward()
                optimizer.step()
                total += loss.item() * len(rows)
            history["train_loss"].append(total / n_train)
            history["epochs_run"] = epoch

            msg = f"epoch {epoch:3d} | train loss {history['train_loss'][-1]:.4f}"
            if has_val:
                val_loss = _mean_loss(model, loss_fn, X_val, y_val,
                                      cfg["batch_size"], device)
                history["val_loss"].append(val_loss)
                msg += f" | val loss {val_loss:.4f}"
                if val_loss < best_val:
                    best_val, bad_epochs = val_loss, 0
                    best_state = copy.deepcopy(model.state_dict())
                    history["best_epoch"] = epoch
                else:
                    bad_epochs += 1
            if verbose:
                print(msg)
            if has_val and cfg["patience"] is not None and bad_epochs >= cfg["patience"]:
                if verbose:
                    print(f"Early stop: no val improvement for {cfg['patience']} epochs.")
                break

    if best_state is not None:
        model.load_state_dict(best_state)
        if verbose:
            print(f"Restored weights from epoch {history['best_epoch']} "
                  f"(val loss {best_val:.4f}).")
    else:
        history["best_epoch"] = history["epochs_run"]
    history["train_seconds"] = t["seconds"]
    model.eval()
    return history