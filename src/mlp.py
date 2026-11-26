import numpy as np
import torch
from torch import nn

N_LABELS = 217
DEFAULT_MLP_SETTINGS = {"hidden_sizes": (512,), "dropout": 0.5}


class SkillMLP(nn.Module):
    def __init__(self, input_dim: int, n_labels: int = N_LABELS,
                 hidden_sizes=(512,), dropout: float = 0.5):
        super().__init__()
        hidden_sizes = tuple(hidden_sizes)
        if not 1 <= len(hidden_sizes) <= 2:
            raise ValueError(f"SkillMLP uses 1-2 hidden layers, got {len(hidden_sizes)}")
        if not 0 <= dropout < 1:
            raise ValueError(f"dropout must be in [0, 1), got {dropout}")

        layers, in_size = [], input_dim
        for size in hidden_sizes:
            layers += [nn.Linear(in_size, size), nn.ReLU(), nn.Dropout(dropout)]
            in_size = size
        layers.append(nn.Linear(in_size, n_labels))
        self.net = nn.Sequential(*layers)

        self.input_dim = input_dim
        self.n_labels = n_labels

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)

    @torch.no_grad()
    def predict_proba(self, x: torch.Tensor) -> torch.Tensor:
        was_training = self.training
        self.eval()
        probs = torch.sigmoid(self.forward(x))
        self.train(was_training)
        return probs


def build_mlp(input_dim: int, n_labels: int = N_LABELS, seed: int | None = 42,
              **settings) -> SkillMLP:
    if seed is not None:
        torch.manual_seed(seed)
    return SkillMLP(input_dim, n_labels, **{**DEFAULT_MLP_SETTINGS, **settings})


def build_loss(pos_weight=None) -> nn.BCEWithLogitsLoss:
    if pos_weight is not None:
        pos_weight = torch.as_tensor(pos_weight, dtype=torch.float32)
    return nn.BCEWithLogitsLoss(pos_weight=pos_weight)


# Rare labels have hundreds of negatives per positive; uncapped weights that
# large make the model predict them everywhere, so the weight is capped.
DEFAULT_POS_WEIGHT_CAP = 10.0


def compute_pos_weight(y_train, max_weight: float | None = DEFAULT_POS_WEIGHT_CAP) -> np.ndarray:
    y = np.asarray(y_train, dtype=np.float64)
    positives = y.sum(axis=0)
    negatives = y.shape[0] - positives
    weight = np.divide(negatives, positives, out=np.ones_like(positives),
                       where=positives > 0)
    upper = np.inf if max_weight is None else max_weight
    return np.clip(weight, 1.0, upper).astype(np.float32)