"""Bounded CPU adapter for the official TabM parameter-efficient ensemble.

Training follows https://github.com/yandex-research/tabm: optimize each member's
loss separately, then average probabilities at inference. This experiment uses
numeric features without feature embeddings and does not access held-out labels.
"""

import math
from contextlib import contextmanager
from dataclasses import dataclass

import numpy as np
from sklearn.preprocessing import StandardScaler

_DEFAULTS = {
    "n_blocks": 2,
    "d_block": 32,
    "k": 8,
    "dropout": 0.1,
    "epochs": 8,
    "batch_size": 256,
    "learning_rate": 0.002,
    "weight_decay": 0.01,
}


def _config(config, seed):
    if not isinstance(config, dict) or set(config) - _DEFAULTS.keys():
        raise ValueError("Unknown TabM configuration fields.")
    resolved = _DEFAULTS | config
    for name, high in {
        "n_blocks": 4,
        "d_block": 256,
        "k": 32,
        "epochs": 50,
        "batch_size": 4096,
    }.items():
        if type(resolved[name]) is not int or not 1 <= resolved[name] <= high:
            raise ValueError(f"Invalid TabM configuration: {name}.")
    for name in ("dropout", "learning_rate", "weight_decay"):
        value = resolved[name]
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            raise ValueError(f"Invalid TabM configuration: {name}.")
        if not math.isfinite(value) or not 0 <= value <= 1:
            raise ValueError(f"Invalid TabM configuration: {name}.")
    if resolved["dropout"] == 1 or resolved["learning_rate"] == 0:
        raise ValueError("TabM needs dropout < 1 and a positive learning rate.")
    if type(seed) is not int or not 0 <= seed < 2**32:
        raise ValueError("TabM seed must be an integer in [0, 2**32).")
    return resolved.copy()


def _matrix(x, *, n_features=None, empty_ok=False):
    array = np.asarray(x, dtype=np.float64)
    if array.ndim != 2 or array.shape[1] < 1 or (not empty_ok and len(array) == 0):
        raise ValueError("TabM features must be a nonempty two-dimensional matrix.")
    if n_features is not None and array.shape[1] != n_features:
        raise ValueError("TabM prediction feature count differs from training.")
    if not np.isfinite(array).all():
        raise ValueError("TabM features must be finite; unavailable rows must be excluded.")
    return array


def _dependencies():
    try:
        import tabm
        import torch
    except ImportError as exc:
        raise RuntimeError("The optional TabM experiment requires torch and tabm.") from exc
    return torch, tabm


@contextmanager
def _single_thread(torch):
    previous = torch.get_num_threads()
    try:
        torch.set_num_threads(1)
        yield
    finally:
        torch.set_num_threads(previous)


@contextmanager
def _training_runtime(torch, seed):
    deterministic = torch.are_deterministic_algorithms_enabled()
    warn_only = torch.is_deterministic_algorithms_warn_only_enabled()
    with _single_thread(torch), torch.random.fork_rng(devices=[]):
        try:
            # Seed only the CPU generator, preserving unrelated CUDA RNG state.
            torch.random.default_generator.manual_seed(seed)
            torch.use_deterministic_algorithms(True)
            yield
        finally:
            torch.use_deterministic_algorithms(deterministic, warn_only=warn_only)


def _transform(scaler, x):
    transformed = scaler.transform(x)
    if not np.isfinite(transformed).all():
        raise ValueError("TabM standardization produced non-finite features.")
    return np.clip(transformed, -10, 10).astype(np.float32)


@dataclass
class TabMEstimator:
    """A fixed fitted CPU model; prediction never fits or updates its scaler."""

    model_: object
    scaler_: StandardScaler
    config: dict
    training_audit: dict

    @property
    def classes_(self):
        return np.array([0, 1], dtype=np.int64)

    def predict_proba(self, x):
        x = _matrix(x, n_features=self.scaler_.n_features_in_, empty_ok=True)
        if len(x) == 0:
            return np.empty((0, 2), dtype=np.float64)
        torch, _ = _dependencies()
        probabilities = np.empty(len(x), dtype=np.float64)
        batch_size = self.config["batch_size"]
        with _single_thread(torch), torch.inference_mode():
            for start in range(0, len(x), batch_size):
                stop = start + batch_size
                batch = torch.from_numpy(_transform(self.scaler_, x[start:stop]))
                logits = self.model_(batch).squeeze(-1)
                probabilities[start:stop] = logits.sigmoid().mean(dim=1).cpu().numpy()
        if not np.isfinite(probabilities).all():
            raise ValueError("TabM returned non-finite probabilities.")
        return np.column_stack((1 - probabilities, probabilities))


def fit_tabm(x, y, weights, config, seed):
    """Fit TabM on supplied training rows only, without early stopping.

    Sample weights must be positive and aligned with the training rows. Their
    normalization uses this training fold only; no additional class balancing
    is silently applied. The caller owns split isolation and threshold selection.
    """
    resolved = _config(config, seed)
    x = _matrix(x)
    y = np.asarray(y, dtype=np.float64)
    weights = np.asarray(weights, dtype=np.float64)
    if y.shape != (len(x),) or not np.isfinite(y).all() or set(np.unique(y)) != {0, 1}:
        raise ValueError("TabM training labels must contain both binary classes and match x.")
    if weights.shape != (len(x),) or not np.isfinite(weights).all() or (weights <= 0).any():
        raise ValueError("TabM weights must be finite, positive, and match the training rows.")
    normalized = weights / weights.max()
    normalized = (normalized / normalized.mean()).astype(np.float32)
    if not np.isfinite(normalized).all() or (normalized <= 0).any():
        raise ValueError("TabM weights have an unsupported numeric range.")

    scaler = StandardScaler().fit(x)
    if not all(np.isfinite(value).all() for value in (scaler.mean_, scaler.scale_, scaler.var_)):
        raise ValueError("TabM training features overflowed standardization statistics.")
    transformed = _transform(scaler, x)
    torch, tabm = _dependencies()
    rng = np.random.default_rng(seed)
    losses = []
    with _training_runtime(torch, seed), torch.device("cpu"):
        model = tabm.TabM.make(
            n_num_features=x.shape[1],
            d_out=1,
            arch_type="tabm",
            n_blocks=resolved["n_blocks"],
            d_block=resolved["d_block"],
            k=resolved["k"],
            dropout=resolved["dropout"],
        ).float()
        model.train()
        optimizer = torch.optim.AdamW(
            model.parameters(),
            lr=resolved["learning_rate"],
            weight_decay=resolved["weight_decay"],
        )
        x_tensor = torch.from_numpy(transformed)
        y_tensor = torch.from_numpy(y.astype(np.float32))
        weight_tensor = torch.from_numpy(normalized)
        for _ in range(resolved["epochs"]):
            order = rng.permutation(len(x))
            loss_sum = 0.0
            for start in range(0, len(x), resolved["batch_size"]):
                indices = torch.from_numpy(order[start : start + resolved["batch_size"]])
                logits = model(x_tensor[indices]).squeeze(-1)
                targets = y_tensor[indices, None].expand_as(logits)
                per_member = torch.nn.functional.binary_cross_entropy_with_logits(
                    logits, targets, reduction="none"
                )
                loss = (per_member.mean(dim=1) * weight_tensor[indices]).mean()
                if not torch.isfinite(loss):
                    raise ValueError("TabM training loss became non-finite.")
                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                optimizer.step()
                loss_sum += float(loss.detach()) * len(indices)
            losses.append(loss_sum / len(x))
        model.eval()

    audit = {
        "package": "tabm",
        "package_version": tabm.__version__,
        "architecture": "tabm",
        "rows": len(x),
        "features": x.shape[1],
        "class_counts": {str(label): int((y == label).sum()) for label in (0, 1)},
        "seed": seed,
        "epochs": resolved["epochs"],
        "epoch_train_weighted_bce": losses,
        "sample_weight_mean_after_normalization": float(normalized.mean()),
        "preprocessing": "training-fold StandardScaler then clip [-10, 10]",
        "scaler_mean": scaler.mean_.tolist(),
        "scaler_scale": scaler.scale_.tolist(),
        "early_stopping": False,
        "device": "cpu",
        "cpu_threads": 1,
        "loss_reduction": "mean member BCE, then mean training-normalized weighted row loss",
        "prediction_reduction": "mean member probabilities",
        "config": resolved.copy(),
    }
    return TabMEstimator(model, scaler, resolved.copy(), audit)
