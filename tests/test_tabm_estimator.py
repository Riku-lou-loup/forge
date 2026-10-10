"""Synthetic checks for the optional neural estimator; no recordings are read."""

import numpy as np
import pytest

from forge.ml.tabm_estimator import fit_tabm

torch = pytest.importorskip("torch")
tabm = pytest.importorskip("tabm")


@pytest.fixture
def training():
    rng = np.random.default_rng(91)
    x = rng.normal(size=(48, 4))
    y = (x[:, 0] > 0).astype(int)
    weights = np.linspace(0.5, 2, len(x))
    config = {"n_blocks": 1, "d_block": 8, "k": 2, "epochs": 2, "batch_size": 16}
    return x, y, weights, config


def test_deterministic_fit_and_global_state_restoration(training):
    x, y, weights, config = training
    rng_state = torch.random.get_rng_state().clone()
    np_state = np.random.get_state()
    threads = torch.get_num_threads()
    deterministic = torch.are_deterministic_algorithms_enabled()
    warn_only = torch.is_deterministic_algorithms_warn_only_enabled()
    first = fit_tabm(x, y, weights, config, seed=13)
    second = fit_tabm(x, y, weights, config, seed=13)
    np.testing.assert_array_equal(first.predict_proba(x), second.predict_proba(x))
    assert first.training_audit == second.training_audit
    assert torch.equal(rng_state, torch.random.get_rng_state())
    assert np_state[0] == np.random.get_state()[0]
    np.testing.assert_array_equal(np_state[1], np.random.get_state()[1])
    assert np_state[2:] == np.random.get_state()[2:]
    assert torch.get_num_threads() == threads
    assert torch.are_deterministic_algorithms_enabled() == deterministic
    assert torch.is_deterministic_algorithms_warn_only_enabled() == warn_only


def test_actual_tabm_probabilities_and_independent_member_inference(training):
    x, y, weights, config = training
    fitted = fit_tabm(x, y, weights, config, seed=13)
    assert isinstance(fitted.model_, tabm.TabM)
    assert not fitted.model_.training
    assert next(fitted.model_.parameters()).device.type == "cpu"
    probabilities = fitted.predict_proba(x)
    assert probabilities.shape == (len(x), 2)
    assert ((probabilities >= 0) & (probabilities <= 1)).all()
    np.testing.assert_allclose(probabilities.sum(axis=1), 1)
    np.testing.assert_array_equal(fitted.classes_, [0, 1])
    transformed = np.clip(fitted.scaler_.transform(x[:5]), -10, 10).astype(np.float32)
    with torch.inference_mode():
        expected = fitted.model_(torch.from_numpy(transformed)).sigmoid().mean(dim=1)[:, 0]
    np.testing.assert_allclose(probabilities[:5, 1], expected.numpy(), rtol=1e-6)
    assert fitted.predict_proba(np.empty((0, 4))).shape == (0, 2)
    assert len(fitted.training_audit["epoch_train_weighted_bce"]) == config["epochs"]
    config["epochs"] = 99
    assert fitted.config["epochs"] == 2
    assert fitted.training_audit["config"]["epochs"] == 2


def test_scaler_uses_training_only_and_predictions_ignore_future_rows(training):
    x, y, weights, config = training
    fitted = fit_tabm(x, y, weights, config, seed=13)
    np.testing.assert_allclose(fitted.scaler_.mean_, x.mean(axis=0))
    np.testing.assert_allclose(fitted.scaler_.var_, x.var(axis=0))
    before = fitted.scaler_.mean_.copy()
    first = fitted.predict_proba(x[:11])
    altered_future = np.full((55, x.shape[1]), 1e10)
    second = fitted.predict_proba(np.vstack((x[:11], altered_future)))[:11]
    np.testing.assert_allclose(first, second, atol=1e-7)
    np.testing.assert_array_equal(before, fitted.scaler_.mean_)
    assert fitted.scaler_.n_samples_seen_ == len(x)


def test_training_weight_scale_does_not_change_fit(training):
    x, y, weights, config = training
    first = fit_tabm(x, y, weights, config, seed=13)
    second = fit_tabm(x, y, 100 * weights, config, seed=13)
    np.testing.assert_allclose(first.predict_proba(x), second.predict_proba(x), atol=1e-7)


@pytest.mark.parametrize(
    "failure",
    [
        "nan_x",
        "empty_x",
        "wrong_y",
        "one_class",
        "nonbinary",
        "nan_y",
        "wrong_weights",
        "zero_weight",
        "nan_weight",
    ],
)
def test_invalid_training_data(training, failure):
    x, y, weights, config = training
    if failure == "nan_x":
        x[0, 0] = np.nan
    elif failure == "empty_x":
        x = x[:0]
    elif failure == "wrong_y":
        y = y[:, None]
    elif failure == "one_class":
        y[:] = 0
    elif failure == "nonbinary":
        y[0] = 2
    elif failure == "nan_y":
        y = y.astype(float)
        y[0] = np.nan
    elif failure == "wrong_weights":
        weights = weights[:2]
    elif failure == "zero_weight":
        weights[0] = 0
    elif failure == "nan_weight":
        weights[0] = np.nan
    with pytest.raises(ValueError):
        fit_tabm(x, y, weights, config, seed=13)


@pytest.mark.parametrize(
    "override",
    [
        {"epochs": 0},
        {"epochs": True},
        {"k": 10000},
        {"dropout": 1},
        {"learning_rate": 0},
        {"weight_decay": np.inf},
        {"unknown": 1},
    ],
)
def test_invalid_config(training, override):
    x, y, weights, config = training
    with pytest.raises(ValueError):
        fit_tabm(x, y, weights, config | override, seed=13)


def test_invalid_prediction_features(training):
    x, y, weights, config = training
    fitted = fit_tabm(x, y, weights, config, seed=13)
    with pytest.raises(ValueError, match="feature count"):
        fitted.predict_proba(np.zeros((3, 5)))
    with pytest.raises(ValueError, match="finite"):
        fitted.predict_proba(np.full((3, 4), np.nan))


def test_global_state_restored_after_training_failure(training, monkeypatch):
    x, y, weights, config = training
    before = torch.random.get_rng_state().clone()
    threads = torch.get_num_threads()
    deterministic = torch.are_deterministic_algorithms_enabled()
    warn_only = torch.is_deterministic_algorithms_warn_only_enabled()

    def fail(**kwargs):
        raise RuntimeError("injected model construction failure")

    monkeypatch.setattr(tabm.TabM, "make", fail)
    with pytest.raises(RuntimeError, match="injected"):
        fit_tabm(x, y, weights, config, seed=13)
    assert torch.equal(before, torch.random.get_rng_state())
    assert torch.get_num_threads() == threads
    assert torch.are_deterministic_algorithms_enabled() == deterministic
    assert torch.is_deterministic_algorithms_warn_only_enabled() == warn_only
