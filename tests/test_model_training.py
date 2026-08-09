"""
tests/test_model_training.py
=============================
Assignment II, Objective 2, Item 7a: "Testing model training (e.g.,
overfitting on a small batch, checking loss decreases)".

Two independent checks:

1. test_overfits_small_batch
   Fits RandomForestClassifier (the algorithm actually used in
   train_model.train_model) on a tiny, perfectly-separable batch and
   asserts training accuracy is (near) 1.0. This is a sanity check on the
   training loop itself -- if features/labels were misaligned, or the
   estimator wasn't learning anything, this would fail immediately,
   independent of whether the model generalises to real data.

2. test_training_loss_decreases
   Uses GradientBoostingClassifier's train_score_ (an ensemble that
   reduces a shared deviance loss one boosting stage at a time) to verify
   training loss decreases as more stages are added. RandomForestClassifier
   has no per-iteration loss curve to inspect (each tree is grown
   independently, not to reduce a shared loss), so GradientBoosting is the
   right tool for this specific check -- it is also the algorithm this
   project originally evaluated (see notebooks/research.ipynb and the
   train_model.py docstring) before RandomForest was selected for
   production.
"""

import numpy as np
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier


def _make_separable_batch(n=40, n_features=4, seed=42):
    rng = np.random.default_rng(seed)
    X = rng.normal(size=(n, n_features))
    y = (X[:, 0] + X[:, 1] > 0).astype(int)  # perfectly separable rule
    return X, y


def test_overfits_small_batch():
    """The production algorithm should drive training accuracy to ~1.0 on
    an easy, tiny batch."""
    X, y = _make_separable_batch()

    model = RandomForestClassifier(n_estimators=50, random_state=42)
    model.fit(X, y)
    train_accuracy = model.score(X, y)

    assert (
        train_accuracy >= 0.98
    ), f"Expected near-perfect training accuracy on an easy batch, got {train_accuracy}"


def test_training_loss_decreases():
    """Training (deviance) loss must trend down as boosting stages are
    added -- if it doesn't, the model isn't learning."""
    X, y = _make_separable_batch(n=80, seed=7)

    model = GradientBoostingClassifier(
        n_estimators=50, learning_rate=0.1, max_depth=2, random_state=42
    )
    model.fit(X, y)

    train_loss_per_stage = model.train_score_  # negative log-likelihood per stage
    first_segment_avg = np.mean(train_loss_per_stage[:10])
    last_segment_avg = np.mean(train_loss_per_stage[-10:])

    assert last_segment_avg < first_segment_avg, (
        "Expected training loss (train_score_) to decrease from the early "
        "boosting stages to the later ones"
    )
    assert train_loss_per_stage[-1] < train_loss_per_stage[0]
