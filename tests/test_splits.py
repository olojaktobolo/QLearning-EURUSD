"""Invariant checks for qlearning.splits. Run with: poetry run python -m tests.test_splits"""

import numpy as np

from qlearning.splits import purged_kfold, purged_walk_forward


def test_walk_forward_no_leakage():
    n, purge = 10_000, 24
    folds = purged_walk_forward(n, n_folds=4, purge=purge)
    assert len(folds) == 4
    for f in folds:
        # Train strictly precedes test, with at least `purge` bars of gap.
        assert f.train.max() < f.test.min() - purge + 1
        assert f.test.min() - f.train.max() - 1 >= purge
        # Positions are strictly increasing and in range.
        assert f.train[0] >= 0 and f.test[-1] <= n - 1
    # Test blocks are disjoint, ordered, and jointly cover [min_train, n).
    all_test = np.concatenate([f.test for f in folds])
    assert len(all_test) == len(set(all_test.tolist()))
    assert (np.diff(all_test) > 0).all()
    assert all_test[-1] == n - 1


def test_walk_forward_rolling_window():
    folds = purged_walk_forward(1_000, n_folds=4, min_train=300, purge=10, expanding=False)
    for f in folds:
        assert len(f.train) == 300 - 10  # fixed window minus purge


def test_kfold_purge_and_embargo():
    n, purge, embargo = 5_000, 24, 24
    folds = purged_kfold(n, n_folds=5, purge=purge, embargo=embargo)
    all_test = np.concatenate([f.test for f in folds])
    assert len(all_test) == n  # test blocks partition the sample
    for f in folds:
        t0, t1 = f.test[0], f.test[-1]
        forbidden = set(range(max(0, t0 - purge), min(n, t1 + 1 + embargo)))
        assert forbidden.isdisjoint(f.train.tolist())


def test_too_small_raises():
    try:
        purged_walk_forward(50, n_folds=4, min_train=10, purge=10)
    except ValueError:
        pass
    else:
        raise AssertionError("expected ValueError for purge >= min_train")


if __name__ == "__main__":
    for fn in [v for k, v in list(globals().items()) if k.startswith("test_")]:
        fn()
        print(f"ok  {fn.__name__}")
    print("all split invariants hold")
