"""Leak-free time-series cross-validation: purged walk-forward and purged K-fold.

Splits operate on integer bar positions, so they work on the single-asset
EURUSD frame today and on a (timesteps, assets) panel later. Use
`purged_walk_forward` for honest backtesting (train strictly precedes test);
`purged_kfold` (Lopez de Prado, "Advances in Financial ML" ch. 7) squeezes
more evaluation out of small datasets at the cost of training on data that
postdates the test block.

Terminology:
- purge: drop the last `purge` bars of any training data that immediately
  precedes a test block, so a label computed over a forward horizon h
  (the label at bar t looks at t..t+h) can never straddle the boundary.
  Set purge >= your label horizon.
- embargo: drop the first `embargo` bars of training data that immediately
  follows a test block, so serial correlation cannot leak the test period
  back into training. Only meaningful in K-fold (in walk-forward no
  training data follows the test block).
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class Fold:
    fold: int
    train: np.ndarray  # integer bar positions
    test: np.ndarray


def purged_walk_forward(
    n_samples: int,
    n_folds: int = 4,
    *,
    min_train: int | None = None,
    purge: int = 0,
    expanding: bool = True,
) -> list[Fold]:
    """Chronological folds: train on the past, test on the next block.

    The bars in [test_start - purge, test_start) are excluded from training.
    `expanding=True` grows the training window each fold; False keeps a
    rolling window of `min_train` bars.
    """
    if min_train is None:
        min_train = n_samples // (n_folds + 1)
    test_size = (n_samples - min_train) // n_folds
    if min_train <= purge or test_size < 1:
        raise ValueError(
            f"n_samples={n_samples} too small for n_folds={n_folds}, "
            f"min_train={min_train}, purge={purge}"
        )
    folds = []
    for k in range(n_folds):
        test_start = min_train + k * test_size
        test_end = test_start + test_size if k < n_folds - 1 else n_samples
        train_start = 0 if expanding else test_start - min_train
        train = np.arange(train_start, test_start - purge)
        test = np.arange(test_start, test_end)
        folds.append(Fold(k, train, test))
    return folds


def purged_kfold(
    n_samples: int,
    n_folds: int = 5,
    *,
    purge: int = 0,
    embargo: int = 0,
) -> list[Fold]:
    """Purged K-fold: each fold's test block is a contiguous slice; training
    is every other bar minus `purge` bars before and `embargo` bars after
    the test block.
    """
    edges = np.linspace(0, n_samples, n_folds + 1, dtype=int)
    folds = []
    for k in range(n_folds):
        t0, t1 = int(edges[k]), int(edges[k + 1])
        train_mask = np.ones(n_samples, dtype=bool)
        train_mask[max(0, t0 - purge) : min(n_samples, t1 + embargo)] = False
        folds.append(Fold(k, np.nonzero(train_mask)[0], np.arange(t0, t1)))
    return folds


def describe(folds: list[Fold], index=None) -> str:
    """Human-readable fold summary; pass a pandas DatetimeIndex to show dates."""
    lines = []
    for f in folds:
        def span(pos):
            if len(pos) == 0:
                return "(empty)"
            if index is not None:
                return f"{index[pos[0]]} .. {index[pos[-1]]} ({len(pos)} bars)"
            return f"{pos[0]} .. {pos[-1]} ({len(pos)} bars)"

        lines.append(f"fold {f.fold}: train {span(f.train)} | test {span(f.test)}")
    return "\n".join(lines)
