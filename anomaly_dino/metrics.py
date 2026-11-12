"""Exact floating-point AUROC with in-place negative sorting and bounded overhead."""
import numpy as np


def positive_negative_auc(positive, negative):
    # Sorting values in place avoids sklearn's large full-image int64 ranking arrays.
    if not len(positive) or not len(negative):
        raise ValueError('Both positive and negative observations are required')
    positive.sort()
    negative.sort()
    wins = np.uint64(0)
    ties = np.uint64(0)
    for start in range(0, len(positive), 250000):
        part = positive[start:start + 250000]
        lower = np.searchsorted(negative, part, side='left')
        upper = np.searchsorted(negative, part, side='right')
        wins += lower.sum(dtype=np.uint64)
        ties += (upper - lower).sum(dtype=np.uint64)
    return float((float(wins) + .5 * float(ties)) / (len(positive) * len(negative)))
