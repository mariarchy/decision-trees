import numpy as np


def gini_impurity(Y: np.ndarray):
    """
    Y must be a 1D array
    """
    assert len(Y.shape) == 1

    if len(Y) == 0:
        return 0.0
    _, counts = np.unique(Y, return_counts=True)
    p = counts / counts.sum()
    return 1.0 - np.sum(p**2)


def get_split_score(Y: np.ndarray, Y_left: np.ndarray, Y_right: np.ndarray):
    """
    Score splits by the gini impurity reduction (gini impurity of the parent minus
    the weighted gini impurity of the subtrees).

    All arrays must be a 1D array.
    """
    assert len(Y.shape) == 1
    assert len(Y.shape) == len(Y_left.shape) and len(Y.shape) == len(Y_right.shape)

    n = len(Y)
    n_left = len(Y_left)
    n_right = len(Y_right)

    if n_left == 0 or n_right == 0:
        return 0.0

    parent_imp = gini_impurity(Y)
    left_imp = gini_impurity(Y_left)
    right_imp = gini_impurity(Y_right)

    left_weight = n_left / n
    right_weight = n_right / n
    weighted_split_imp = left_weight * left_imp + right_weight * right_imp

    return parent_imp - weighted_split_imp


def get_nonconstant_features(X: np.ndarray) -> np.ndarray[int]:
    assert len(X.shape) == 2
    # Get all features (i) where i is not constant between samples by
    # comparing each comlumn to its first element and returning the
    # columns where values across all rows are true
    mask = np.any(X != X[0, :], axis=0)
    n_features = X.shape[1]
    return np.arange(n_features)[mask]
