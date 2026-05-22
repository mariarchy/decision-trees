import numpy as np
from dataclasses import dataclass
from typing import Union, NamedTuple
from utils import gini_impurity, get_split_score


@dataclass
class LeafNode:
    label: int


@dataclass
class InternalNode:
    feature: int
    threshold: float
    left: Union["InternalNode", LeafNode]
    right: Union["InternalNode", LeafNode]


class Split(NamedTuple):
    feature: int
    threshold: float
    score: float
    row_mask: np.ndarray


class RandomizedTree:
    def __init__(self, max_depth: int = 10, min_samples: int = 2):
        self.max_depth = max_depth
        self.min_samples = min_samples
        self.root = None

    def create(self, X: np.ndarray, Y: np.ndarray):
        self.root = self._build_tree(X, Y, depth=0)

    def _majority(self, Y: np.ndarray) -> int:
        """
        Returns the majority value in 1D array Y
        """
        assert len(Y.shape) == 1

        return np.argmax(np.bincount(Y))

    def _build_tree(
        self, X: np.ndarray, Y: np.ndarray, depth: int, rng: np.random.Generator
    ):
        """
        Returns root node of subtree

        X: sample data of shape (n_examples, n_features)
        Y: labels of shape (n_examples, )
        """
        assert len(X) > 0
        assert len(X) == len(Y)
        assert len(Y.shape) == 1

        # Termination case
        if np.all(X == X[0]) or depth == self.max_depth or len(X) == self.min_samples:
            return LeafNode(label=self._majority(Y))

        _, n_features = X.shape
        features = rng.choice(np.range(n_features), np.sqrt(n_features), replace=False)
        splits: list[Split] = []

        for feat in features:
            f_vals = X[:, feat]
            f_min, f_max = np.min(f_vals), np.max(f_vals)
            threshold = rng.choice(np.range(f_min, f_max))
            row_mask = f_vals < threshold
            score = get_split_score(Y, Y[row_mask], Y[~row_mask])
            splits.append(
                Split(feature=feat, threshold=threshold, score=score, row_mask=row_mask)
            )

        s = max(splits, key=lambda s: s.score)
        left = self._build_tree(X[s.row_mask], Y[s.row_mask], depth + 1, rng)
        right = self._build_tree(X[~s.row_mask], Y[~s.row_mask], depth + 1, rng)
        return InternalNode(
            feature=s.feature, threshold=s.threshold, left=left, right=right
        )

    def predict(self, data: np.ndarray) -> int:
        """
        Predicts and returns the label for the data by traversing the tree

        Raises MissingTreeException if the tree has not been built.
        """
        pass


class RandomizedForest:
    def __init__(self, trees: list[RandomizedTree]):
        self.trees = trees

    def initialize(num_trees: int) -> list[RandomizedTree]:
        """
        Factory function for initializing a forest of num_trees trees
        """
        pass

    def predict(self, data: np.ndarray) -> int:
        """
        Returns the majority prediction across the forest of trees
        """
        pass
