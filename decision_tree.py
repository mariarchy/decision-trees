import numpy as np
from dataclasses import dataclass
from typing import Union, NamedTuple
from utils import get_nonconstant_features, get_split_score, majority


@dataclass
class LeafNode:
    label: int

    def height(self):
        return 1


@dataclass
class InternalNode:
    feature: int
    threshold: float
    left: Union["InternalNode", LeafNode]
    right: Union["InternalNode", LeafNode]

    def height(self) -> int:
        """
        Compute max height of the subtree at this node
        """
        left = self.left.height()
        right = self.right.height()

        return max(left, right) + 1


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

    def create(self, X: np.ndarray, Y: np.ndarray, seed: int = 42):
        rng = np.random.default_rng(seed)
        self.root = self.build_tree(X, Y, depth=0, rng=rng)

    def build_tree(
        self, X: np.ndarray, Y: np.ndarray, depth: int, rng: np.random.Generator
    ):
        """
        Returns root node of subtree

        X: sample data of shape (n_examples, n_features)
        Y: labels of shape (n_examples, )
        """
        assert len(X) > 0
        assert len(X) == len(Y)
        assert len(X.shape) == 2
        assert len(Y.shape) == 1

        # Termination case
        if (
            np.all(X == X[0])
            or np.all(Y == Y[0])
            or depth == self.max_depth
            or len(X) <= self.min_samples
        ):
            return LeafNode(label=majority(Y))

        nonconst_feats = get_nonconstant_features(X)
        n_sample_feats = int(np.sqrt(len(nonconst_feats)))

        # If all features are constant, create a leaf node with the majority.
        if len(nonconst_feats) == 0:
            return LeafNode(label=majority(Y))

        features = rng.choice(nonconst_feats, size=n_sample_feats, replace=False)

        splits: list[Split] = []
        for feat in features:
            f_vals = X[:, feat]
            f_min, f_max = np.min(f_vals), np.max(f_vals)

            assert f_min != f_max  # We should not be handling any constant features

            u = rng.random()  # [0, 1)

            # (f_min, f_max], f_min should be exclusive to avoid bad splits
            threshold = f_min + (1 - u) * (f_max - f_min)
            row_mask = f_vals < threshold
            left, right = Y[row_mask], Y[~row_mask]
            score = get_split_score(Y, left, right)

            # We should not be choosing split values that yield empty subtrees
            assert len(left) > 0 and len(right) > 0

            splits.append(
                Split(feature=feat, threshold=threshold, score=score, row_mask=row_mask)
            )

        assert len(splits) > 0
        s = max(splits, key=lambda s: s.score)
        left = self.build_tree(X[s.row_mask], Y[s.row_mask], depth + 1, rng)
        right = self.build_tree(X[~s.row_mask], Y[~s.row_mask], depth + 1, rng)
        return InternalNode(
            feature=s.feature, threshold=s.threshold, left=left, right=right
        )

    def traverse(
        self, node: Union[InternalNode, LeafNode], data: np.ndarray
    ) -> LeafNode:
        assert len(data.shape) == 1
        if isinstance(node, LeafNode):
            return node

        f, t = node.feature, node.threshold
        assert f < len(data)

        next_node = node.left if data[f] < t else node.right
        return self.traverse(next_node, data)

    def predict(self, data: np.ndarray) -> int:
        """
        Predicts and returns the label for the data by traversing the tree

        Raises exception if the tree has not been built.
        """
        if self.root is None:
            raise Exception("Tree has not been built")

        leaf = self.traverse(self.root, data)
        return leaf.label

    def height(self) -> int:
        return self.root.height() if self.root is not None else -1

    def is_built(self) -> bool:
        return self.root is not None


class RandomizedForest:
    def __init__(self, trees: list[RandomizedTree] | None = None):
        self.trees = trees if trees is not None else []

    def initialize(
        self, X: np.ndarray, Y: np.ndarray, num_trees: int = 100
    ) -> list[RandomizedTree]:
        """
        Factory function for initializing a forest of num_trees trees
        The index of the tree is used as the rng seed value to produce
        a list of randomized trees
        """
        trees = []
        for i in range(num_trees):
            t = RandomizedTree()
            t.create(X, Y, seed=i)
            trees.append(t)
        self.trees = trees
        return trees

    def predict(self, data: np.ndarray) -> int | None:
        """
        Returns the majority prediction across the forest of trees
        """
        if len(self.trees) == 0:
            return None
        preditions = np.array([t.predict(data) for t in self.trees])
        return majority(preditions)
