import numpy as np
from dataclasses import dataclass
from typing import Union


@dataclass
class LeafNode:
    label: int


@dataclass
class InternalNode:
    feature: int
    threshold: float
    left: Union['InternalNode', LeafNode]
    right: Union['InternalNode', LeafNode]


class RandomizedTree:
    def __init__(self, max_depth: int = 10, min_samples_split: int = 2):
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.root = None

    def create(self, X: np.ndarray, Y: np.ndarray):
        self.root = self._build_tree(X, Y, depth=0)
    
    def _build_tree(self, X: np.ndarray, Y: np.ndarray, depth: int):
        pass

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