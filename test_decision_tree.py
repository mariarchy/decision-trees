import numpy as np
import unittest
from decision_tree import LeafNode, RandomizedTree


class TestRandomizedTree(unittest.TestCase):
    def test_majority_invalid_shape(self):
        Y = np.array([[1, 1, 1]])
        tree = RandomizedTree()
        with self.assertRaises(AssertionError):
            tree._majority(Y)

    def test_majority_works(self):
        Y = np.array([1, 1, 1, 2, 2, 3, 3])
        tree = RandomizedTree()
        assert tree._majority(Y) == 1

    def test_build_tree_empty_sample(self):
        X = np.array([])
        Y = np.array([[1, 1, 1]])
        tree = RandomizedTree()
        rng = np.random.default_rng(42)
        with self.assertRaises(AssertionError):
            tree.build_tree(X, Y, 0, rng)

    def test_build_tree_invalid_shape_samples(self):
        X = np.array([1, 2, 3])
        Y = np.array([1, 2, 3])
        tree = RandomizedTree()
        rng = np.random.default_rng(42)
        with self.assertRaises(AssertionError):
            tree.build_tree(X, Y, 0, rng)

    def test_build_tree_mismatch_samples(self):
        X = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]])
        Y = np.array([1, 2])
        tree = RandomizedTree()
        rng = np.random.default_rng(42)
        with self.assertRaises(AssertionError):
            tree.build_tree(X, Y, 0, rng)

    def test_build_tree_invalid_shape_labels(self):
        X = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]])
        Y = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]])
        tree = RandomizedTree()
        rng = np.random.default_rng(42)
        with self.assertRaises(AssertionError):
            tree.build_tree(X, Y, 0, rng)

    def test_build_tree_identical_sample_data(self):
        X = np.array([[1.0, 1.0, 1.0], [1.0, 1.0, 1.0], [1.0, 1.0, 1.0]])
        Y = np.array([1, 1, 1])
        tree = RandomizedTree()
        rng = np.random.default_rng(42)

        assert tree.build_tree(X, Y, 0, rng) == LeafNode(label=1)

    def test_build_tree_max_depth_reached(self):
        X = np.array([[1.0, 2.0, 3.0], [1.0, 2.0, 3.0], [1.0, 2.0, 3.0]])
        Y = np.array([1, 2, 1])
        tree = RandomizedTree(max_depth=10)
        rng = np.random.default_rng(42)

        assert tree.build_tree(X, Y, depth=10, rng=rng) == LeafNode(label=1)

    def test_build_tree_min_sample_reached(self):
        X = np.array([[1.0, 2.0, 3.0], [1.0, 2.0, 3.0], [1.0, 2.0, 3.0]])
        Y = np.array([1, 2, 1])
        tree = RandomizedTree(max_depth=10, min_samples=3)
        rng = np.random.default_rng(42)

        assert tree.build_tree(X, Y, depth=5, rng=rng) == LeafNode(label=1)

    def test_build_tree_all_features_constant(self):
        X = np.array([[1.0, 2.0, 3.0], [1.0, 2.0, 3.0], [1.0, 2.0, 3.0]])
        Y = np.array([1, 2, 1])
        tree = RandomizedTree()
        rng = np.random.default_rng(42)

        assert tree.build_tree(X, Y, depth=5, rng=rng) == LeafNode(label=1)
