import numpy as np
import unittest
from decision_tree import LeafNode, RandomizedTree, InternalNode


class TestInternalNode(unittest.TestCase):
    def test_height_shallow(self):
        node = InternalNode(
            feature=0, threshold=100, left=LeafNode(label=0), right=LeafNode(label=1)
        )
        assert node.height() == 2

    def test_height_deep(self):
        left = InternalNode(
            feature=0, threshold=100, left=LeafNode(label=0), right=LeafNode(label=1)
        )
        right = InternalNode(
            feature=1, threshold=10, left=LeafNode(label=0), right=LeafNode(label=2)
        )
        right2 = InternalNode(
            feature=2, threshold=20, left=LeafNode(label=0), right=right
        )
        root = InternalNode(feature=0, threshold=100, left=left, right=right2)
        assert root.height() == 4


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
