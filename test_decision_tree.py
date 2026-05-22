import numpy as np
import unittest
from decision_tree import LeafNode, RandomizedTree, InternalNode, RandomizedForest
from test_fixtures import X_deep, Y_deep, X_shallow, Y_shallow


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
        Y = np.array([1, 2, 1])
        tree = RandomizedTree()
        rng = np.random.default_rng(42)

        assert tree.build_tree(X, Y, 0, rng) == LeafNode(label=1)

    def test_build_tree_unanimous_labels(self):
        X = np.array([[1.0, 2.0, 3.0], [1.0, 1.0, 1.0], [1.0, 2.0, 4.0]])
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

    def test_build_tree_shallow(self):
        # One level of recursion
        tree = RandomizedTree(max_depth=10, min_samples=2)
        rng = np.random.default_rng(42)
        root = tree.build_tree(X_shallow, Y_shallow, 0, rng)

        assert root.height() == 2

    def test_build_tree_deep(self):
        # Deep fixture, we expect a height of 9 with a seed of 42.
        tree = RandomizedTree(max_depth=10, min_samples=2)
        rng = np.random.default_rng(42)
        root = tree.build_tree(X_deep, Y_deep, 0, rng)

        assert root.height() == 9

    def test_traverse_invalid_data_shape(self):
        tree = RandomizedTree(max_depth=10, min_samples=2)
        rng = np.random.default_rng(42)
        root = tree.build_tree(X_shallow, Y_shallow, 0, rng)

        invalid_data = np.array([[1, 2], [1, 1]])
        with self.assertRaises(AssertionError):
            tree.traverse(root, invalid_data)

    def test_traverse_shallow(self):
        tree = RandomizedTree()
        rng = np.random.default_rng(42)
        root = tree.build_tree(X_shallow, Y_shallow, 0, rng)

        data = np.array([5, 2])
        assert tree.traverse(root, data) == LeafNode(label=0)

    def test_traverse_deep(self):
        tree = RandomizedTree()
        rng = np.random.default_rng(42)
        root = tree.build_tree(X_deep, Y_deep, 0, rng)

        data = np.array([17, 12, 10, 5])
        assert tree.traverse(root, data) == LeafNode(label=0)

    def test_predict_unbuilt_tree(self):
        tree = RandomizedTree()

        data = np.array([5, 2])
        with self.assertRaises(Exception):
            tree.predict(data)

    def test_predict_shallow(self):
        tree = RandomizedTree()
        tree.create(X_shallow, Y_shallow, seed=42)

        data = np.array([5, 2])
        assert tree.predict(data) == 0

    def test_predict_deep(self):
        tree = RandomizedTree()
        tree.create(X_deep, Y_deep, seed=42)

        data = np.array([17, 12, 10, 5])
        assert tree.predict(data) == 0


class TestRandomizedForest(unittest.TestCase):
    def test_initialize_works(self):
        forest = RandomizedForest()
        forest.initialize(X_deep, Y_deep, num_trees=100)
        assert len(forest.trees) == 100
        assert all(t.is_built() for t in forest.trees)

    def test_predict_no_trees(self):
        forest = RandomizedForest()
        data = np.array([17, 12, 10, 5])
        assert forest.predict(data) is None

    def test_predict_works(self):
        forest = RandomizedForest()
        forest.initialize(X_deep, Y_deep, num_trees=100)

        data = np.array([17, 12, 10, 5])
        assert forest.predict(data) == 0
