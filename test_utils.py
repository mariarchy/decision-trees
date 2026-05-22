import numpy as np
import unittest
from utils import gini_impurity, get_split_score, get_nonconstant_features, majority


class TestGiniImpurity(unittest.TestCase):
    def test_gini_impurity_invalid_shape_errors(self):
        arr = np.array([[1, 2], [3, 4]])
        with self.assertRaises(AssertionError):
            gini_impurity(arr)

    def test_gini_impurity_empty_array(self):
        arr = np.array([])
        assert gini_impurity(arr) == 0

    def test_gini_impurity_works(self):
        arr = np.array([1.0, 1.0, 2.0, 2.0, 3.0])
        assert np.isclose(gini_impurity(arr), 0.64)


class TestSplitScore(unittest.TestCase):
    def test_get_split_score_invalid_shape_parent(self):
        parent = np.array([[1, 2], [3, 4]])
        left = np.array([4])
        right = np.array([1, 2, 3])
        with self.assertRaises(AssertionError):
            get_split_score(parent, left, right)

    def test_get_split_score_invalid_shape_children_left(self):
        parent = np.array([1, 2, 3, 4])
        left = np.array([[4]])
        right = np.array([1, 2, 3])
        with self.assertRaises(AssertionError):
            get_split_score(parent, left, right)

    def test_get_split_score_invalid_shape_children_right(self):
        parent = np.array([1, 2, 3, 4])
        left = np.array([4])
        right = np.array([[1, 2, 3]])
        with self.assertRaises(AssertionError):
            get_split_score(parent, left, right)

    def test_get_split_score_empty_left(self):
        parent = np.array([1, 2, 3, 4])
        left = np.array([1, 2, 3, 4])
        right = np.array([])
        assert get_split_score(parent, left, right) == 0

    def test_get_split_score_empty_right(self):
        parent = np.array([1, 2, 3, 4])
        left = np.array([1, 2, 3, 4])
        right = np.array([])
        assert get_split_score(parent, left, right) == 0

    def test_get_split_score_works(self):
        parent = np.array([1, 1, 2, 2, 3])  # 0.64
        left = np.array([2, 2, 3])  # 0.4444444
        right = np.array([1, 1])  # 0
        assert np.isclose(get_split_score(parent, left, right), 0.3733333333)


class TestGetNonConstantFeatures(unittest.TestCase):
    def test_get_nonconstant_features_invalid_shape_errors(self):
        X = np.array([1, 2, 3, 4])
        with self.assertRaises(AssertionError):
            get_nonconstant_features(X)

    def test_get_nonconstant_features_all_constant(self):
        X = np.array([[1, 2], [1, 2]])
        assert len(get_nonconstant_features(X)) == 0

    def test_get_nonconstant_features_some_constant(self):
        X = np.array([[1, 2, 3], [1, 2, 4]])
        expected = np.array([2])
        assert np.array_equal(get_nonconstant_features(X), expected)


class TestMajority(unittest.TestCase):
    def test_majority_invalid_shape(self):
        Y = np.array([[1, 1, 1]])
        with self.assertRaises(AssertionError):
            majority(Y)

    def test_majority_works(self):
        Y = np.array([1, 1, 1, 2, 2, 3, 3])
        assert majority(Y) == 1
