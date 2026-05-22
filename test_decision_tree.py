import numpy as np
import unittest
from decision_tree import RandomizedTree


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
