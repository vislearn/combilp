import combilp

import unittest

class SmallTests(unittest.TestCase):

    def test_walker(self):
        shape = [2, 3]
        correct = [(0, 0), (0, 1), (0, 2), (1, 0), (1, 1), (1, 2)]
        actual = list(combilp.walk_shape(shape))
        self.assertSequenceEqual(actual, correct)
