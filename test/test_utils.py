import combilp
import numpy
import unittest

from combilp import Model, Factor, Reparametrization
from combilp.utils import find_unary_factor_indices, count_neighbors, unary_reparametrization, compute_strict_arc_consistency, reparametrize_border_factor, reparametrize_border

def create_random_model():
    num_vars = 10
    num_labs = 5
    model = Model([num_labs] * num_vars)
    for var in range(num_vars):
        data = numpy.random.random(num_labs).reshape(num_labs)
        model.add_factor(Factor([var], data=data))

    for var0 in range(num_vars):
        for var1 in range(var0 + 1, num_vars):
            data = numpy.random.random(num_labs * num_labs).reshape(num_labs, num_labs)
            model.add_factor(Factor([var0, var1], data=data))

    return model

class HelperTest(unittest.TestCase):

    def setUp(self):
        self.model = Model([2, 2, 2])
        self.model.add_factor(Factor([0], shape=[2]))
        self.model.add_factor(Factor([1], shape=[2]))
        self.model.add_factor(Factor([0, 1], shape=[2, 2]))
        self.model.add_factor(Factor([1, 2], shape=[2, 2]))
        self.model.add_factor(Factor([2], shape=[2]))

    def test_working_model(self):
        result = find_unary_factor_indices(self.model)
        self.assertEqual(list(result), [0, 1, 4])

    def test_no_unary(self):
        del self.model.factors[1]
        self.assertRaises(RuntimeError, find_unary_factor_indices, self.model)

    def test_multiple_unaries(self):
        self.model.add_factor(Factor([1], shape=[2]))
        self.assertRaises(RuntimeError, find_unary_factor_indices, self.model)

    def test_neighbor_count(self):
        result = count_neighbors(self.model)
        self.assertEqual(list(result), [1, 2, 1])

        self.model.add_factor(Factor([0, 1, 2], shape=[2, 2, 2]))
        result = count_neighbors(self.model)
        self.assertEqual(list(result), [2, 3, 2])

class UnaryRepaTest(unittest.TestCase):

    def test_repa(self):
        for i in range(100):
            model = create_random_model()
            repa = Reparametrization(model)
            unary_reparametrization(repa)
            for factor_index, factor in enumerate(model.factors):
                if factor.number_of_variables == 1:
                    data = repa.get_factor_value(factor_index)
                    self.assertAlmostEqual(data.min(), 0)
                    self.assertAlmostEqual(data.max(), 0)

class StrictArcConsistencyTest(unittest.TestCase):

    def setUp(self):
        self.model = Model([2, 2, 2, 2])
        data_unary = numpy.asarray([0.0, 1.0])
        data_unary2 = numpy.asarray([10.0, 0.0])
        data_pairwise = numpy.asarray([0.0, 1.0, 1.0, 0.0]).reshape(2, 2)

        for var in range(4):
            self.model.add_factor(Factor([var], data=data_unary))

        for var0, var1 in ((0, 1), (0, 2), (1, 3), (2, 3)):
            self.model.add_factor(Factor([var0, var1], data=data_pairwise))

        self.model.factors[3].data = data_unary2

    def test_model(self):
        self.assertAlmostEqual(self.model.evaluate([0, 0, 0, 0]), 10)
        self.assertAlmostEqual(self.model.evaluate([0, 0, 0, 1]), 2)

    def test_wrong_model(self):
        self.assertRaises(AssertionError, compute_strict_arc_consistency, self.model)

    def test_correct_model(self):
        repa = Reparametrization(self.model)
        unary_reparametrization(repa)
        mask, sac_labeling = compute_strict_arc_consistency(repa.reparametrize_model())
        self.assertEqual(list(mask), [True, False, False, True])
        self.assertEqual(list(sac_labeling), [0, 0, 0, 1])

    def test_inversed_model(self):
        for factor in self.model.factors:
            if factor.number_of_variables == 1:
                factor.data = factor.data[::-1]

        repa = Reparametrization(self.model)
        unary_reparametrization(repa)
        mask, sac_labeling = compute_strict_arc_consistency(repa.reparametrize_model())
        self.assertEqual(list(mask), [True, False, False, True])
        self.assertEqual(list(sac_labeling), [1, 1, 1, 0])

class BorderReparametrizationTest(unittest.TestCase):

    def test_single_factor(self):
        for i in range(100):
            model = create_random_model()
            repa = Reparametrization(model)

            factor_index, factor = next((i, f) for i, f in enumerate(model.factors) if f.number_of_variables == 2)
            reparametrize_border_factor(repa, factor_index, [1])

            data = repa.get_factor_value(factor_index)
            index = [slice(None) for i in range(factor.number_of_variables)]
            for right_label in range(factor.shape[1]):
                index[1] = right_label
                self.assertAlmostEqual(data[index].min(), 0)
