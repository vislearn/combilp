import numpy
import unittest

import combilp
from combilp import Model, Factor, Reparametrization

def generate_random_chain_model():
    model = Model([2, 3, 1, 3])

    for var, labels in enumerate(model.shape):
        data = numpy.random.random(labels)
        model.add_factor(Factor([var], data=data))

    for var2 in range(1, model.number_of_variables):
        var1 = var2 - 1
        data = numpy.random.random([model.shape[x] for x in [var1, var2]])
        model.add_factor(Factor([var1, var2], data=data))

    return model

class ReparametrizationTester(unittest.TestCase):

    def setUp(self):
        self.model = generate_random_chain_model()

    def test_identity(self):
        repa = Reparametrization(self.model)

        for factor_index, factor in enumerate(self.model.factors):
            for labeling in combilp.walk_shape(factor.shape):
                self.assertAlmostEqual(factor.data[labeling],
                    repa.get_factor_value(factor_index, labeling))

    def test_identity_model(self):
        repa = Reparametrization(self.model)
        rmodel = repa.reparametrize_model()

        for labeling in combilp.walk_shape(self.model.shape):
            self.assertAlmostEqual(self.model.evaluate(labeling),
                rmodel.evaluate(labeling))

    def test_primal_equality(self):
        for _ in range(500):
            repa = Reparametrization(self.model)
            repa.data = numpy.random.random(repa.data.shape)
            repa.data = (repa.data - 0.5) * 100
            rmodel = repa.reparametrize_model()

            for labeling in combilp.walk_shape(self.model.shape):
                self.assertAlmostEqual(self.model.evaluate(labeling),
                    rmodel.evaluate(labeling))

                self.assertNotAlmostEqual(self.model.bound, rmodel.bound)

    def test_small_example(self):
        model = Model([2, 2])

        f = Factor([0], [2])
        f.data[0] = 10
        f.data[1] = 20
        model.add_factor(f)

        f = Factor([1], [2])
        f.data[0] = 30
        f.data[1] = 40
        model.add_factor(f)

        f = Factor([0, 1], [2, 2])
        f.data[0, 0] = 1
        f.data[0, 1] = 2
        f.data[1, 0] = 3
        f.data[1, 1] = 4
        model.add_factor(f)

        repa = Reparametrization(model)
        repa.data[:] = [9, 50, 6, -5]

        numpy.testing.assert_almost_equal(repa.get_factor(2), [9, 50, 6, -5])
        numpy.testing.assert_almost_equal(repa.get_factor(2, 0), [9, 50])
        numpy.testing.assert_almost_equal(repa.get_factor(2, 1), [6, -5])

        numpy.testing.assert_almost_equal(repa.get_factor_value(0), [19, 70])
        numpy.testing.assert_almost_equal(repa.get_factor_value(1), [36, 35])
        numpy.testing.assert_almost_equal(repa.get_factor_value(2), numpy.array([-14, -2, -53, -41]).reshape(2,2))

    def test_dynamic(self):
        repa = Reparametrization(self.model)
        rmodel = repa.reparametrize_model(dynamic=True)
        self.assertAlmostEqual(rmodel.bound, self.model.bound)

        for _ in range(500):
            for factor_index, factor in enumerate(self.model.factors):
                if factor.number_of_variables >= 2:
                    data = repa.get_factor(factor_index)
                    data[:] = (numpy.random.random(data.shape) - 0.5) * 100

            self.assertNotAlmostEqual(self.model.bound, rmodel.bound)
