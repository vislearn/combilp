# Copyright (c) 2017 Stefan Haller
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.

import functools
import numpy

from .model import Model, Factor, DynamicFactor, DTYPE_INDEX

class BasicReparametrization:

    __slots__ = ('model', 'unaries', 'factors', 'data')

    def __init__(self, model):
        self.model = model

        self.unaries = [[] for _ in range(self.model.number_of_variables)]
        self.factors = numpy.zeros(self.model.number_of_factors, dtype=DTYPE_INDEX)
        size = 0
        for i, factor in enumerate(model.factors):
            if factor.number_of_variables > 1:
                for j, var in enumerate(factor.variables):
                    self.unaries[var].append((i, j))
                self.factors[i] = size
                size += sum(factor.shape)
        self.data = numpy.zeros(size)

    def get_factor(self, factor_index, rel_var_index=None):
        factor = self.model.factors[factor_index]
        assert(factor.number_of_variables > 1)

        start = self.factors[factor_index]
        end = start + sum(factor.shape)
        result = self.data[start:end]

        if rel_var_index is not None:
            start = sum(factor.shape[0:rel_var_index])
            end = start + factor.shape[rel_var_index]
            result = result[start:end]

        return result

    def get_factor_value(self, factor_index, labeling=None):
        factor = self.model.factors[factor_index]
        result = numpy.copy(factor.data)
        if factor.number_of_variables == 1:
            variable = factor.variables[0]
            for x in self.unaries[variable]:
                result += self.get_factor(*x)
        else:
            for rel_var_index, labels in enumerate(factor.shape):
                current = self.get_factor(factor_index, rel_var_index)

                def populate_shape(idx):
                    if idx == rel_var_index:
                        return labels
                    else:
                        return 1

                dest_shape = map(populate_shape, range(factor.number_of_variables))
                current = current.reshape(tuple(dest_shape))

                # Relies heavily on broadcasting.
                # To future me: If you look at this code here and wonder what
                # the heck it is doing, it took ~30min to come up with this
                # numpy-ish solution.
                result -= current 

        if labeling is not None:
            return result[labeling]

        return result

    def reparametrize_model(self, dynamic=True):
        rgm = Model(self.model.shape)
        for i, factor in enumerate(self.model.factors):
            if dynamic:
                f = DynamicFactor(factor.variables, factor.shape,
                        func=functools.partial(self.get_factor_value, i))
            else:
                f = Factor(factor.variables, data=self.get_factor_value(i))
            rgm.add_factor(f)
        return rgm

Reparametrization = BasicReparametrization
