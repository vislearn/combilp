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

import numpy

DTYPE_INDEX = numpy.int32
DTYPE_LABEL = numpy.int32
DTYPE_VALUE = numpy.float64

def make_labeling(size):
    return numpy.zeros(size, dtype=DTYPE_LABEL)

def walk_shape(shape):
    x = make_labeling(len(shape))
    while True:
        yield tuple(x)

        i = len(shape) - 1
        x[i] += 1
        while True:
            if x[i] >= shape[i]:
                x[i] = 0
                i -= 1
                if i >= 0:
                    x[i] += 1
                else:
                    return
            else:
                break

def walk_sub_shape(shape, fixed):
    fixed_indices = [ v for v, l in fixed ]
    my_shape = [ x for i, x in enumerate(shape) if i not in fixed_indices ]
    for coords in walk_shape(my_shape):
        coords = list(coords)
        for var, lab in fixed:
            coords[var:var] = [lab]
        yield tuple(coords)

class Factor:

    __slots__ = ('variables', 'data')

    def __init__(self, variables, shape=None, data=None):
        self.variables = numpy.asarray(variables, dtype=DTYPE_INDEX)
        if shape is not None and data is None:
            assert(len(variables) == len(shape))
            self.data = numpy.zeros(shape, dtype=DTYPE_VALUE)
        elif shape is None and data is not None:
            assert(len(variables) == len(data.shape))
            self.data = data
        else:
            raise ValueError('Received both shape and data argument.')

    def evaluate(self, labeling):
        assert(len(labeling) == len(self.shape))
        value = self.data[tuple(labeling)]
        return value

    def local_labeling(self, labeling):
        local_labeling = make_labeling(len(self.variables))
        for i, var in enumerate(self.variables):
            local_labeling[i] = labeling[self.variables[i]]
        return local_labeling

    def local_evaluate(self, labeling):
        return self.evaluate(self.local_labeling(labeling))

    @property
    def number_of_variables(self):
        return len(self.variables)

    @property
    def shape(self):
        return self.data.shape

class DynamicFactor(Factor):

    __slots__ = ('shape', 'func')

    def __init__(self, variables, shape, func):
        self.variables = variables
        self.shape = shape
        self.func = func

    @property
    def data(self):
        return self.func()

class Model:

    __slots__ = ('shape', 'factors', 'constant')

    def __init__(self, shape):
        self.shape = numpy.asarray(shape, dtype=DTYPE_LABEL)
        self.factors = []
        self.constant = 0.0

    def add_constant(self, constant):
        self.constant += constant

    def add_factor(self, factor):
        assert(factor.number_of_variables > 0)
        self.factors.append(factor)

    def evaluate(self, labeling):
        assert(len(labeling) == len(self.shape))
        return sum(f.local_evaluate(labeling) for f in self.factors) + self.constant

    @property
    def bound(self):
        return sum(f.data.min() for f in self.factors) + self.constant

    @property
    def number_of_variables(self):
        return len(self.shape)

    def number_of_labels(self, variable):
        return self.shape[variable]

    @property
    def number_of_factors(self):
        return len(self.factors)

    def factors_of_variable(self, variable):
        for factor in self.factors:
            if variable in factor.variables:
                yield factor
