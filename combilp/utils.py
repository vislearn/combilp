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
import time

from .model import DTYPE_INDEX, make_labeling, walk_shape

class PerformanceMeasurement:
    def __init__ (self):
        self.last = 0.0
        self.total = 0.0

    def __enter__(self):
        self.start = time.perf_counter()
        self.end = None

    def __exit__(self, *args):
        self.end = time.perf_counter()
        self.last = self.end - self.start
        self.total += self.last

class PerformanceStatistics:
    def __getattr__(self, name):
        newly_constructed = PerformanceMeasurement()
        setattr(self, name, newly_constructed)
        return newly_constructed

    def items(self):
        return self.__dict__.items()

def find_unary_factor_indices(model):
    flags = numpy.zeros(model.number_of_variables, dtype=bool)
    result = numpy.zeros(model.number_of_variables, dtype=DTYPE_INDEX)
    for i, factor in enumerate(model.factors):
        if factor.number_of_variables == 1:
            var, = factor.variables
            if flags[var]:
                raise RuntimeError('Multiple unary factors detected.')
            result[var] = i
            flags[var] = True
    if not flags.all():
        raise RuntimeError('Not all variables have a unary factor.')
    return result

def count_neighbors(model):
    result = numpy.zeros(model.number_of_variables, dtype=DTYPE_INDEX)
    for factor in model.factors:
        if factor.number_of_variables >= 2:
            for variable in factor.variables:
                result[variable] += 1
    return result

def unary_reparametrization(repa):
    unaries = find_unary_factor_indices(repa.model)
    neighbor_count = count_neighbors(repa.model)
    for factor_idx, factor in enumerate(repa.model.factors):
        if factor.number_of_variables >= 2:
            for local_variable_index, variable in enumerate(factor.variables):
                r = repa.get_factor(factor_idx, local_variable_index)
                r -= repa.get_factor_value(unaries[variable]) / neighbor_count[variable]
                neighbor_count[variable] -= 1
    assert(neighbor_count.sum() == 0)

# Important, assumes that unary_reparametrization has been run!
def compute_strict_arc_consistency(model):
    assert(sum(abs(factor.data.max() - factor.data.min()) for factor in model.factors if factor.number_of_variables == 1) <= 1e-8)

    count_similar = 0
    for factor in model.factors:
        if factor.number_of_variables >= 2:
            count_similar += ((factor.data - factor.data.min()) < 1e-8).sum() - 1
    print('SAC computation detected {} highly similar values.'.format(count_similar))


    class Consistency:
        __slots__ = ('is_first', 'is_sac', 'label')

        def __init__(self):
            self.is_first = True
            self.is_sac = False
            self.label = 0 # FIXME

        def check(self, label):
            if self.is_first:
                self.is_first = False
                self.is_sac = True
                self.label = label
            elif self.is_sac:
                self.is_sac = self.label == label

    consistencies = [Consistency() for i in range(model.number_of_variables)]
    for factor_index, factor in enumerate(model.factors):
        if factor.number_of_variables < 2:
            continue

        arg_minimum = numpy.unravel_index(factor.data.argmin(), factor.shape)
        minimum = factor.data[arg_minimum]

        for variable_index, variable in enumerate(factor.variables):
            consistencies[variable].check(arg_minimum[variable_index])

    # FIXME: If is_first then set label to argmin and is_sac to true.

    mask = numpy.zeros(model.number_of_variables, dtype=bool)
    labeling = make_labeling(model.number_of_variables)
    for i in range(model.number_of_variables):
        mask[i] = consistencies[i].is_sac
        labeling[i] = consistencies[i].label

    print('Arc inconsistent set size = {} / {} ({:.2f}%)'.format(
        model.number_of_variables - mask.sum(), model.number_of_variables,
        (model.number_of_variables - mask.sum()) / model.number_of_variables * 100.0))

    return mask, labeling

def reparametrize_border(repa, mask):
    for factor_index, factor in enumerate(repa.model.factors):
        flags = [mask[x] for x in factor.variables]
        if False in flags and True in flags:
            reparametrize_border_factor(repa, factor_index, [i for i, v in enumerate(flags) if v])

def reparametrize_border_factor(repa, factor_index, to_vars):
    factor = repa.model.factors[factor_index]

    for to_labeling in walk_shape([factor.shape[x] for x in to_vars]):
        index = [slice(None) for i in range(factor.number_of_variables)]
        for i, l in zip(to_vars, to_labeling):
            index[i] = l
        minimum = repa.get_factor_value(factor_index)[tuple(index)].min()

        for i, l in zip(to_vars, to_labeling):
            repa.get_factor(factor_index, i)[l] += minimum / len(to_vars)
