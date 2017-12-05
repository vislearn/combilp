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

import itertools
import math
import numpy
import sys

from .model import walk_shape, walk_sub_shape, Model, Factor
from .reparametrization import Reparametrization
from .utils import (compute_strict_arc_consistency, PerformanceMeasurement,
    PerformanceStatistics, reparametrize_border, reparametrize_border_factor,
    unary_reparametrization)
from .subsolvers.lp import TRWS
from .subsolvers.ilp import Cplex

DEFAULTS = {
    'lp_solver': TRWS,
    'ilp_solver': Cplex,
    'max_iterations': sys.maxsize,
    'lp': {},
    'ilp': {},
}

class CombiLP(object):

    def __init__(self, model, parameters=None):
        self.model = model
        self.mask = [False] * model.number_of_variables
        self.timers = PerformanceStatistics()

        self.parameters = DEFAULTS.copy()
        if (parameters):
            self.parameters.update(parameters)

    def solve_lp_relaxation(self):
        print('--- Solving LP Relaxation ---', flush=True)

        with self.timers.lp_solver_construction:
            lp_solver = self.parameters['lp_solver'](self.model, self.parameters['lp'])
        with self.timers.lp_solver_solve:
            lp_solver.solve()
        with self.timers.reparametrization:
            self.repa = lp_solver.get_repametrization()
            unary_reparametrization(self.repa)
        with self.timers.sac_computation:
            self.sac_mask, self.sac_labeling = compute_strict_arc_consistency(
                self.repa.reparametrize_model(dynamic=False))

    # This function does not perform border repa, watch out!
    def add_varariable_to_ilp(self, variable):
        assert(not self.mask[variable])
        self.mask[variable] = True
        with self.timers.add_variable:
            self.ilp_solver.add_variable(variable)

        with self.timers.add_factor:
            for factor in self.rmodel.factors_of_variable(variable):
                is_included = True
                for neighbor in factor.variables:
                    if variable != neighbor and not self.mask[neighbor]:
                        is_included = False

                if is_included:
                    self.ilp_solver.add_factor(factor)

    # This function performs dense repa once before adding the variables.
    def add_varariables_to_ilp(self, variables):
        assert(not any(self.mask[x] for x in variables))
        print('Adding {} variables to ILP subproblem.'.format(len(variables)))

        with self.timers.reparametrization:
            new_mask = self.mask.copy()
            for variable in variables:
                new_mask[variable] = True
            reparametrize_border(self.repa, new_mask)

        for variable in variables:
            self.add_varariable_to_ilp(variable)

        assert(self.mask == new_mask)

    def update_labeling(self):
        ilp_labeling = self.ilp_solver.labeling()
        self.labeling = numpy.copy(self.sac_labeling)
        for variable, masked in enumerate(self.mask):
            if masked:
                self.labeling[variable] = ilp_labeling[variable]

    def update_lower_bound(self):
        self.mismatches = []
        self.lower_bound = 0.0
        for factor in self.rmodel.factors:
            ilp_flags = [self.mask[x] for x in factor.variables]
            if not any(ilp_flags) or all(ilp_flags):
                self.lower_bound += factor.local_evaluate(self.labeling)
            else:
                minimum = factor.data.min()
                self.lower_bound += minimum
                if abs(factor.local_evaluate(self.labeling) - minimum) > 1e-10:
                    self.mismatches.append(factor)

    def run_iteration(self):
        with self.timers.ilp_solver_solve:
            self.ilp_solver.solve()
        print('elapsed ILP time: {:.2f}s'.format(self.timers.ilp_solver_solve.last))
        print('solution for ILP: {}'.format(self.ilp_solver.upper_bound()))

        with self.timers.evaluate:
            self.update_labeling()
            self.upper_bound = self.model.evaluate(self.labeling)
            self.update_lower_bound()

        print('current bounds: {} <= {} (delta: {})'.format(self.lower_bound,
            self.upper_bound, self.upper_bound - self.lower_bound))

    def solve(self):
        time_everything = PerformanceMeasurement()
        with time_everything:
            self.solve_lp_relaxation()
            self.rmodel = self.repa.reparametrize_model(dynamic=True)
            self.ilp_solver = self.parameters['ilp_solver'](self.rmodel, self.parameters['ilp'])

            self.add_varariables_to_ilp([v for v, x in enumerate(self.sac_mask) if not x])
            assert(self.mask == [not x for x in self.sac_mask])

            for combilp_iteration in range(self.parameters['max_iterations']):
                print('\n--- CombiLP iteration {} ({} variables) ---'.format(
                    combilp_iteration + 1, sum(self.mask)), flush=True)
                self.run_iteration()
                print('{} mismatching factors'.format(len(self.mismatches)))
                if not self.mismatches:
                    break

                variables_to_add = set(filter(lambda x: not self.mask[x],
                    itertools.chain.from_iterable(factor.variables for factor in self.mismatches)))
                self.add_varariables_to_ilp(variables_to_add)

        misc = time_everything.total - sum(t.total for _, t in self.timers.items())
        print('\n--- Summary ---')
        print('solution = {}'.format(self.upper_bound))
        print('combilp iterations = {}'.format(combilp_iteration + 1))
        print('final ilp size = {}'.format(sum(self.mask)))
        for name, performance in self.timers.items():
            print('time({}) = {:.2f}s'.format(name, performance.total))
        print('time(UNKNOWN) = {:.2f}s'.format(misc))
        print('time(TOTAL) = {:.2f}s'.format(time_everything.total))
