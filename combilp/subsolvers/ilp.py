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

import ctypes
import itertools
import math
import numpy

from ctypes import POINTER, c_bool, c_int32, c_int64, c_void_p
from numpy.ctypeslib import ndpointer

from .common import SubSolver
from ..model import Factor, Model, make_labeling, walk_shape, walk_sub_shape

try:
    import cplex
except ImportError:
    cplex = None

try:
    import gurobipy
except ImportError:
    gurobipy = None

def cplex_is_optimal(cpl):
    optimal_statuses = (
        cpl.solution.status.MIP_optimal,
        cpl.solution.status.MIP_optimal,
        cpl.solution.status.optimal_tolerance,
    )
    return cpl.solution.get_status() in optimal_statuses

def range_length(start, length):
    return range(start, start + length)

class ILPSolver(SubSolver):

    def add_full_model(self):
        for variable in range(self.model.number_of_variables):
            self.add_variable(variable)

        for factor in self.model.factors:
            self.add_factor(factor)

    def add_variable(self, variable):
        raise NotImplementedError

    def add_factor(self, factor):
        raise NotImplementedError

    def lower_bound(self):
        return self.upper_bound()

    def prepare(self):
        pass

class Cplex(ILPSolver):

    DEFAULT_PARAMETERS = {
        'threads': None,
    }

    def __init__(self, model, parameters=None):
        super().__init__(model, parameters)
        if not cplex:
            raise RuntimeError('Required module cplex is not available.')

        self._cplex = cplex.Cplex()
        self._variables = [None] * model.number_of_variables # FIXME: consider numpy array
        self.constant = 0.0

    def index_of_variable(self, variable, label=0):
        assert(self._variables[variable] is not None)
        assert(label >= 0 and label < self.model.shape[variable])
        return self._variables[variable] + label

    def add_variable(self, variable):
        self._variables[variable] = self._cplex.variables.get_num()
        num_labs = self.model.shape[variable]
        lb = [0.0] * num_labs
        ub = [1.0] * num_labs
        types = [self._cplex.variables.type.integer] * num_labs
        self._cplex.variables.add(lb=lb, ub=ub, types=types)

        ind = range_length(self._variables[variable], num_labs)
        lin_expr = cplex.SparsePair(ind, val=[1.0]*len(ind))
        self._cplex.linear_constraints.add(lin_expr=[lin_expr], senses=['E'], rhs=[1.0])

    def add_factor(self, factor):
        # copy factor for normalization
        factor = Factor(factor.variables, data=numpy.copy(factor.data))
        minimum = factor.data.min()
        self.constant += minimum
        factor.data -= minimum

        # handle infinity
        factor.data[factor.data > cplex.infinity] = cplex.infinity
        factor.data[factor.data < -cplex.infinity] = -cplex.infinity

        if factor.number_of_variables == 1:
            self._add_factor_unary(factor)
        else:
            self._add_factor_generic(factor)

    def _add_factor_unary(self, factor):
        variable, = factor.variables
        start = self.index_of_variable(variable)
        end = int(start + self.model.shape[variable] - 1)
        old = self._cplex.objective.get_linear(start, end)
        self._cplex.objective.set_linear(
            (i, o+v) for i, o, v in zip(range(start, end+1), old, factor.data))

    def _add_factor_generic(self, factor):
        first_index = self._cplex.variables.get_num()
        obj = factor.data.ravel()
        lb = [0.0] * len(obj)
        ub = [1.0] * len(obj)
        self._cplex.variables.add(lb=lb, ub=ub, obj=obj)

        number_of_constraints = sum(factor.shape)
        lin_expr = [cplex.SparsePair() for x in range(number_of_constraints)]

        it = iter(lin_expr)
        for variable, labels in zip(factor.variables, factor.shape):
            for label in range(labels):
                current = next(it)
                current.ind.append(self.index_of_variable(variable, label))
                current.val.append(-1.0)

        # FIXME: This here is rather slow
        it = numpy.nditer(factor.data, flags=['c_index', 'multi_index'])
        while not it.finished:
            contraint_index = 0
            for local_variable, labels in enumerate(factor.shape):
                current = lin_expr[contraint_index + it.multi_index[local_variable]]
                current.ind.append(first_index + it.index)
                current.val.append(1.0)
                contraint_index += labels
            it.iternext()
        self._cplex.linear_constraints.add(lin_expr=lin_expr,
            senses=['E']*len(lin_expr), range_values=[0.0]*len(lin_expr))

    def solve(self):
        if self.parameters['threads']:
            self._cplex.parameters.threads.set(self.parameters['threads'])
        self._cplex.solve()
        if not cplex_is_optimal(self._cplex):
            raise RuntimeError('CPLEX inference was not optimal')

    def upper_bound(self):
        return self._cplex.solution.get_objective_value() + self.constant

    def labeling(self):
        result = [None] * self.model.number_of_variables
        for variable in range(len(result)):
            if self._variables[variable] is None:
                continue

            for label in range(self.model.number_of_labels(variable)):
                if self._cplex.solution.get_values(self.index_of_variable(variable, label)) > .5:
                    result[variable] = label
                    break
        return result

class Gurobi(ILPSolver):

    DEFAULT_PARAMETERS = {
        'threads': 0,
    }

    def __init__(self, model, parameters=None):
        super().__init__(model, parameters)
        if not gurobipy:
            raise RuntimeError('Required module gurobipy is not available.')

        self._env = gurobipy.Env()
        self._model = gurobipy.Model(env=self._env)
        self._obj = gurobipy.LinExpr()
        self._variables = [None] * model.number_of_variables
        self._constant = 0.0

    def add_variable(self, variable):
        num_labs = int(self.model.shape[variable])
        result = self._model.addVars(num_labs, lb=0.0, ub=1.0, vtype=gurobipy.GRB.BINARY)
        self._model.addConstr(result.sum() == 1.0)
        self._variables[variable] = result

    def add_factor(self, factor):
        # copy factor for normalization
        factor = Factor(factor.variables, data=numpy.copy(factor.data))
        minimum = factor.data.min()
        self._constant += minimum
        factor.data -= minimum

        # handle infinity
        factor.data[factor.data > gurobipy.GRB.INFINITY] = gurobipy.GRB.INFINITY
        factor.data[factor.data < -gurobipy.GRB.INFINITY] = -gurobipy.GRB.INFINITY

        if factor.number_of_variables == 1:
            self._add_factor_unary(factor)
        else:
            self._add_factor_generic(factor)

    def _add_factor_unary(self, factor):
        variable, = factor.variables
        self._obj.addTerms(factor.data, self._variables[variable].select())

    def _add_factor_generic(self, factor):
        result = self._model.addVars(*factor.shape, lb=0.0, ub=1.0)
        # FIXME: Check correct index ordering
        self._obj.addTerms(factor.data.ravel(), result.select())

        # FIXME: This is slow as hell.
        for variable, num_labels in enumerate(factor.shape):
            for label in range(num_labels):
                unary_var = self._variables[factor.variables[variable]][label]
                selector = ['*'] * num_labels
                selector[variable] = label
                lin_expr = sum(result.select(*selector)) - unary_var
                self._model.addConstr(lin_expr == 0.0)

    def solve(self):
        self._model.setParam('Threads', self.parameters['threads'])
        self._model.setObjective(self._obj, gurobipy.GRB.MINIMIZE)
        self._model.optimize()

    def upper_bound(self):
        return self._model.getObjective().getValue() + self._constant

    def labeling(self):
        result = [None] * self.model.number_of_variables
        for variable in range(len(result)):
            if self._variables[variable] is None:
                continue

            assert(len([k for k, v in self._variables[variable].items() if v.x >.5]) == 1)

            for label, var in self._variables[variable].items():
                if var.x > .5:
                    result[variable] = label
                    break
        return result

class NonIterativeSolver(ILPSolver):

    DEFAULT_PARAMETERS = {
        'warmstart': None,
    }

    def __init__(self, model, parameters=None):
        super().__init__(model, parameters)
        self.solver = None
        self.variable_map = None
        self.variables = []
        self.factors = []
        self.warmstart = self.parameters['warmstart']
        del self.parameters['warmstart']

    def add_variable(self, variable):
        self.variables.append(variable)

    def add_factor(self, factor):
        self.factors.append(factor)

    def prepare(self):
        print('Rebuilding full model...')
        self.variable_map = dict((x, i) for i, x in enumerate(self.variables))
        model = Model([self.model.number_of_labels(x) for x in self.variables])
        for factor in self.factors:
            wrapped_factor = Factor([self.variable_map[x] for x in factor.variables],
                data=factor.data)
            model.add_factor(wrapped_factor)

        params = self.parameters
        if self.warmstart is not None:
            print('Adjusting warmstart parameter...')
            if self.solver:
                prev_labeling = self.labeling()
            else:
                prev_labeling = [None] * self.model.number_of_variables
            current_warmstart = make_labeling(model.number_of_variables)
            for variable in range(model.number_of_variables):
                original_variable = self.variables[variable]
                if prev_labeling[original_variable] is not None:
                    current_warmstart[variable] = prev_labeling[original_variable]
                else:
                    current_warmstart[variable] = self.warmstart[original_variable]
            params = self.parameters.copy()
            params['warmstart'] = current_warmstart

        print('Reconstructing new solver...')
        self.solver = self._SOLVER(model, params)
        self.solver.add_full_model()

    def solve(self):
        return self.solver.solve()

    def upper_bound(self):
        return self.solver.upper_bound()

    def labeling(self):
        result = [None] * self.model.number_of_variables
        for variable, label in enumerate(self.solver.labeling()):
            result[self.variables[variable]] = label
        return result

class CplexNonIterative(NonIterativeSolver):
    _SOLVER = Cplex

class GurobiNonIterative(NonIterativeSolver):
    _SOLVER = Gurobi

class ToulBar2(NonIterativeSolver):
    class _SOLVER(ILPSolver):

        DEFAULT_PARAMETERS = {
            'scaling_factor': 1e6,
            'warmstart': None,
            'vac': True,
        }

        def __init__(self, model, parameters=None):
            super().__init__(model, parameters)
            self.constant = 0
            self._init_library()

            min_cost, max_cost = c_int64(), c_int64()
            self._initialize(self.parameters['vac'], min_cost, max_cost)
            self.min_cost, self.max_cost = min_cost.value, max_cost.value

        def __del__(self):
            if self._solver:
                self._destroy(self._solver)
                self._solver = None

        def _init_library(self):
            self._lib = ctypes.cdll.LoadLibrary('libcombilp_toulbar2_stub.so')

            self._initialize = self._lib.combilp_toulbar2_stub_initialize
            self._initialize.argtypes = [c_bool, POINTER(c_int64), POINTER(c_int64)]

            self._create = self._lib.combilp_toulbar2_stub_create
            self._create.argtypes = [c_int32, ndpointer(dtype=c_int32)]
            self._create.restype = c_void_p

            self._destroy = self._lib.combilp_toulbar2_stub_destroy
            self._destroy.argtypes = [c_void_p]

            self._add_factor = self._lib.combilp_toulbar2_stub_add_factor
            self._add_factor.argtypes = [c_void_p, c_int32, ndpointer(dtype=c_int32), ndpointer(dtype=c_int32), ndpointer(dtype=c_int64)]

            self._solve = self._lib.combilp_toulbar2_stub_solve
            self._solve.argtypes = [c_void_p, c_int64]
            self._solve.restype = c_bool

            self._get_labeling = self._lib.combilp_toulbar2_stub_get_labeling
            self._get_labeling.argtypes = [c_void_p, ndpointer(dtype=c_int32)]

        def add_full_model(self):
            self._solver = self._create(self.model.number_of_variables, self.model.shape)
            for factor in self.model.factors:
                shape = [self.model.number_of_labels(x) for x in factor.variables]
                shape = numpy.asarray(shape, dtype=c_int32)
                self._add_factor(self._solver, factor.number_of_variables,
                    factor.variables, shape, self.convert_costs(factor.data))
                self.constant += factor.data.min()

        def solve(self):
            ub = self.max_cost
            if self.parameters['warmstart'] is not None:
                possible_ub = (self.model.evaluate(self.parameters['warmstart']) - self.constant) * self.parameters['scaling_factor']
                if not math.isinf(possible_ub):
                    ub = int(possible_ub) + 1

            result = self._solve(self._solver, ub)
            if not result:
                raise RuntimeError('Inference was not optimal.')

        def upper_bound(self):
            labeling = self.labeling()
            return self.model.evaluate(labeling)

        def labeling(self):
            result = make_labeling(self.model.number_of_variables)
            self._get_labeling(self._solver, result)
            return result

        def convert_costs(self, values):
            minimal = values.min()
            result = numpy.asarray((values - minimal) * self.parameters['scaling_factor'],
                dtype=c_int64)
            result[numpy.isposinf(values)] = self.max_cost
            return result
