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
import numpy
import sys

from ctypes import CFUNCTYPE, POINTER, c_double, c_int, c_int32, c_void_p, c_size_t
from numpy.ctypeslib import ndpointer

from .common import SubSolver
from ..model import walk_shape
from ..reparametrization import Reparametrization

class LPSolver(SubSolver):
    def get_repametrization(self):
        raise NotImplementedError

class TRWS(LPSolver):

    DEFAULT_PARAMETERS = {
        'max_iterations': 2000,
        'threads': 1,
    }

    def __init__(self, model, parameters=None):
        super().__init__(model, parameters)
        self._init_library()

        self.model = model
        self._energy = self._energy_create(model.number_of_variables, model.shape,
            sum(1 for x in model.factors if x.number_of_variables > 1))

        edge_counter = 0
        for i, factor in enumerate(model.factors):
            if factor.number_of_variables == 1:
                self._energy_add_unary(self._energy, factor.variables[0], factor.data)
            elif factor.number_of_variables == 2:
                self._energy_add_pairwise(self._energy, edge_counter, *factor.variables, factor.data)
                edge_counter += 1
            else:
                raise RuntimeError('Unsupported factor arity.')

        self._energy_finalize(self._energy)
        self._solver = self._solver_create(self._energy)

    def __del__(self):
        if self._energy:
            self._energy_destroy(self._energy)
            self._energy = None
        if self._solver:
            self._solver_destroy(self._solver)
            self._solver = None

    def _init_library(self):
        self._lib = ctypes.cdll.LoadLibrary('libcombilp_trws_stub.so')

        self._energy_create = self._lib.combilp_trws_stub_energy_create
        self._energy_create.argtypes = [c_int32, ndpointer(dtype=c_int32), c_int32]
        self._energy_create.restype = c_void_p

        self._energy_add_unary = self._lib.combilp_trws_stub_energy_add_unary
        self._energy_add_unary.argtypes = [c_void_p, c_int32, ndpointer(dtype=c_double)]

        self._energy_add_pairwise = self._lib.combilp_trws_stub_energy_add_pairwise
        self._energy_add_pairwise.argtypes = [c_void_p, c_int32, c_int32, c_int32, ndpointer(dtype=c_double)]

        self._energy_finalize = self._lib.combilp_trws_stub_energy_finalize
        self._energy_finalize.argtypes = [c_void_p]

        self._energy_destroy = self._lib.combilp_trws_stub_energy_destroy
        self._energy_destroy.argtypes = [c_void_p]

        self._solver_create = self._lib.combilp_trws_stub_solver_create
        self._solver_create.argtypes = [c_void_p]
        self._solver_create.restype = c_void_p

        self._solve = self._lib.combilp_trws_stub_solve
        self._solve.argtypes = [c_void_p, c_int, c_int]

        self._solver_destroy = self._lib.combilp_trws_stub_destroy_solver
        self._solver_destroy.argtypes = [c_void_p]

        self._get_backward_messages = self._lib.combilp_trws_stub_get_backward_messages
        self._get_backward_messages.argtypes = [c_void_p, c_int32, ndpointer(dtype=c_double)]

    def solve(self):
        self._solve(self._solver,
            self.parameters['max_iterations'],
            self.parameters['threads'])

    def get_repametrization(self):
        repa = Reparametrization(self.model)
        edge_counter = 0
        for i, factor in enumerate(self.model.factors):
            if factor.number_of_variables == 2:
                self._get_backward_messages(self._solver, edge_counter,
                    repa.get_factor(i, 0))
                edge_counter += 1

                # recompute forward messages
                values = repa.get_factor_value(i)
                repa_values = repa.get_factor(i, 1)
                for label in range(factor.shape[1]):
                    minimum = values[:,label].min()
                    repa_values[label] = minimum
        return repa

class SRMP(LPSolver):

    DEFAULT_PARAMETERS = {
        'max_iterations': 2000,
    }

    def __init__(self, model, parameters=None):
        super().__init__(model, parameters)
        self._init_library()

        self._solver = self._create(self.model.number_of_variables, self.model.shape)
        for factor in self.model.factors:
            assert(factor.data.flags.c_contiguous)
            self._add_factor(self._solver, factor.number_of_variables,
                factor.variables, factor.data)

    def __del__(self):
        if self._solver:
            self._destroy(self._solver)
            self._solver = None

    def _init_library(self):
        self._lib = ctypes.cdll.LoadLibrary('libcombilp_srmp_stub.so')
        self._message_func_type = CFUNCTYPE(None, c_size_t, POINTER(c_int32), c_int32, POINTER(c_double), POINTER(c_double))
        self._message_func_type.from_param = self._message_func_type

        self._create = self._lib.combilp_srmp_stub_create
        self._create.argtypes = [c_int32, ndpointer(dtype=c_int32)]
        self._create.restype = c_void_p

        self._destroy = self._lib.combilp_srmp_stub_destroy
        self._destroy.argtypes = [c_void_p]

        self._add_factor = self._lib.combilp_srmp_stub_add_factor
        self._add_factor.argtypes = [c_void_p, c_int32, ndpointer(dtype=c_int32), ndpointer(dtype=c_double)]

        self._solve = self._lib.combilp_srmp_stub_solve
        self._solve.argtypes = [c_void_p, c_int]

        self._extract_messages = self._lib.combilp_srmp_stub_extract_messages
        self._extract_messages.argtypes = [c_void_p, self._message_func_type]

    def solve(self):
        self._solve(self._solver, self.parameters['max_iterations'])

    def get_repametrization(self):
        result = Reparametrization(self.model)

        # The edge_iterator of SRMP returns factors *exactly* in our own factor
        # order which is *awesome*. :)
        current_factor = 0

        def find_factor(variables):
            nonlocal current_factor
            for factor_index in range(current_factor, len(self.model.factors)):
                factor = self.model.factors[factor_index]
                if numpy.array_equal(factor.variables, variables):
                    return factor_index, factor
                current_factor = factor_index

        def func(alphas_size, alphas, beta, message, message_end):
            alphas = [alphas[i] for i in range(alphas_size)]
            factor_index, factor = find_factor(alphas)
            local_variable_index = alphas.index(beta)
            r = result.get_factor(factor_index, local_variable_index)
            r[:] = message[:r.size]

        self._extract_messages(self._solver, func)
        return result
