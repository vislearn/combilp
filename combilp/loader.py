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

from ctypes import POINTER, CFUNCTYPE, c_char_p, c_double, c_int

from .model import Model, Factor

_opengm_stub = None

def _init_opengm_stub():
    g = globals()

    if g['_opengm_stub']:
        return

    g['_opengm_stub'] = ctypes.cdll.LoadLibrary('libcombilp_opengm_stub.so')

    g['_init_shape_func_type'] = CFUNCTYPE(None, c_int, POINTER(c_int))
    g['_init_shape_func_type'].from_param = g['_init_shape_func_type']
    g['_add_factor_func_type'] = CFUNCTYPE(None, c_int, POINTER(c_int), POINTER(c_double))
    g['_add_factor_func_type'].from_param = g['_add_factor_func_type']

    g['_load_from_file'] = _opengm_stub.combilp_opengm_stub_load_from_file
    g['_load_from_file'].argtypes = [c_char_p, c_char_p, g['_init_shape_func_type'], g['_add_factor_func_type']]
    g['_load_from_file'].restype = None

def load_hdf5(filename, dataset='gm'):
    _init_opengm_stub()

    model = None

    def init_shape(number_of_variables, shape):
        nonlocal model
        model = Model([shape[i] for i in range(number_of_variables)]) # FIXME: slow

    def add_factor(number_of_variables, variables, data):
        variables = [variables[i] for i in range(number_of_variables)]
        shape = [model.shape[v] for v in variables]
        factor = Factor(variables, shape)
        assert(factor.data.flags.c_contiguous)
        it = numpy.nditer(factor.data, flags=['c_index'], op_flags=['readwrite'])
        while not it.finished: # FIXME: slow
            value = data[it.index]
            it[0] = value
            it.iternext()
        model.add_factor(factor)

    _load_from_file(filename.encode('utf-8'), dataset.encode('utf-8'), init_shape, add_factor)
    return model
