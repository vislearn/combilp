// Copyright (c) 2017 Stefan Haller
//
// Permission is hereby granted, free of charge, to any person obtaining a copy
// of this software and associated documentation files (the "Software"), to deal
// in the Software without restriction, including without limitation the rights
// to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
// copies of the Software, and to permit persons to whom the Software is
// furnished to do so, subject to the following conditions:
//
// The above copyright notice and this permission notice shall be included in all
// copies or substantial portions of the Software.
//
// THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
// IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
// FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
// AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
// LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
// OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
// SOFTWARE.

#include <cstdint>

#include <optim/part_opt/trws_machine.h>

typedef std::int32_t IndexType;
typedef std::int32_t LabelType;
typedef double ValueType;
typedef trws_machine<float_v4> Trws;
typedef energy_auto<float> TrwsEnergy;

extern "C" {

TrwsEnergy*
combilp_trws_stub_energy_create
(
	IndexType number_of_variables,
	const LabelType *shape,
	IndexType number_of_edges
)
{
	LabelType maxK = 0;
	for (IndexType i = 0; i < number_of_variables; ++i)
		maxK = std::max(maxK, shape[i]);

	TrwsEnergy *energy = new TrwsEnergy;

	energy->set_nV(number_of_variables);
	energy->G._nV = number_of_variables;
	energy->K.resize(number_of_variables);
	for (IndexType i = 0; i < number_of_variables; ++i)
		energy->K[i] = shape[i];
	
	energy->set_nE(number_of_edges);
	energy->maxK = maxK;

	return energy;
}

void
combilp_trws_stub_energy_add_unary
(
	TrwsEnergy *energy,
	IndexType var,
	const ValueType *data
)
{
	// double is not a type, set_f1 does not use template parameter!
	dynamic::num_array<double, 1> f1(energy->K[var]);
	for (LabelType i = 0; i < energy->K[var]; ++i)
		f1[i] = data[i];
	energy->set_f1(var, f1);
}

void
combilp_trws_stub_energy_add_pairwise
(
	TrwsEnergy *energy,
	IndexType edgeIndex,
	IndexType var0,
	IndexType var1,
	const ValueType *data
)
{
	// double is not a type, set_f2 does not use template parameter!
	dynamic::num_array<double, 2> f2(exttype::mint2(energy->K[var0], energy->K[var1]));
	int z = 0;
	for (LabelType i = 0; i < energy->K[var0]; ++i)
		for (LabelType j = 0; j < energy->K[var1]; ++j, ++z)
			f2(i,j) = data[z];
	energy->G.E[edgeIndex][0] = var0;
	energy->G.E[edgeIndex][1] = var1;
	energy->set_f2(edgeIndex, f2);
}

void
combilp_trws_stub_energy_finalize(TrwsEnergy *energy)
{
	energy->G.edge_index();
	energy->init();
	energy->report();
}

void
combilp_trws_stub_energy_destroy(TrwsEnergy *energy)
{
	delete energy;
}

Trws*
combilp_trws_stub_solver_create(TrwsEnergy *energy)
{
	Trws *trws = new Trws();
	trws->init(energy);
	return trws;
}

void
combilp_trws_stub_solve(Trws *trws, int iterations, int threads)
{
	trws->ops->max_it = iterations;
	trws->ops->max_CPU = threads;
	trws->ops->it_batch = 100;
	trws->ops->rel_gap_tol = 1e-5;
	trws->ops->rel_conv_tol = 1e-5;
	trws->run_converge();
}

void
combilp_trws_stub_destroy_solver(Trws *trws)
{
	delete trws;
}


void combilp_trws_stub_get_backward_messages
(
	Trws *trws,
	IndexType edgeIndex,
	ValueType *backward
)
{
	std::vector<Trws::type> msgs;
	trws->get_message(edgeIndex, false, msgs);
	std::copy(msgs.begin(), msgs.end(), backward);
}

}
