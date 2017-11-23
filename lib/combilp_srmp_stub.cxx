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
#include <iostream>
#include <limits>

#include <srmp/SRMP.h>
#include <srmp/edge_iterator.h>

typedef std::int32_t IndexType;
typedef std::int32_t LabelType;
typedef double ValueType;

extern "C" {

srmpLib::Energy*
combilp_srmp_stub_create(
	IndexType number_of_variables,
	const LabelType *shape
)
{
	srmpLib::Energy *energy = new srmpLib::Energy(number_of_variables);

	for (IndexType var = 0; var < number_of_variables; ++var)
		energy->AddNode(shape[var]);
	
	return energy;
}

void
combilp_srmp_stub_destroy(srmpLib::Energy *energy)
{
	delete energy;
}

void
combilp_srmp_stub_add_factor(
	srmpLib::Energy *energy,
	IndexType arity,
	const IndexType *variables,
	const ValueType *costs
)
{
	static_assert(sizeof(srmpLib::Energy::NodeId) == sizeof(IndexType));
	energy->AddFactor(arity, const_cast<srmpLib::Energy::NodeId*>(variables),
		const_cast<double*>(costs));
}

void
combilp_srmp_stub_solve(
	srmpLib::Energy *energy,
	int max_iterations
)
{
	srmpLib::Energy::Options options;
	options.method = srmpLib::Energy::Options::SRMP;
	options.iter_max = max_iterations;
	options.time_max = std::numeric_limits<double>::infinity();
	options.sort_flag = -1;
	options.print_times = false;
	options.verbose = true;

	energy->SetMinimalEdges();
	energy->Solve(options);
}

void
combilp_srmp_stub_extract_messages
(
	srmpLib::Energy *energy,
	void (*func)(size_t alpha_size, IndexType *alpha, IndexType beta, ValueType *message_begin, ValueType *message_end)
)
{
	for (srmpLib::EdgeIterator it(energy); it.valid(); ++it) {
		if (! (it->alpha.size() >= 2 && it->beta.size() == 1))
			throw std::runtime_error("This relaxation type is not supported by the LPReparametrisationStorage!");

		func(it->alpha.size(), it->alpha.data(), it->beta[0], it->message_begin(), it->message_end());
	}
}

}
