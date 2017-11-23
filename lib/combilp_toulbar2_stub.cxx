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

#include <algorithm>
#include <climits>
#include <cstdint>
#include <functional>
#include <numeric>
#include <sstream>

#include <toulbar2lib.hpp>

typedef std::int32_t IndexType;
typedef std::int32_t LabelType;
typedef std::int64_t ValueType;

extern "C" {

void
combilp_toulbar2_stub_initialize
(
	ValueType *out_min_cost,
	ValueType *out_max_cost
)
{
	tb2init();
	ToulBar2::uai = 1;
	ToulBar2::bayesian = true;
	ToulBar2::vac = 1;
	ToulBar2::vacValueHeuristic = true;
	ToulBar2::hbfs = 1;
	ToulBar2::hbfsGlobalLimit = 10000;
	ToulBar2::DEE = 1;

	*out_min_cost = MIN_COST;
	*out_max_cost = MAX_COST;
}

WeightedCSPSolver*
combilp_toulbar2_stub_create
(
	IndexType number_of_variables,
	LabelType *shape
)
{
	auto *solver = WeightedCSPSolver::makeWeightedCSPSolver(MAX_COST);
	auto *prob = solver->getWCSP();

	for (IndexType i = 0; i < number_of_variables; ++i) {
		std::stringstream ss;
		ss << "x" << i;
		prob->makeEnumeratedVariable(ss.str(), 0, shape[i] - 1);
	}

	return solver;
}

void
combilp_toulbar2_stub_destroy
(
	WeightedCSPSolver *solver
)
{
	delete solver;
}

void
combilp_toulbar2_stub_add_factor
(
	WeightedCSPSolver *solver,
	IndexType arity,
	const IndexType *variables,
	const IndexType *shape,
	const ValueType *costs
)
{
	const size_t total = std::accumulate(shape, shape + arity, size_t{1}, std::multiplies<size_t>());
	std::vector<Cost> cost_vector(costs, costs + total);

	auto *prob = solver->getWCSP();
	assert(arity > 0);
	switch (arity) {
		case 1:
			prob->postUnaryConstraint(variables[0], cost_vector);
			break;
		case 2:
			prob->postBinaryConstraint(variables[0], variables[1], cost_vector);
			break;
		case 3:
			prob->postTernaryConstraint(variables[0], variables[1], variables[2], cost_vector);
			break;
		default: {
			std::vector<Value> variable_vector(variables, variables + arity);
			int constraint = prob->postNaryConstraintBegin(&variable_vector[0], arity, 0);

			variable_vector.assign(arity, 0);
			for (size_t i = 0; i < total; ++i) {
				prob->postNaryConstraintTuple(constraint, &variable_vector[0], arity, cost_vector[i]);

				++variable_vector[arity - 1];
				for (IndexType j = 0; j < arity; ++j) {
					auto index = arity - j - 1;
					if (variable_vector[index] >= shape[index]) {
						--variable_vector[index];
						++variable_vector[index - 1];
					}
				}
			}

			prob->postNaryConstraintEnd(constraint);
		}
	}
}

bool
combilp_toulbar2_stub_solve
(
	WeightedCSPSolver *solver
)
{
	auto *prob = solver->getWCSP();
	prob->sortConstraints(); // Needs to be called before search.

	return solver->solve();
}

void
combilp_toulbar2_stub_get_labeling
(
	WeightedCSPSolver *solver,
	IndexType *out_labeling
)
{
	const std::vector<int> &solution = solver->getWCSP()->getSolution();
	std::copy(solution.begin(), solution.end(), out_labeling);
}

}
