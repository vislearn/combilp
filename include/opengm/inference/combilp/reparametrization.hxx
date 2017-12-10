//
// File: reparamtrization.hxx
//
// This file is part of OpenGM.
//
// Copyright (C) 2016-2017 Stefan Haller
//
// Permission is hereby granted, free of charge, to any person obtaining a copy
// of this software and associated documentation files (the "Software"), to
// deal in the Software without restriction, including without limitation the
// rights to use, copy, modify, merge, publish, distribute, sublicense, and/or
// sell copies of the Software, and to permit persons to whom the Software is
// furnished to do so, subject to the following conditions:
//
// The above copyright notice and this permission notice shall be included in
// all copies or substantial portions of the Software.
//
// THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
// IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
// FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
// AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
// LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING
// FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS
// IN THE SOFTWARE.
//

#pragma once
#ifndef OPENGM_COMBILP_REPARAMETRIZATION_HXX
#define OPENGM_COMBILP_REPARAMETRIZATION_HXX

#include <iomanip>
#include <iostream>
#include <stack>
#include <vector>

#include <opengm/graphicalmodel/graphicalmodel.hxx>
#include <opengm/inference/auxiliary/lp_reparametrization.hxx>

#include <opengm/inference/combilp/utils.hxx>

#ifdef WITH_CPLEX
#include <opengm/inference/lpcplex.hxx>
#endif

namespace opengm {
namespace combilp {

//
// Push potentials into pairwise terms. The unary potentials is split evenly
// across all neighbors (more correctly all neighboring factors).
//
template<class ACC, class REPA>
void
pairwiseReparametrization
(
	REPA &repa
)
{

	typedef typename REPA::GraphicalModelType GraphicalModelType;
	typedef typename GraphicalModelType::IndexType IndexType;
	typedef typename GraphicalModelType::LabelType LabelType;
	typedef typename GraphicalModelType::ValueType ValueType;

	const GraphicalModelType &gm = repa.graphicalModel();

#ifdef OPENGM_COMBILP_DEBUG
	{
		std::vector<float> values(gm.numberOfVariables());
		for (IndexType i = 0; i < gm.numberOfVariables(); ++i) {
			std::vector<float> tmp(gm.numberOfLabels(i));
			for (IndexType j = 0; j < gm.numberOfLabels(i); ++j) {
				tmp[j] = repa.getVariableValue(i, j);
			}
			std::sort(tmp.begin(), tmp.end());
			values[i] = tmp[tmp.size() / 2];
		}
		std::sort(values.begin(), values.end());
		float median = values[values.size() / 2];
		std::cout << "Median of unary medians = " << median << std::endl;
	}
#endif

	//
	// Count the number of neighboring nodes for each node.
	//
	std::vector<IndexType> neighbor_count(gm.numberOfVariables(), 0);
	for (IndexType i = 0; i < gm.numberOfFactors(); ++i) {
		if (gm[i].numberOfVariables() >= 2) {
			for (IndexType j = 0; j < gm[i].numberOfVariables(); ++j)
				++neighbor_count[gm[i].variableIndex(j)];
		}
	}

	//
	// Redistribute weights onto edges.
	//
	for (IndexType i = 0; i < gm.numberOfFactors(); ++i) {
		if (gm[i].numberOfVariables() >= 2) {
			for (int j = 0; j < gm[i].numberOfVariables(); ++j) {
				IndexType var = gm[i].variableIndex(j);
				auto &r = repa.get(i, j);
				for (IndexType k = 0; k < gm.numberOfLabels(var); ++k)
					r[k] += repa.getVariableValue(var, k) / neighbor_count[var];
				--neighbor_count[var];
			}
		}
	}

	//
	// Check that it worked.
	//
#ifndef NDEBUG
	for (IndexType i = 0; i < gm.numberOfVariables(); ++i) {
		OPENGM_ASSERT_OP(neighbor_count[i], ==, 0)
		bool is_connected = false;
		for (IndexType j = 0; j < gm.numberOfFactors(i); ++j)
			if (gm[gm.factorOfVariable(i, j)].numberOfVariables() > 1)
				is_connected = true;
		if (is_connected)
			for (LabelType j = 0; j < gm.numberOfLabels(i); ++j)
				OPENGM_ASSERT_OP(std::abs(repa.getVariableValue(i, j)), <=, 1e-6);
	}
#endif
}

//
// Computes the set of strictly arc consistent nodes by inspecting the
// potential values. This function only looks at factors of order >= 2 and
// unary values are (almost) completely ignored.
//
// IMPORTANT: This function assumes that pairwiseReparametrization(...) was on
// the reparametrization passed to it.
//
template<class ACC, class REPA>
void
computeStrictArcConsistency
(
	REPA &repa,
	std::vector<typename REPA::GraphicalModelType::LabelType> &labeling,
	std::vector<bool> &mask
)
{

	typedef typename REPA::GraphicalModelType GraphicalModelType;
	typedef typename GraphicalModelType::IndexType IndexType;
	typedef typename GraphicalModelType::LabelType LabelType;
	typedef typename GraphicalModelType::ValueType ValueType;
	typedef typename GraphicalModelType::FactorType FactorType;

	const GraphicalModelType &gm = repa.graphicalModel();

	labeling.assign(gm.numberOfVariables(), 0);
	mask.assign(gm.numberOfVariables(), false);

	struct Consistency {
		bool is_first;
		bool is_sac;
		LabelType label;

		Consistency() : is_first(true), is_sac(false), label(0) { }

		void check(LabelType l) {
			if (is_first) {
				is_first = false;
				is_sac = true;
				label = l;
			} else if (is_sac) {
				is_sac = label == l;
			}
		}
	};

#ifdef OPENGM_COMBILP_DEBUG
	int count_similar = 0;
#endif

	std::vector<Consistency> consistencies(gm.numberOfVariables());
	for (IndexType i = 0; i < gm.numberOfFactors(); ++i) {
		if (gm[i].numberOfVariables() > 1) {
			ValueType minimum = ACC::template neutral<ValueType>();
			FastSequence<LabelType> ll(gm[i].numberOfVariables());

			typedef ShapeWalker<typename FactorType::ShapeIteratorType> ShapeWalkerType;
			{
				ShapeWalkerType walker(gm[i].shapeBegin(), gm[i].numberOfVariables());
				for (size_t j = 0; j < gm[i].size(); ++j, ++walker) {
					ValueType value = repa.getFactorValue(i, walker.coordinateTuple().begin());
					if (ACC::bop(value, minimum)) {
						minimum = value;
						ll = walker.coordinateTuple();
					}
				}
			}

			for (IndexType j = 0; j < gm[i].numberOfVariables(); ++j) {
				IndexType var = gm[i].variableIndex(j);
				consistencies[var].check(ll[j]);
			}

#ifdef OPENGM_COMBILP_DEBUG
			{
				ShapeWalkerType walker(gm[i].shapeBegin(), gm[i].numberOfVariables());
				for (size_t j = 0; j < gm[i].size(); ++j, ++walker) {
					ValueType value = repa.getFactorValue(i, walker.coordinateTuple().begin());
					if (std::abs(value - minimum) <= 1e-8)
						++count_similar;
				}
				--count_similar;
			}
#endif
		}
	}

	for (IndexType var = 0; var < gm.numberOfVariables(); ++var) {
		mask[var] = consistencies[var].is_sac;
		labeling[var] = consistencies[var].label;
	}

#ifdef OPENGM_COMBILP_DEBUG
	std::cout << "SAC computation detected " << count_similar << " highly similar values." << std::endl;
#endif
}

//
// Dense CombiLP parametrisation
//
// The potentials on the boundary will be pushed from the unary term to the
// pairwise term and later pushed from the pairwise term to the unary term
// inside the ILP subproblem. This will preserve informations from the
// outside problem during the ILP solving phase.
//
// If a unary node is coupled with more than one pairwise factor to the ILP
// subproblem, the unary potential gets equally divided among all those
// pairwise factors.
//

template<class ACC, class REPA>
void
dense_reparametrization
(
	REPA &repa,
	const std::vector<bool> &mask
)
{
	typedef typename REPA::GraphicalModelType GraphicalModelType;
	typedef typename GraphicalModelType::IndexType IndexType;
	typedef typename GraphicalModelType::LabelType LabelType;

	const GraphicalModelType &gm = repa.graphicalModel();
	std::vector<IndexType> borderFactorCounter;
	std::vector<IndexType> borderFactors;

	computeBoundaryFactors(gm, mask, &borderFactors, &borderFactorCounter);
	for (auto it = borderFactors.begin(); it != borderFactors.end(); ++it) {
		for (IndexType lvar = 0; lvar < gm[*it].numberOfVariables(); ++lvar) {
			IndexType var = gm[*it].variableIndex(lvar);

			if (mask[var])
				continue;

			typename REPA::UnaryFactor &r = repa.get(*it, lvar);
			for (LabelType l = 0; l < gm.numberOfLabels(var); ++l) {
				OPENGM_ASSERT(borderFactorCounter[var] != 0);
				r[l] += repa.getVariableValue(var, l) / borderFactorCounter[var];
			}
			--borderFactorCounter[var];
		}
	}
}

//
// Mask helper functions
//

template<class GM>
std::vector<bool>
mask_depth_first
(
	const GM &gm,
	const std::vector<bool> &mask,
	int iterations
)
{
	std::vector<bool> mask_new = mask;
	for (int i = 0; i < iterations; ++i) {
		std::vector<bool> mask_old = mask_new;

		for (typename GM::IndexType f = 0; f < gm.numberOfFactors(); ++f) {
			if (gm[f].numberOfVariables() != 2)
				continue;

			if (mask_old[gm[f].variableIndex(0)])
				mask_new[gm[f].variableIndex(1)] = true;
			if (mask_old[gm[f].variableIndex(1)])
				mask_new[gm[f].variableIndex(0)] = true;
		}
	}

	return mask_new;
}

std::vector<bool>
mask_minus(
	const std::vector<bool> &a,
	const std::vector<bool> &b
)
{
	OPENGM_ASSERT_OP(a.size(), ==, b.size());
	std::vector<bool> result(a.size());
	for(size_t i = 0; i < a.size(); ++i)
		result[i] = a[i] && (! b[i]);
	return result;
}

std::vector<bool>
mask_intersect(
	const std::vector<bool> &a,
	const std::vector<bool> &b
)
{
	OPENGM_ASSERT_OP(a.size(), ==, b.size());
	std::vector<bool> result(a.size());
	for (size_t i = 0; i < a.size(); ++i)
		result[i] = a[i] && b[i];
	return result;
}


//
// Functions used for debugging and calculating debug metrics.
//

template<class REPA>
typename REPA::GraphicalModelType::ValueType
calculate_unary_sum
(
	const REPA &repa,
	const std::vector<bool> &mask
)
{
	typename REPA::GraphicalModelType::ValueType sum = 0;

	for (typename REPA::GraphicalModelType::IndexType i = 0; i < repa.graphicalModel().numberOfVariables(); ++i) {
		if (!mask[i])
			continue;

		typename REPA::GraphicalModelType::ValueType minimum = repa.getVariableValue(i, 0);
		for (typename REPA::GraphicalModelType::LabelType j = 0; j <repa.graphicalModel().numberOfLabels(i); ++j) {
			typename REPA::GraphicalModelType::ValueType v = repa.getVariableValue(i, j);
			sum += v;
			minimum = std::min(minimum, v);
		}

		// Normalization
		sum -= minimum * repa.graphicalModel().numberOfLabels(i);
	}

	return sum;
}

template<class REPA>
std::vector<typename REPA::GraphicalModelType::ValueType>
calculate_difference_of_best_unary_labels
(
	const REPA &repa,
	const std::vector<bool> &mask
)
{
	std::vector<typename REPA::GraphicalModelType::ValueType> result;
	for (typename REPA::GraphicalModelType::IndexType i = 0; i < repa.graphicalModel().numberOfVariables(); ++i) {
		if (!mask[i])
			continue;

		std::vector<typename REPA::GraphicalModelType::ValueType> best_values;
		best_values.push_back(repa.getVariableValue(i, 0));

		for (typename REPA::GraphicalModelType::LabelType j = 1; j <repa.graphicalModel().numberOfLabels(i); ++j) {
			typename REPA::GraphicalModelType::ValueType v = repa.getVariableValue(i, j);

			if (v < best_values[0])
				best_values.insert(best_values.begin(), v);
			else if (v < best_values[1])
				best_values.insert(best_values.begin() + 1, v);

			best_values.resize(2);
		}

		result.push_back(best_values[1] - best_values[0]);
	}
	return result;
}

template<class T>
T median(std::vector<T> vec)
{
	std::sort(vec.begin(), vec.end());
	size_t s = vec.size();
	if (s % 2 == 0)
		return (vec[s/2 - 1] + vec[s/2]) / static_cast<T>(2);
	else
		return vec[s/2];
}

template<class REPA>
void
print_debug_information
(
	const REPA &repa,
	const std::vector<bool> &mask_ilp,
	const std::vector<bool> &mask_delta_a,
	const std::vector<bool> &mask_a_prime
)
{
	std::streamsize old_precision = std::cout.precision();
	std::cout << "sum of normalized unaries in B  = " << calculate_unary_sum(repa, mask_ilp) << std::endl;
	std::cout << "sum of normalized unaries in A' = " << calculate_unary_sum(repa, mask_a_prime) << std::endl;
	std::cout << "sum of normalized unaries in δA = " << calculate_unary_sum(repa, mask_delta_a) << std::endl;

	auto diff_two_best = calculate_difference_of_best_unary_labels(repa, mask_delta_a);
	OPENGM_ASSERT_OP(diff_two_best.size(), ==, std::count(mask_delta_a.begin(), mask_delta_a.end(), true));
	std::cout << "median(diff(2-best)) in δA = " << median(diff_two_best) << std::endl;

	size_t num_of_zero = std::count_if(diff_two_best.begin(), diff_two_best.end(),
		[](const typename REPA::GraphicalModelType::ValueType &v) { return std::abs(v) <= 1e-6; });
	std::cout << "num of 0 in diff(2-best) in δA = " << num_of_zero << " / " << diff_two_best.size() << " (" << std::setprecision(5) << (100.0d * num_of_zero / diff_two_best.size()) << "%)" << std::endl;
	std::cout.precision(old_precision);
}

//
// Stripe LP redistribution logic.
//

#ifdef WITH_CPLEX
template<class REPA>
void
redistribute_by_stripe_lp
(
	REPA &repa,
	const std::vector<bool> &mask_delta_a,
	const std::vector<bool> &mask_a_prime
)
{
	typedef typename REPA::GraphicalModelType GraphicalModelType;
	typedef typename GraphicalModelType::IndexType IndexType;
	typedef typename GraphicalModelType::LabelType LabelType;
	typedef typename GraphicalModelType::ValueType ValueType;
	typedef typename GraphicalModelType::FactorType FactorType;

	const GraphicalModelType &gm = repa.graphicalModel();

	std::cout << "Gathering unaries and pairwise terms" << std::endl;
	std::vector<bool> unary_found(gm.numberOfVariables(), false);
	std::vector<IndexType> unary(gm.numberOfVariables());
	std::vector<IndexType> pairwise;
	std::vector<std::vector<std::pair<IndexType, IndexType>>> neighbors(gm.numberOfVariables());
	for (IndexType f = 0; f < gm.numberOfFactors(); ++f) {
		OPENGM_ASSERT_OP(gm[f].numberOfVariables(), >=, 1);
		OPENGM_ASSERT_OP(gm[f].numberOfVariables(), <=, 2);

		if (gm[f].numberOfVariables() == 1) {
			OPENGM_ASSERT(!unary_found[gm[f].variableIndex(0)]);
			unary[gm[f].variableIndex(0)] = f;
			unary_found[gm[f].variableIndex(0)] = true;
		} else {
			IndexType idx = pairwise.size();
			pairwise.push_back(f);
			neighbors[gm[f].variableIndex(0)].push_back(std::make_pair(gm[f].variableIndex(1), idx));
			neighbors[gm[f].variableIndex(1)].push_back(std::make_pair(gm[f].variableIndex(0), idx));
		}
	}
	OPENGM_ASSERT_OP(std::count(unary_found.begin(), unary_found.end(), false), ==, 0);

	OPENGM_ASSERT_OP(unary.size(), ==, gm.numberOfVariables());
	OPENGM_ASSERT_OP(unary.size() + pairwise.size(), ==, gm.numberOfFactors());

	std::cout << "Adding variables" << std::endl;
	IndexType phi_counter = 0;
	std::vector<IndexType> phi(pairwise.size());
	for (IndexType i = 0; i < pairwise.size(); ++i) {
		if (mask_a_prime[gm[pairwise[i]].variableIndex(0)] && mask_a_prime[gm[pairwise[i]].variableIndex(1)]) {
			phi[i] = phi_counter;
			phi_counter += gm[pairwise[i]].numberOfLabels(0) + gm[pairwise[i]].numberOfLabels(1);
		}
	}

	// IMPORTANT NOTE: Everything which needs to index the “x” object is pretty
	// slow. It is orders of magnitudes faster to put things first into an
	// IloNumArray and then push it into the model.
	IloEnv env;
	IloRangeArray c(env);
	IloObjective obj(env, 0, IloObjective::Maximize);
	IloNumVarArray x(env, phi_counter, -IloInfinity, IloInfinity);
	IloNumArray obj_coeffs(env, phi_counter);

	std::cout << "Objective" << std::endl;
	for (IndexType i = 0; i < pairwise.size(); ++i) {
		if (!mask_a_prime[gm[pairwise[i]].variableIndex(0)] || !mask_a_prime[gm[pairwise[i]].variableIndex(1)])
			continue;

		if (mask_delta_a[gm[pairwise[i]].variableIndex(0)])
			for (LabelType j = 0; j < gm[unary[gm[pairwise[i]].variableIndex(0)]].size(); ++j)
				obj_coeffs[phi[i] + j] = 1;

		if (mask_delta_a[gm[pairwise[i]].variableIndex(1)])
			for (LabelType j = 0; j < gm[unary[gm[pairwise[i]].variableIndex(1)]].size(); ++j)
				obj_coeffs[phi[i] + gm[pairwise[i]].numberOfLabels(0) + j] = 1;
	}
	obj.setLinearCoefs(x, obj_coeffs);
	
	std::cout << "Unary constraints" << std::endl;
	for (IndexType u = 0; u < unary.size(); ++u) {
		if (!mask_a_prime[u])
			continue;

		std::vector<size_t> factors;
		for (const auto &it : neighbors[u]) {
			if (mask_a_prime[it.first])
				factors.push_back(it.second);
		}

		ValueType minimum = repa.getVariableValue(unary[u], 0);
		std::vector<ValueType> rhs(gm[unary[u]].size());
		for (LabelType i = 0; i < gm[unary[u]].size(); ++i) {
			rhs[i] = repa.getVariableValue(unary[u], i);
			minimum = std::min(minimum, rhs[i]);
		}
		std::transform(rhs.begin(), rhs.end(), rhs.begin(), std::bind2nd(std::minus<ValueType>(), minimum));

		for (LabelType i = 0; i < gm[unary[u]].size(); ++i) {
			IloRange r(env, -rhs[i], IloInfinity);
			for (const auto &pw : factors) {
				if (gm[pairwise[pw]].variableIndex(0) == u)
					r.setLinearCoef(x[phi[pw] + i], 1);
				else
					r.setLinearCoef(x[phi[pw] + gm[pairwise[pw]].numberOfLabels(0) + i], 1);
			}
			c.add(r);
		}
	}

	std::cout << "Pairwise constraints" << std::endl;
	for (IndexType i = 0; i < pairwise.size(); ++i) {
		if (!mask_a_prime[gm[pairwise[i]].variableIndex(0)] || !mask_a_prime[gm[pairwise[i]].variableIndex(1)])
			continue;
	
		ShapeWalker<typename FactorType::ShapeIteratorType> walker(gm[pairwise[i]].shapeBegin(), gm[pairwise[i]].numberOfVariables());
		for (IndexType j = 0; j < gm[pairwise[i]].size(); ++j, ++walker) {
			IloRange r(env, -repa.getFactorValue(pairwise[i], walker.coordinateTuple().begin()), IloInfinity);
			r.setLinearCoef(x[phi[i] + walker.coordinateTuple()[0]], -1);
			r.setLinearCoef(x[phi[i] + gm[pairwise[i]].numberOfLabels(0) + walker.coordinateTuple()[1]], -1);
			c.add(r);
		}
	}

	std::cout << "Solving" << std::endl;
	IloModel model(env);
	model.add(x);
	model.add(obj);
	model.add(c);
	IloCplex cplex(model);
	cplex.setParam(IloCplex::TiLim, 60);
	cplex.setParam(IloCplex::RootAlg, IloCplex::Primal);
	cplex.setParam(IloCplex::PreInd, false);

	IloNumArray starting_point(env, phi_counter);
	for (IndexType i = 0; i < phi_counter; ++i)
		starting_point[i] = 0;
	cplex.setStart(starting_point, 0, x, 0, 0, 0);
	
	bool optimal = cplex.solve();
	if (!optimal && !cplex.isPrimalFeasible()) {
		std::cout << "No feasible solution after timelimit, applying hack..." << std::endl;
		cplex.setParam(IloCplex::TiLim, 1e+75);
		cplex.setParam(IloCplex::ObjULim, 0); // Can’t use getObjValue(), it is also wrong
		optimal = cplex.solve();
	}

	std::cout << "cplex.getStatus() = " << cplex.getStatus() << std::endl;
	std::cout << "cplex.getCplexStatus() = " << cplex.getCplexStatus() << std::endl;
	std::cout << "cplex.getObjValue() = " << cplex.getObjValue() << std::endl;

	if (!optimal) {
		std::cout << "CPLEX did not find optimal solution for A’ LP" << std::endl;
		if (cplex.isPrimalFeasible()) {
			std::cout << "CPLEX found primal feasible solution" << std::endl;
		} else {
			const char *s = "CPLEX did not find *any* feasible solution for A’ LP";
			std::cout << s << std::endl;
			throw std::runtime_error(s);
		}
	}

	IloNumArray sol(env);
	cplex.getValues(x, sol);

	IndexType curphi = 0;
	for (IndexType fidx = 0; fidx < gm.numberOfFactors(); ++fidx) {
		if (gm[fidx].numberOfVariables() < 2)
			continue;

		if (!mask_a_prime[gm[fidx].variableIndex(0)] || !mask_a_prime[gm[fidx].variableIndex(1)]) {
			++curphi;
			continue;
		}

		std::pair<ValueType*, ValueType*> its;
		its = repa.getIterators(fidx, 0);
		for (LabelType i = 0; i < gm[fidx].numberOfLabels(0); ++i)
			*(its.first + i) -= sol[phi[curphi] + i];

		its = repa.getIterators(fidx, 1);
		for (LabelType i = 0; i < gm[fidx].numberOfLabels(1); ++i)
			*(its.first + i) -= sol[phi[curphi] + gm[fidx].numberOfLabels(0) + i];

		++curphi;
	}

	env.end();
}
#endif

//
// Tree-based potential redistribution logic.
//

template<class GM>
void
calculate_masks
(
	const GM &gm,
	const std::vector<bool> &mask_ilp,
	/* int radius, */
	std::vector<bool> *mask_delta_a,
	std::vector<bool> *mask_a_prime
)
{
	OPENGM_ASSERT_OP(mask_ilp.size(), ==, gm.numberOfVariables());
	mask_delta_a->resize(mask_ilp.size());
	mask_a_prime->resize(mask_ilp.size());

	OPENGM_ASSERT_OP(std::count(mask_ilp.begin(), mask_ilp.end(), true), >, 0);

#ifdef OPENGM_COMBILP_DEBUG
	std::cout << "Calculating masks" << std::endl;
#endif
	std::vector<bool> tmp = mask_depth_first(gm, mask_ilp, 1);
	*mask_delta_a = mask_minus(tmp, mask_ilp);

	// We could restrict the region to a specific radius around the ILP
	// subproblem, but currently we just use the whole remaining graph.
	tmp = mask_depth_first(gm, *mask_delta_a, 20);
	*mask_a_prime = mask_minus(tmp, mask_ilp);
	//std::transform(mask_ilp.begin(), mask_ilp.end(), mask_a_prime->begin(), std::logical_not<bool>());

	OPENGM_ASSERT_OP(
		std::count(mask_a_prime->begin(), mask_a_prime->end(), true),
		>=,
		std::count(mask_delta_a->begin(), mask_delta_a->end(), true));

#ifdef OPENGM_COMBILP_DEBUG
	std::cout << "=== MASK SIZES ===" << std::endl;
	std::cout << "total =" << mask_ilp.size() << std::endl;
	std::cout << "mask_ilp = " << std::count(mask_ilp.begin(), mask_ilp.end(), true) << std::endl;
	std::cout << "mask_delta_a = " << std::count(mask_delta_a->begin(), mask_delta_a->end(), true) << std::endl;
	std::cout << "mask_a_prime = " << std::count(mask_a_prime->begin(), mask_a_prime->end(), true) << std::endl;
#endif
}

template<class GM>
struct EdgeInfo {
	typename GM::IndexType factor;
	bool direction_forward;
};

template<class ACC, class REPA>
void
redistribute_potentials_on_mask
(
	REPA &repa,
	const std::vector<bool> &mask_delta_a,
	const std::vector<bool> &mask_a_prime
)
{
	typedef typename REPA::GraphicalModelType GraphicalModelType;
	typedef typename GraphicalModelType::IndexType IndexType;
	typedef typename GraphicalModelType::LabelType LabelType;
	typedef typename GraphicalModelType::ValueType ValueType;

	const GraphicalModelType &gm = repa.graphicalModel();

	typedef EdgeInfo<GraphicalModelType> EdgeInfoType;
	// number trees crossing this variable
	std::vector<IndexType> variable_tree_count(gm.numberOfVariables());
	// pairwise factors only belong to one tree (only pw factors considered)
	std::vector<bool> factor_enabled(gm.numberOfFactors());
	// stack to remember edge pushing operations
	std::stack<EdgeInfoType, std::vector<EdgeInfoType>> edge_stack;
	// remember positions of tree endpoints (nodes) for each node in δA
	std::vector<std::vector<IndexType>> delta_a;

	// Disable all factors which are neither pairwise factors nor lie in the
	// area of A’.
	for (IndexType i = 0; i < gm.numberOfFactors(); ++i) {
		factor_enabled[i] = true;

		IndexType nV = gm[i].numberOfVariables();
		// FIXME: Allow constant factors.
		if (nV < 1 || nV > 2) {
			throw std::runtime_error("Only second-order models are supported.");
		}

		if (nV != 2)
			factor_enabled[i] = false;

		IndexType var0 = gm[i].variableIndex(0);
		IndexType var1 = gm[i].variableIndex(1);

		if (!mask_a_prime[var0] || !mask_a_prime[var1])
			factor_enabled[i] = false;

		if (mask_delta_a[var0] && mask_delta_a[var1])
			factor_enabled[i] = false;
	}

	// Create start configuration by finding all nodes in δA.
	// We create a new tree for each node in δA
	for (IndexType i = 0; i < gm.numberOfVariables(); ++i) {
		if (mask_delta_a[i]) {
			delta_a.emplace_back();
			delta_a.back().push_back(i);
		}
	}
#ifdef OPENGM_COMBILP_DEBUG
	std::cout << "Will build " << delta_a.size() << " independent trees." << std::endl;
#endif

	// Discover all the trees by doing a breadth-first-search.
	size_t nodes_in_working_set;
	do {
#ifdef OPENGM_COMBILP_DEBUG
		std::cout << "Doing another extension pass." << std::endl;
#endif
		IndexType node_count = 0;
		for (auto &current_delta_a : delta_a) {
			std::vector<IndexType> working_set = current_delta_a;
			current_delta_a.clear();

			for (const auto &endpoint : working_set) {
				for (IndexType i = 0; i < gm.numberOfFactors(endpoint); ++i) {
					IndexType fidx = gm.factorOfVariable(endpoint, i);
					if (factor_enabled[fidx]) {
						factor_enabled[fidx] = false;

						EdgeInfoType edge;
						edge.factor = fidx;
						edge.direction_forward = gm[fidx].variableIndex(1) == endpoint;
						edge_stack.push(edge);

						IndexType neighbor = gm[fidx].variableIndex(edge.direction_forward ? 0 : 1);
						++variable_tree_count[neighbor];
						current_delta_a.push_back(neighbor);
					}
				}
			}
		}

		nodes_in_working_set = std::accumulate(delta_a.begin(), delta_a.end(), 0,
			[](int a, std::vector<IndexType> b) { return a + b.size(); });
#ifdef OPENGM_COMBILP_DEBUG
		std::cout << "There are " << nodes_in_working_set << " nodes in the current working set." << std::endl;
#endif
	} while(nodes_in_working_set != 0);

#ifdef OPENGM_COMBILP_DEBUG
	std::cout << "Tree discovery done. I found " << edge_stack.size() << " edges for message passing." << std::endl;

	std::cout << "variable_tree_count: " << variable_tree_count;
#endif

#if !defined(NDBEUG) || defined(OPENGM_DEBUG)
	for (IndexType i = 0; i < gm.numberOfVariables(); ++i)
		if (mask_a_prime[i] && !mask_delta_a[i])
			OPENGM_ASSERT_OP(variable_tree_count[i], >=, 1);
	std::cout << "All unaries in A'\\δA are part of at least one tree." << std::endl;
#endif

	// Message passing phase. Just edge from stack and push message in the right
	// direction. Unary potentials have to be considered specially, because the
	// potential is divided among all trees.
	while (!edge_stack.empty()) {
		const EdgeInfoType &e = edge_stack.top();
		IndexType var_from_lid = e.direction_forward ? 0 : 1;
		IndexType var_to_lid   = e.direction_forward ? 1 : 0;
		IndexType var_from_id  = gm[e.factor].variableIndex(var_from_lid);
		IndexType var_to_id    = gm[e.factor].variableIndex(var_to_lid);

		OPENGM_ASSERT_OP(variable_tree_count[var_from_id], >, 0);

		// Push from unary into pencil.
		auto it = repa.getIterators(e.factor, var_from_lid).first;
		for (LabelType i = 0; i < gm[e.factor].numberOfLabels(var_from_lid); ++i, ++it) {
			*it += repa.getVariableValue(var_from_id, i) / variable_tree_count[var_from_id];
		}
		--variable_tree_count[var_from_id];

#if !defined(NDEBUG) || defined(OPENGM_DEBUG)
		// Check that unaries are zero if no trees shares this node anymore.
		if (variable_tree_count[var_from_id] == 0) {
			for (LabelType i = 0; i < gm[e.factor].numberOfLabels(var_from_lid); ++i) {
				OPENGM_ASSERT_OP(repa.getVariableValue(var_from_id, i), >=, -1e-6);
			}
		}
#endif

		// Push minimal value into unary potential of opposite side.
		it = repa.getIterators(e.factor, var_to_lid).first;
		for (LabelType i = 0; i < gm[e.factor].numberOfLabels(var_to_lid); ++i, ++it) {
			// Calculate minimum element of pencil.
			ValueType minimum = ACC::template neutral<ValueType>();
			for (LabelType j = 0; j < gm[e.factor].numberOfLabels(var_from_lid); ++j) {
				FastSequence<LabelType> labeling(2);
				labeling[var_to_lid]   = i;
				labeling[var_from_lid] = j;
				ACC::op(repa.getFactorValue(e.factor, labeling.begin()), minimum);
			}

			// Push message
			*it -= minimum;
		}

#if !defined(NDEBUG) || defined(OPENGM_DEBUG)
		// Check that at least one pairwise potential in pencil is zero.
		it = repa.getIterators(e.factor, var_to_lid).first;
		for (LabelType i = 0; i < gm[e.factor].numberOfLabels(var_to_lid); ++i, ++it) {
			bool is_zero = false;
			for (LabelType j = 0; j < gm[e.factor].numberOfLabels(var_from_lid); ++j) {
				FastSequence<LabelType> labeling(2);
				labeling[var_to_lid]   = i;
				labeling[var_from_lid] = j;
				if (std::abs(repa.getFactorValue(e.factor, labeling.begin())) <= 1e-6) {
					is_zero = true;
					break;
				}
			}
			OPENGM_ASSERT(is_zero);
		}
#endif

		edge_stack.pop();
	}

#ifdef OPENGM_COMBILP_DEBUG
	std::cout << "Message passing phase finished." << std::endl;
#endif

#if !defined(NDEBUG) || defined(OPENGM_DEBUG)
	for (IndexType i = 0; i < gm.numberOfVariables(); ++i) {
		OPENGM_ASSERT_OP(variable_tree_count[i], ==, 0);

		if (!mask_a_prime[i] || mask_delta_a[i])
			continue;

		for (LabelType j = 0; j < gm.numberOfLabels(i); ++j)
			OPENGM_ASSERT_OP(repa.getVariableValue(i, j), >=, -1e-6);

		for (IndexType j = 0; j < gm.numberOfFactors(i); ++j) {
			IndexType fidx = gm.factorOfVariable(i, j);

			OPENGM_ASSERT_OP(gm[fidx].numberOfVariables(), >=, 1);
			OPENGM_ASSERT_OP(gm[fidx].numberOfVariables(), <=, 2);

			if (gm[fidx].numberOfVariables() == 2) {
				FastSequence<LabelType> labeling(2);
				for (labeling[0] = 0; labeling[0] < gm[fidx].numberOfLabels(0); ++labeling[0])
					for (labeling[1] = 0; labeling[1] < gm[fidx].numberOfLabels(1); ++labeling[1])
						OPENGM_ASSERT_OP(repa.getFactorValue(fidx, labeling.begin()), >=, -1e-6);
			}
		}
	}
#endif
}

template<class ACC, class REPA>
void
redistribute_potentials
(
	REPA &repa,
	const bool denseVersionEnabled,
	const std::vector<bool> &mask_ilp
)
{
#ifdef OPENGM_COMBILP_DEBUG
	std::cout << "Using tree_decomposition for optimizing nodes in δA." << std::endl;
#endif

	if (repa.graphicalModel().numberOfVariables() == std::count(mask_ilp.begin(), mask_ilp.end(), true)) {
#ifdef OPENGM_COMBILP_DEBUG
		std::cout << "Nothing to do, ILP subproblem already as large as original problem." << std::endl;
#endif
		return;
	}

	// The redistribution of potentials was developed for the dense version for
	// CombiLP. To make it also work for the standard version, we have to apply
	// this quickfix here, because the mask also contains the boundary nodes
	// (yellow nodes).
	std::vector<bool> tmp_mask = mask_ilp;
	if (!denseVersionEnabled){
#ifdef OPENGM_COMBILP_DEBUG
		std::cout << "Applying quickfix for standard version of CombiLP." << std::endl;
#endif
		std::transform(mask_ilp.begin(), mask_ilp.end(), tmp_mask.begin(), std::logical_not<bool>());
		tmp_mask = mask_depth_first(repa.graphicalModel(), tmp_mask, 1);
		tmp_mask = mask_minus(mask_ilp, tmp_mask);
	}

	std::vector<bool> mask_delta_a, mask_a_prime;
	calculate_masks(repa.graphicalModel(), tmp_mask, &mask_delta_a, &mask_a_prime);

#ifdef OPENGM_COMBILP_DEBUG
	std::cout << "================================================================================" << std::endl;
	std::cout << "REPA DEBUG INFO (BEFORE)" << std::endl;
	print_debug_information(repa, tmp_mask, mask_delta_a, mask_a_prime);
	std::cout << "================================================================================" << std::endl;
#endif

	redistribute_potentials_on_mask<ACC>(repa, mask_delta_a, mask_a_prime);
	//redistribute_by_stripe_lp(repa, mask_delta_a, mask_a_prime);

#ifdef OPENGM_COMBILP_DEBUG
	std::cout << "================================================================================" << std::endl;
	std::cout << "REPA DEBUG INFO (AFTER)" << std::endl;
	print_debug_information(repa, tmp_mask, mask_delta_a, mask_a_prime);
	std::cout << "================================================================================" << std::endl;
#endif
}

} // namespace combilp
} // namespace opengm

#endif
