//
// File: utils.hxx
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
#ifndef OPENGM_COMBILP_UTILS_HXX
#define OPENGM_COMBILP_UTILS_HXX

#include <algorithm>
#include <vector>

#include <opengm/graphicalmodel/graphicalmodel.hxx>
#include <opengm/inference/auxiliary/lp_reparametrization.hxx>

namespace opengm {
namespace combilp {

// Add all neighboring nodes to the mask.
template<class GM>
void
dilateMask
(
	const GM &gm,
	typename GM::IndexType varId,
	std::vector<bool> &mask
)
{
	typedef typename GM::IndexType IndexType;
	typedef typename GM::FactorType FactorType;
	OPENGM_ASSERT_OP(varId, <, gm.numberOfVariables());
	OPENGM_ASSERT_OP(mask.size(), ==, gm.numberOfVariables());

	typename GM::IndexType numberOfFactors = gm.numberOfFactors(varId);
	// Look for all factors which contain node varId. Add all these
	// neighbors to the mask.
	for (IndexType i = 0; i < gm.numberOfFactors(varId); ++i) {
		const FactorType &f = gm[ gm.factorOfVariable(varId, i) ];

		// Set all mask for all variables in factor to true.
		for (IndexType j = 0; j < f.numberOfVariables(); ++j)
			mask[ f.variableIndex(j) ] = true;
	}
}

// Add all neighboring nodes for any already activated node to the mask.
template<class GM>
void
dilateMask
(
	const GM &gm,
	std::vector<bool> &mask
)
{
	OPENGM_ASSERT_OP(mask.size(), ==, gm.numberOfVariables());
	std::vector<bool> input = mask;
	for (typename GM::IndexType i = 0; i < gm.numberOfVariables(); ++i)
		if (input[i])
			dilateMask(gm, i, mask);
}

template<class GM>
void
computeBoundaryMask
(
	const GM &gm,
	const std::vector<bool> &sac_mask,
	std::vector<bool> &boundaryMask,
	bool from_sac_side = false
)
{
	typedef typename GM::IndexType IndexType;
	typedef typename GM::FactorType FactorType;
	OPENGM_ASSERT_OP(sac_mask.size(), ==, gm.numberOfVariables());

	boundaryMask.assign(sac_mask.size(), false);
	for (IndexType i = 0; i < sac_mask.size(); ++i) {
		if (sac_mask[i] == from_sac_side) continue;

		for (IndexType j = 0; j < gm.numberOfFactors(i); ++j) {
			if (boundaryMask[i]) break;

			typedef typename FactorType::VariablesIteratorType Iter;
			const FactorType& f = gm[ gm.factorOfVariable(i, j) ];
			for (Iter it = f.variableIndicesBegin(); it != f.variableIndicesEnd(); ++it) {
				if (sac_mask[*it] == from_sac_side) {
					boundaryMask[i] = true;
					break;
				}
			}
		}
	}
}

template<class GM>
void
computeBoundaryFactors
(
	const GM &gm,
	const std::vector<bool> &sac_mask,
	std::vector<typename GM::IndexType> *boundaryFactors,
	std::vector<typename GM::IndexType> *boundaryFactorCounter
)
{
	typedef typename GM::IndexType IndexType;
	typedef typename GM::FactorType FactorType;
	
	if (boundaryFactors)
		boundaryFactors->resize(0);
	if (boundaryFactorCounter)
		boundaryFactorCounter->assign(gm.numberOfVariables(), 0);

	for (IndexType fidx = 0; fidx < gm.numberOfFactors(); ++fidx) {
		const FactorType &factor = gm[fidx];

		bool first_side = false;
		bool second_side = false;
		for (IndexType lvar = 0; lvar < factor.numberOfVariables(); ++lvar) {
			IndexType var = factor.variableIndex(lvar);
			if (sac_mask[var])
				first_side = true;
			if (!sac_mask[var])
				second_side = true;
		}
		if (!first_side || !second_side)
			continue;

		if (boundaryFactors)
			boundaryFactors->push_back(fidx);

		if (boundaryFactorCounter)
			for (IndexType lvar = 0; lvar < factor.numberOfVariables(); ++lvar)
				++(*boundaryFactorCounter)[factor.variableIndex(lvar)];
	}
}

// Easier label mismatching for NON-dense version of CombiLP.
template<class GM>
bool
mismatchingLabels
(
	const std::vector<typename GM::LabelType> &labeling_sac,
	const std::vector<typename GM::LabelType> &labeling_non_sac,
	const std::vector<bool> &mask_non_sac,
	std::vector<typename GM::IndexType> &mismatches
)
{
	OPENGM_ASSERT_OP(labeling_sac.size(), ==, mask_non_sac.size());
	OPENGM_ASSERT_OP(labeling_non_sac.size(), ==, mask_non_sac.size());
	mismatches.clear();
	for (typename GM::IndexType var = 0; var < mask_non_sac.size(); ++var)
		if (mask_non_sac[var] && (labeling_non_sac[var] != labeling_sac[var]))
			mismatches.push_back(var);
	return mismatches.empty();
}

// Label mismatching function used by dense version of CombiLP.
// Functions iterates over all border factors and checks that the corresponding
// potential is zero. (Assumption is that the function denseReparametrization
// was used before inference on the masked part.)
template<class ACC, class GM>
bool
mismatchingLabelsDense
(
	const GM &gm,
	const std::vector<typename GM::LabelType> &labeling,
	const std::vector<bool> &mask_non_sac,
	std::vector<typename GM::IndexType> &mismatches,
	typename GM::ValueType &gap,
	typename GM::ValueType &boundaryBound
)
{
	typedef typename GM::IndexType IndexType;
	typedef typename GM::LabelType LabelType;
	typedef typename GM::ValueType ValueType;

	OPENGM_ASSERT_OP(labeling.size(), ==, mask_non_sac.size());
	mismatches.clear();

	std::vector<typename GM::IndexType> boundaryFactors;
	computeBoundaryFactors<GM>(gm, mask_non_sac, &boundaryFactors, NULL);

	gap = 0;
	boundaryBound = 0;
	for (auto it = boundaryFactors.begin(); it != boundaryFactors.end(); ++it) {
		const typename GM::FactorType &factor = gm[*it];

		opengm::FastSequence<IndexType> fixedVars;
		opengm::FastSequence<LabelType> fixedLabs;
		for (IndexType i = 0; i < factor.numberOfVariables(); ++i) {
			if (mask_non_sac[factor.variableIndex(i)]) {
				fixedVars.push_back(i);
				fixedLabs.push_back(labeling[factor.variableIndex(i)]);
			}
		}

		typedef SubShapeWalker<typename GM::FactorType::ShapeIteratorType,
			opengm::FastSequence<IndexType>, opengm::FastSequence<LabelType>> WalkerType;
		WalkerType walker(factor.shapeBegin(), factor.numberOfVariables(), fixedVars, fixedLabs);
		ValueType minimum = ACC::template neutral<ValueType>();
		for (size_t size = 0; size < walker.subSize(); ++size, ++walker) {
			ACC::op(factor(walker.coordinateTuple().begin()), minimum);
		}
		boundaryBound += minimum;

		opengm::FastSequence<LabelType> flabeling(factor.numberOfVariables());
		for (IndexType i = 0; i < factor.numberOfVariables(); ++i)
			flabeling[i] = labeling[factor.variableIndex(i)];

		// TODO: Improve this line to get an optimal edge and be independent on
		// numerical issues.
		ValueType arc = factor(flabeling.begin());
		if (std::abs(arc - minimum) > 1e-20) {
			gap += std::abs(arc - minimum);

			// FIXME: We actually add too many and possibly duplicated
			// mismatches here. So the reported value of `mismatches.size()`
			// might be misleading.
			for (IndexType i = 0; i < factor.numberOfVariables(); ++i)
				mismatches.push_back(factor.variableIndex(i));
		}
	}
	return mismatches.empty();
}

template<class GM>
GraphicalModelManipulator<GM>
maskToManipulator
(
	const GM &gm,
	const std::vector<typename GM::LabelType> &labeling,
	const std::vector<bool> &mask,
	bool dense_version
)
{
	OPENGM_ASSERT_OP(gm.numberOfVariables(), ==, labeling.size());
	OPENGM_ASSERT_OP(gm.numberOfVariables(), ==, mask.size());
	GraphicalModelManipulator<GM> result(gm,
		dense_version ? GraphicalModelManipulator<GM>::BOUND
		              : GraphicalModelManipulator<GM>::DROP);

	for (typename GM::IndexType i = 0; i < gm.numberOfVariables(); ++i)
		if (! mask[i])
			result.fixVariable(i, labeling[i]);

	result.lock();
	result.buildModifiedSubModels();
	return result;
}

template<class ACC, class REPA>
void
checkStrictArcConsistency
(
	const REPA &repa,
	const std::vector<typename REPA::GraphicalModelType::LabelType> &labeling,
	const std::vector<bool> &mask_non_sac,
	const typename REPA::GraphicalModelType::ValueType tolerance = 1e-8
)
{
#if !defined(NDEBUG) || defined(OPENGM_DEBUG)
#ifdef OPENGM_COMBILP_DEBUG
	std::cout << "Checking strict arc-consistency property on reparemtrization..." << std::endl;
#endif
	typedef typename REPA::GraphicalModelType GraphicalModelType;
	typedef typename GraphicalModelType::IndexType IndexType;
	typedef typename GraphicalModelType::LabelType LabelType;
	typedef typename GraphicalModelType::ValueType ValueType;
	typedef typename GraphicalModelType::FactorType FactorType;

	const GraphicalModelType &gm = repa.graphicalModel();

	// Check unary values
	for (IndexType i = 0; i < gm.numberOfVariables(); ++i) {
		if (mask_non_sac[i])
			continue;

		for (LabelType j = 0; j < gm.numberOfLabels(i); ++j) {
			// TODO: Assumes minimization problem.
			OPENGM_ASSERT_OP(
				repa.getVariableValue(i, j),
				>=,
				repa.getVariableValue(i, labeling[i]) - tolerance
			);
		}
	}

	// Check pairwise values
	for (IndexType i = 0; i < gm.numberOfFactors(); ++i) {
		if (gm[i].numberOfVariables() == 1)
			continue;

		bool all_sac = true;
		for (IndexType j = 0; j < gm[i].numberOfVariables(); ++j)
			if (mask_non_sac[gm[i].variableIndex(j)])
				all_sac = false;
		if (! all_sac)
			continue;

		ShapeWalker<typename FactorType::ShapeIteratorType> shapeWalker(
			gm[i].shapeBegin(), gm[i].numberOfVariables());
		ValueType minimum = ACC::template neutral<ValueType>();
		for (size_t j = 0; j < gm[i].size(); ++j, ++ shapeWalker)
			ACC::op(repa.getFactorValue(i, shapeWalker.coordinateTuple().begin()), minimum);

		FastSequence<LabelType> l(gm.numberOfVariables());
		for (IndexType j = 0; j < gm[i].numberOfVariables(); ++j)
			l[j] = labeling[gm[i].variableIndex(j)];
		// TODO: Assumes minimization problem (because of the minus).
		OPENGM_ASSERT(ACC::bop(repa.getFactorValue(i, l.begin()) - tolerance, minimum));
	}
#endif
}

template<class GM>
void
checkCorrputionInGM
(
	const GM &gm
)
{
#if !defined(NDEBUG) || defined(OPENGM_DEBUG)
#ifdef OPENGM_COMBILP_DEBUG
	std::cout << "Checking for corruption..." << std::endl;
#endif
	typedef typename GM::IndexType IndexType;
	typedef typename GM::LabelType LabelType;
	typedef typename GM::ValueType ValueType;

	for (IndexType i = 0; i < gm.numberOfFactors(); ++i) {
		bool has_inf = false, has_nan = false;
		ShapeWalker<typename GM::FactorType::ShapeIteratorType> shapeWalker(
			gm[i].shapeBegin(), gm[i].numberOfVariables());
		for (size_t j = 0; j < gm[i].size(); ++j, ++shapeWalker) {
			ValueType v = gm[i](shapeWalker.coordinateTuple().begin());

			if (std::isinf(v))
				has_inf = true;

			if (std::isnan(v))
				has_nan = true;
		}

		if (has_inf)
			std::cout << "WARNING: Factor " << i << " contains ±inf values" << std::endl;

		if (has_nan)
			std::cout << "WARNING: Factor " << i << " contains ±nan values" << std::endl;
	}
#endif
}

template<class REPA>
void
checkCorruptionInRepa
(
	const REPA &repa
)
{
#if !defined(NDEBUG) || defined(OPENGM_DEBUG)
	LPReparametrizer<typename REPA::GraphicalModelType> r(repa.graphicalModel());
	r.Reparametrization() = repa;
	typename decltype(r)::ReparametrizedGMType rgm;
	r.getReparametrizedModel(rgm);
	checkCorrputionInGM(rgm);
#endif
}

template<class ITERATOR>
void
saveContainer
(
	const std::string &filename,
	ITERATOR begin,
	ITERATOR end
)
{
	std::ofstream file(filename);
	for (ITERATOR it = begin; it != end; ++it)
		file << *it << std::endl;
}

} // namespace combilp
} // namespace opengm

#endif
