#pragma once
#ifndef OPENGM_VIEW_BOUND_FUNCTION_HXX
#define OPENGM_VIEW_BOUND_FUNCTION_HXX

#include <iterator>

#include <opengm/operations/minimizer.hxx>
#include <opengm/functions/view_fix_variables_function.hxx>

namespace opengm {

template<class GM, class ACC = opengm::Minimizer>
class ViewBoundFunction : public ExplicitFunction<typename GM::ValueType, typename GM::IndexType, typename GM::LabelType> {
public:
	typedef ExplicitFunction<typename GM::ValueType, typename GM::IndexType, typename GM::LabelType> Parent;
	typedef GM GraphicalModelType;
	typedef ACC AccumulationType;
	typedef typename GraphicalModelType::ValueType ValueType;
	typedef typename GraphicalModelType::IndexType IndexType;
	typedef typename GraphicalModelType::LabelType LabelType;
	typedef typename GraphicalModelType::FactorType FactorType;

	template<class INPUT_IT>
	ViewBoundFunction(const FactorType &factor, INPUT_IT fixed_vars_begin, INPUT_IT fixed_vars_end);
};

template<class GM, class ACC>
template<class INPUT_IT>
ViewBoundFunction<GM, ACC>::ViewBoundFunction
(
	const FactorType &factor,
	INPUT_IT fixed_vars_begin,
	INPUT_IT fixed_vars_end
)
{
	opengm::FastSequence<IndexType> variableIndices;
	opengm::FastSequence<IndexType> shape(factor.numberOfVariables() - std::distance(fixed_vars_begin, fixed_vars_end));
	variableIndices.reserve(shape.size());
	for (IndexType i = 0; i < factor.numberOfVariables(); ++i) {
		if (std::find(fixed_vars_begin, fixed_vars_end, i) == fixed_vars_end) {
			shape[variableIndices.size()] = factor.numberOfLabels(i);
			variableIndices.push_back(i);
		}
	}

	this->resize(shape.begin(), shape.end());
	for (auto it = this->begin(); it != this->end(); ++it) {
		*it = AccumulationType::template neutral<ValueType>();
	}

	ShapeWalker<typename FactorType::ShapeIteratorType> walker(factor.shapeBegin(), factor.numberOfVariables());
	for (size_t size = 0; size < factor.size(); ++size, ++walker) {
		opengm::FastSequence<ValueType> current(variableIndices.size());
		for (IndexType i = 0; i < variableIndices.size(); ++i) {
			current[i] = walker.coordinateTuple()[variableIndices[i]];
		}
		AccumulationType::op(factor(walker.coordinateTuple().begin()), (*this)(current.begin()));
	}
}

} // namespace

#endif
