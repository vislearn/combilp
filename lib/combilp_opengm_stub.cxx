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

#include <opengm/functions/explicit_function.hxx>
#include <opengm/functions/fieldofexperts.hxx>
#include <opengm/functions/pottsg.hxx>
#include <opengm/functions/potts.hxx>
#include <opengm/functions/pottsn.hxx>
#include <opengm/functions/truncated_absolute_difference.hxx>
#include <opengm/functions/truncated_squared_difference.hxx>
#include <opengm/graphicalmodel/graphicalmodel_hdf5.hxx>
#include <opengm/graphicalmodel/graphicalmodel.hxx>
#include <opengm/operations/adder.hxx>
#include <opengm/operations/minimizer.hxx>

typedef double ValueType;
typedef size_t IndexType;
typedef size_t LabelType;
typedef opengm::Adder OperatorType;
typedef opengm::Minimizer AccumulatorType;
typedef opengm::DiscreteSpace<IndexType, LabelType> SpaceType;

typedef opengm::meta::TypeListGenerator<
	opengm::ExplicitFunction<ValueType, IndexType, LabelType>,
	opengm::PottsFunction<ValueType, IndexType, LabelType>,
	opengm::PottsNFunction<ValueType, IndexType, LabelType>,
	opengm::PottsGFunction<ValueType, IndexType, LabelType>,
	opengm::TruncatedSquaredDifferenceFunction<ValueType, IndexType, LabelType>,
	opengm::TruncatedAbsoluteDifferenceFunction<ValueType, IndexType, LabelType>,
	opengm::FoEFunction<ValueType, IndexType, LabelType>
>::type FunctionTypeList;

typedef opengm::GraphicalModel<
	ValueType,
	OperatorType,
	FunctionTypeList,
	SpaceType
> GmType;

extern "C" {

void
combilp_opengm_stub_load_from_file
(
	const char *filename,
	const char *dataset,
	void (*init_shape)(int num_vars, const int *shape),
	void (*add_factor)(int num_vars, const int *vars, const double *data)
)
{
	std::cout << "Loading model..." << std::flush;
	GmType gm;
	opengm::hdf5::load(gm, filename, dataset);
	std::cout << " ok" << std::endl;

	std::cout << "Initializing model..." << std::flush;
	{
		std::vector<int> shape(gm.numberOfVariables());
		for (IndexType i = 0; i < gm.numberOfVariables(); ++i)
			shape[i] = gm.numberOfLabels(i);
		init_shape(shape.size(), shape.data());
	}
	std::cout << " ok" << std::endl;

	std::cout << "Initializing factors..." << std::flush;
	for (IndexType i = 0; i < gm.numberOfFactors(); ++i) {
		auto &factor = gm[i];
		std::vector<int> variables(factor.variableIndicesBegin(), factor.variableIndicesEnd());

		std::vector<double> data(factor.size());
		factor.copyValuesSwitchedOrder(data.begin());

		add_factor(factor.numberOfVariables(), variables.data(), data.data());
	}
	std::cout << " ok" << std::endl;
}

}
