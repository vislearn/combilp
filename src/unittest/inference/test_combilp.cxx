#include <opengm/graphicalmodel/graphicalmodel.hxx>
#include <opengm/operations/adder.hxx>
#include <opengm/operations/minimizer.hxx>

#include <opengm/unittests/blackboxtester.hxx>
#include <opengm/unittests/blackboxtests/blackboxtestgrid.hxx>
#include <opengm/unittests/blackboxtests/blackboxtestfull.hxx>
#include <opengm/unittests/blackboxtests/blackboxteststar.hxx>

#include <opengm/inference/combilp_default.hxx>


int main() {
	time_t seed = time(0);
	std::cout << "Random seed: " << seed << std::endl;
	srand(seed);

	typedef opengm::GraphicalModel<double, opengm::Adder> GraphicalModelType;
	typedef opengm::BlackBoxTestGrid<GraphicalModelType> GridTest;
	typedef opengm::BlackBoxTestFull<GraphicalModelType> FullTest;
	typedef opengm::BlackBoxTestStar<GraphicalModelType> StarTest;

	opengm::InferenceBlackBoxTester<GraphicalModelType> tester;
	tester.addTest(new GridTest(15, 15, 10, true,  true, GridTest::RANDOM, opengm::OPTIMAL, 10));

	std::cout << "Test CombiLP ..." << std::endl;
	{
		typedef opengm::CombiLP_TypeGen<GraphicalModelType, opengm::Minimizer,
			opengm::CombiLP_LP_TRWS_Shekhovtsov, opengm::CombiLP_ILP_Cplex>::Type
			CombiLPType;
		typename CombiLPType::Parameter param;
		param.ilpsolverParameter_.integerConstraint_ = true;
		tester.test<CombiLPType>(param);
	}
}
