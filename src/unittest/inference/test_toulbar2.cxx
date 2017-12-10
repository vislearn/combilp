#include <opengm/graphicalmodel/graphicalmodel.hxx>
#include <opengm/inference/external/toulbar2.hxx>
#include <opengm/operations/adder.hxx>
#include <opengm/operations/minimizer.hxx>
#include <opengm/unittests/blackboxtester.hxx>
#include <opengm/unittests/blackboxtests/blackboxtestfull.hxx>
#include <opengm/unittests/blackboxtests/blackboxtestgrid.hxx>
#include <opengm/unittests/blackboxtests/blackboxteststar.hxx>

int main() {
	opengm::external::ToulBar2_initialize();

	typedef opengm::GraphicalModel<double, opengm::Adder> GraphicalModelType;

	typedef opengm::BlackBoxTestGrid<GraphicalModelType> GridTest;
	typedef opengm::BlackBoxTestFull<GraphicalModelType> FullTest;
	typedef opengm::BlackBoxTestStar<GraphicalModelType> StarTest;

	bool randomLabelSize = true;
	opengm::InferenceBlackBoxTester<GraphicalModelType> tester;
	tester.addTest(new GridTest(1,  4,  4, false,  true, GridTest::RANDOM, opengm::OPTIMAL, 2000));
	tester.addTest(new GridTest(1,  4,  4, false, false, GridTest::RANDOM, opengm::OPTIMAL, 2000));
	tester.addTest(new GridTest(1,  4,  8, false,  true, GridTest::RANDOM, opengm::OPTIMAL, 100));
	tester.addTest(new GridTest(1,  4,  8, false, false, GridTest::RANDOM, opengm::OPTIMAL, 100));
	tester.addTest(new GridTest(1,  4, 20,  true,  true, GridTest::RANDOM, opengm::OPTIMAL, 10));
	tester.addTest(new GridTest(1,  4, 20, false, false, GridTest::RANDOM, opengm::OPTIMAL, 10));

	tester.addTest(new GridTest(1, 10,  3, false,  true, GridTest::RANDOM, opengm::OPTIMAL, 10));
	tester.addTest(new GridTest(1,  5,  8, false,  true, GridTest::RANDOM, opengm::OPTIMAL, 10));

	tester.addTest(new GridTest(3,  3,  5, false,  true, GridTest::RANDOM, opengm::OPTIMAL, 5));
	tester.addTest(new GridTest(3,  3,  5, false, false, GridTest::RANDOM, opengm::OPTIMAL, 5));

	tester.addTest(new StarTest(5,      6, false,  true, StarTest::RANDOM, opengm::OPTIMAL, 20));
	tester.addTest(new StarTest(5,      6, false, false, StarTest::RANDOM, opengm::OPTIMAL, 20));

	tester.addTest(new FullTest(3,      6, false,     2, FullTest::RANDOM, opengm::OPTIMAL, 20));

	std::cout << "Test ToulBar2 ..." << std::endl;
	typedef opengm::external::ToulBar2<GraphicalModelType, opengm::Minimizer> Inference;
	Inference::Parameter param;
	tester.test<Inference>(param);
}
