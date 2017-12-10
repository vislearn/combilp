#ifndef OPENGM_INFERENCE_EXTERNAL_TRWS_SHEKHOVTSOV_HXX
#define OPENGM_INFERENCE_EXTERNAL_TRWS_SHEKHOVTSOV_HXX

#include <stdexcept>
#include <string>
#include <vector>
#include <iomanip>

#include <boost/scoped_ptr.hpp>

#include <opengm/graphicalmodel/graphicalmodel.hxx>
#include <opengm/inference/inference.hxx>
#include <opengm/inference/trws/trws_trws.hxx>
#include <opengm/inference/auxiliary/lp_reparametrization.hxx>

#include <trws_shekhovtsov/trws_machine.h>

namespace opengm {
namespace external {
namespace trws_shekhovtsov {

template<class GM>
typename GM::ValueType dual_energy(const LPReparametrisationStorage<GM> &repa)
{
	const GM &gm = repa.graphicalModel();
	typename GM::ValueType energy = 0;
	for (typename GM::IndexType i = 0; i < gm.numberOfFactors(); ++i) {
		typename GM::ValueType minimum = std::numeric_limits<typename GM::ValueType>::infinity();
		typedef ShapeWalker<typename GM::FactorType::ShapeIteratorType> ShapeWalkerType;
		ShapeWalkerType walker(gm[i].shapeBegin(), gm[i].numberOfVariables());
		for (size_t j = 0; j < gm[i].size(); ++j, ++walker)
			minimum = std::min(minimum, repa.getFactorValue(i, walker.coordinateTuple().begin()));
		energy += minimum;
	}
	return energy;
}

template<class GM, class T>
LPReparametrisationStorage<GM> trws_to_repa(const GM &gm, const trws_machine<T> &trws)
{
	LPReparametrisationStorage<GM> repa(gm);

	for (int e = 0, fidx = 0; fidx < gm.numberOfFactors(); ++fidx) {
		if (gm.numberOfVariables(fidx) != 2)
			continue;

		std::vector<typename T::type> msgs;

		// Backward Messages
		const_cast<trws_machine<T>&>(trws).get_message(e, false, msgs);
		OPENGM_ASSERT_OP(gm.numberOfLabels(gm[fidx].variableIndex(0)), ==, msgs.size());
		OPENGM_ASSERT_OP(repa.get(fidx, 0).size(), ==, msgs.size());
		std::transform(msgs.begin(), msgs.end(), repa.get(fidx, 0).begin(), std::negate<typename GM::ValueType>());

		// Forward Messages
		FastSequence<typename GM::LabelType> ll(2);
		auto &repa_values = repa.get(fidx, 1);
		for (ll[1] = 0; ll[1] < gm.numberOfLabels(gm[fidx].variableIndex(1)); ++ll[1]) {
			typename GM::ValueType minimum = std::numeric_limits<typename GM::ValueType>::infinity();
			for (ll[0] = 0; ll[0] < gm.numberOfLabels(gm[fidx].variableIndex(0)); ++ll[0]) {
				typename GM::ValueType value = repa.getFactorValue(fidx, ll.begin());
				minimum = std::min(minimum, value);
			}
			repa_values[ll[1]] = -minimum;

#ifndef NDEBUG
			minimum = std::numeric_limits<typename GM::ValueType>::infinity();
			for (ll[0] = 0; ll[0] < gm.numberOfLabels(gm[fidx].variableIndex(0)); ++ll[0]) {
				typename GM::ValueType value = repa.getFactorValue(fidx, ll.begin());
				minimum = std::min(minimum, value);
			}
			OPENGM_ASSERT_OP(std::abs(minimum), <=, 1e-8);
#endif
		}

		++e;
	}

	OPENGM_ASSERT_OP(std::abs(dual_energy(repa) - trws.LB), <=, 1);

	return repa;
}

template<class GM, class T>
void convert_energy(const GM &gm, energy_auto<T> &energy)
{
	// FIXME: energy_auto<T> is templated by T, but inside double is used.
	typedef double T2; // :-/
	
	// calculate maxK
	typename GM::LabelType maxK = 0;
	for (typename GM::IndexType i = 0; i < gm.numberOfVariables(); ++i)
		maxK = std::max(maxK, gm.numberOfLabels(i));

	// storage for unary and pairwise values
	std::vector<T2> _f1(maxK);
	std::vector<T2> _f2(maxK * maxK);

	// initialize energy
	energy.set_nV(gm.numberOfVariables());
	energy.G._nV = gm.numberOfVariables(); // TODO: SRSLY?
	energy.K.resize(gm.numberOfVariables());

	typename GM::IndexType edge_counter = 0;

	// first pass: gather shape of problem and factors
	for (typename GM::IndexType fidx = 0; fidx < gm.numberOfFactors(); ++fidx) {
		switch (gm.numberOfVariables(fidx)) {
			case 0: {
				std::cout << "WARNING: Constant factor is skipped. Energy and bound of TRW-S output don’t reflect this constant." << std::endl;
				break;
			}
			case 1: {
				typename GM::IndexType v =gm[fidx].variableIndex(0);
				energy.K[v] = gm.numberOfLabels(v);
				} break;
			case 2:
				++edge_counter;
				break;
			default:
				throw std::runtime_error("Only 2nd order models are supported!");
				break;
		}
	}

	energy.set_nE(edge_counter);
	energy.maxK = maxK;
	edge_counter = 0;

	// second pass: set values
	for (typename GM::IndexType fidx = 0; fidx < gm.numberOfFactors(); ++fidx) {
		switch (gm.numberOfVariables(fidx)) {
			case 0: break;
			case 1: {
				typename GM::IndexType v = gm[fidx].variableIndex(0);
				typename GM::LabelType k = gm.numberOfLabels(v);
				dynamic::num_array<T2, 1> f1;
				_f1.resize(k);
				gm[fidx].copyValues(_f1.begin());
				f1.set_ref(&_f1[0], k);
				energy.set_f1(v, f1);
				} break;
			case 2: {
				typename GM::IndexType u = gm[fidx].variableIndex(0);
				typename GM::IndexType v = gm[fidx].variableIndex(1);
				typename GM::LabelType ku = gm.numberOfLabels(u);
				typename GM::LabelType kv = gm.numberOfLabels(v);
				_f2.resize(ku * kv);
				gm[fidx].copyValues(_f2.begin());
				dynamic::num_array<T2, 2> f2;
				f2.set_ref(&_f2[0], exttype::mint2(ku, kv));
				energy.G.E[edge_counter][0] = u;
				energy.G.E[edge_counter][1] = v;
				energy.set_f2(edge_counter, f2);
				++edge_counter;
				} break;
			default:
				throw std::runtime_error("Only 2nd order models are supported!");
				break;
		}
	}

	// finalize energy
	energy.G.edge_index();
	energy.init();
	energy.report();
}

} // namespace trws_shekhovtsov

template <class GM, class ACC>
class TrwsShekhovtsov : public Inference<GM, ACC> {
public:
	struct Parameter {
		Parameter(int trws_iterations=1000)
		: trws_iterations(trws_iterations)
		{
		}

		int trws_iterations;
	
		template<class T>void print(const T &t) const { }
	};

	typedef GM GraphicalModelType;
	typedef ACC AccumulationType;
	typedef typename GraphicalModelType::LabelType LabelType;
	typedef typename GraphicalModelType::IndexType IndexType;
	typedef typename GraphicalModelType::ValueType ValueType;
	typedef typename GraphicalModelType::OperatorType OperatorType;
	typedef typename GraphicalModelType::FactorType FactorType;
	typedef typename GraphicalModelType::IndependentFactorType IndependentFactorType;
	typedef typename GraphicalModelType::FunctionIdentifier FunctionIdentifier;

	typedef typename LPReparametrizer<GraphicalModelType>::ReparametrizedGMType ReparametrizedModelType;

	TrwsShekhovtsov(const GraphicalModelType &gm, const Parameter &param = Parameter());
	std::string name() const { return "opengm::external::TrwsShekhovtsov"; }
	const GraphicalModelType& graphicalModel() const { return *gm_; }
	InferenceTermination infer();

	ValueType bound() const { return trws_.LB; }
	ValueType value() const { return trws_.best_E; }
	InferenceTermination arg(std::vector<LabelType>& out, const size_t num = 1) const;
	void getReparametrization(LPReparametrisationStorage<GraphicalModelType>&);
	void setReparametrization(const LPReparametrisationStorage<GraphicalModelType>&);

private:
	Parameter param_;
	const GraphicalModelType *gm_;
	energy_auto<float> energy_;
	trws_machine<float_v4> trws_;
};

template<class GM, class ACC>
TrwsShekhovtsov<GM, ACC>::TrwsShekhovtsov
(
	const GraphicalModelType &gm,
	const Parameter &param
)
: param_(param)
, gm_(&gm)
{
	trws_shekhovtsov::convert_energy(*gm_, energy_);
	trws_.init(&energy_);
	trws_.ops->max_it = param_.trws_iterations;
	trws_.ops->it_batch = 100;
	//trws_.ops->gap_tol = 0;
	trws_.ops->rel_gap_tol = 1e-5;
	//trws_.ops->conv_tol = 0;
	trws_.ops->rel_conv_tol = 1e-5;
}

template<class GM, class ACC>
InferenceTermination
TrwsShekhovtsov<GM, ACC>::infer()
{
	trws_.run_converge();
	return NORMAL;
}

template<class GM, class ACC>
InferenceTermination
TrwsShekhovtsov<GM, ACC>::arg
(
	std::vector<LabelType> &out,
	const size_t num
) const
{
	out.resize(gm_->numberOfVariables());
	for (IndexType i = 0; i < out.size(); ++i)
		out[i] = trws_.best_x[i];

	// FIXME: return correct value
	return NORMAL;
}

template<class GM, class ACC>
void
TrwsShekhovtsov<GM, ACC>::getReparametrization
(
	LPReparametrisationStorage<GraphicalModelType>& repa
)
{
	repa = trws_shekhovtsov::trws_to_repa(*gm_, trws_);
}

template<class GM, class ACC>
void
TrwsShekhovtsov<GM, ACC>::setReparametrization
(
	const LPReparametrisationStorage<GraphicalModelType>& repa
)
{
	throw std::runtime_error("Not implemented");
}

} // namespace external
} // namespace opengm

#endif
