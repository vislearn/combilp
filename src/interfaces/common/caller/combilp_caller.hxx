/*
 * combilp_caller.hxx
 *
 *  Created on: May 22, 2013
 *      Author: bsavchyn
 */

#ifndef COMBILP_CALLER_HXX_
#define COMBILP_CALLER_HXX_

#include <opengm/opengm.hxx>
#include <opengm/inference/trws/trws_trws.hxx>
#include <opengm/inference/trws/trws_adsal.hxx>
#include <opengm/inference/combilp.hxx>
#include <opengm/inference/combilp_default.hxx>

#include "inference_caller_base.hxx"
#include "../argument/argument.hxx"

namespace opengm {

namespace interface {

namespace combilp {

	const char *LP_TYPE_ADSAL = "ADsal";
	const char *LP_TYPE_SRMP = "SRMP";
	const char *LP_TYPE_TRWSi = "TRWSi";
	const char *LP_TYPE_TRWS = "TRWS";

	const char *ILP_TYPE_CPLEX = "CPLEX";
	const char *ILP_TYPE_TOULBAR2 = "ToulBar2";

	template<CombiLP_LP_Type LP>
	struct Parameter_LP_Helper {
		template<class CALLER, class PARAM>
		void configure(CALLER *caller, PARAM *param) { }
	};

	template<>
	struct Parameter_LP_Helper<CombiLP_LP_SRMP> {
		template<class CALLER, class PARAM>
		void configure(CALLER *caller, PARAM *param) {
			param->iter_max = caller->parameter_lp_iterations;
			param->time_max = std::numeric_limits<double>::infinity();
			param->verbose = caller->parameter_verbose;
			param->sort_flag = -1;
			param->BLPRelaxation_ = true;
		}
	};

	template<>
	struct Parameter_LP_Helper<CombiLP_LP_TRWS_Shekhovtsov> {
		template<class CALLER, class PARAM>
		void configure(CALLER *caller, PARAM *param) {
			param->trws_iterations = caller->parameter_lp_iterations;
			// FIXME: set verbose here
		}
	};

	template<>
	struct Parameter_LP_Helper<CombiLP_LP_TRWSi> {
		template<class CALLER, class PARAM>
		void configure(CALLER *caller, PARAM *param) {
			param->maxNumberOfIterations_ = caller->parameter_lp_iterations;
			param->verbose_ = caller->parameter_verbose;
		}
	};

	template<CombiLP_ILP_Type ILP>
	struct Parameter_ILP_Helper {
		template<class CALLER, class PARAM>
		void configure(CALLER *caller, PARAM *param) { }
	};

	template<>
	struct Parameter_ILP_Helper<CombiLP_ILP_Cplex> {
		template<class CALLER, class PARAM>
		void configure(CALLER *caller, PARAM *param) {
			param->integerConstraint_ = true;
			param->normalizePotential_ = true;
			param->verbose_ = true;
			param->numberOfThreads_ = 1;
			param->verbose_ = caller->parameter_verbose;
		}
	};

	template<class CALLER, CombiLP_LP_Type LP, CombiLP_ILP_Type ILP>
	struct CallerHelper2 {
		CALLER *caller;

		CallerHelper2(CALLER *caller)
		: caller(caller)
		{ }

		void run(typename CALLER::GraphicalModelType &model, typename CALLER::OutputBase &output, const bool verbose) {
			typedef typename CombiLP_TypeGen<typename CALLER::GraphicalModelType, typename CALLER::AccumulationType, LP, ILP>::Type InfType;
			typename InfType::Parameter param;

			param.verbose_ = caller->parameter_verbose;
			param.loadReparametrizationFileName_ = caller->parameter_loadReparametrizationFileName;
			param.saveReparametrizationFileName_ = caller->parameter_saveReparametrizationFileName;
			param.saveProblemMasks_ = caller->parameter_saveProblemMasks;
			param.maskFileNamePre_ = caller->parameter_maskFileNamePre;
			param.maxNumberOfILPCycles_ = caller->parameter_maxNumberOfILPCycles;
			param.enableDenseVersion_ = caller->parameter_dense;
			param.enablePotentialRedistribution_ = caller->parameter_redistribute;

			Parameter_LP_Helper<LP> parameter_lp;
			parameter_lp.configure(caller, &param.lpsolverParameter_);
		
			Parameter_ILP_Helper<ILP> parameter_ilp;
			parameter_ilp.configure(caller, &param.ilpsolverParameter_);
			
			caller->template infer<InfType, typename InfType::TimingVisitorType,
				typename InfType::Parameter>(model, output, verbose, param);

		}
	};

	template<class CALLER, CombiLP_LP_Type LP>
	struct CallerHelper1 {
		CALLER *caller;

		CallerHelper1(CALLER *caller)
		: caller(caller)
		{ }

		void run(typename CALLER::GraphicalModelType &model, typename CALLER::OutputBase &output, const bool verbose) {
			if (caller->parameter_ilp_type == ILP_TYPE_CPLEX) {
				CallerHelper2<CALLER, LP, CombiLP_ILP_Cplex> helper(caller);
				helper.run(model, output, verbose);
			} else if (caller->parameter_ilp_type == ILP_TYPE_TOULBAR2) {
				CallerHelper2<CALLER, LP, CombiLP_ILP_ToulBar> helper(caller);
				helper.run(model, output, verbose);
			} else {
				std::stringstream message;
				message << "Unknown COMBILP_ILP_TYPE: " << caller->parameter_ilp_type;
				throw RuntimeError(message.str());
			}
		}
	};

	template<class CALLER>
	struct CallerHelper {
		CALLER *caller;

		CallerHelper(CALLER *caller)
		: caller(caller)
		{ }

		void run(typename CALLER::GraphicalModelType &model, typename CALLER::OutputBase &output, const bool verbose) {
			if (caller->parameter_lp_type == LP_TYPE_ADSAL) {
				CallerHelper1<CALLER, CombiLP_LP_ADSal> helper(caller);
				helper.run(model, output, verbose);
			} else if (caller->parameter_lp_type == LP_TYPE_SRMP) {
				CallerHelper1<CALLER, CombiLP_LP_SRMP> helper(caller);
				helper.run(model, output, verbose);
			} else if (caller->parameter_lp_type == LP_TYPE_TRWS) {
				CallerHelper1<CALLER, CombiLP_LP_TRWS_Shekhovtsov> helper(caller);
				helper.run(model, output, verbose);
			} else if (caller->parameter_lp_type == LP_TYPE_TRWSi) {
				CallerHelper1<CALLER, CombiLP_LP_TRWSi> helper(caller);
				helper.run(model, output, verbose);
			} else {
				std::stringstream message;
				message << "Unknown COMBILP_LP_TYPE: " << caller->parameter_lp_type;
				throw RuntimeError(message.str());
			}
		}
	};

	template<class CALLER>
	CallerHelper<CALLER> make_caller_helper(CALLER *caller) {
		CallerHelper<CALLER> result(caller);
		return result;
	}
}

template <class IO, class GM, class ACC>
class CombiLPCaller : public InferenceCallerBase<IO, GM, ACC, CombiLPCaller<IO, GM, ACC> > {
protected:
	typedef InferenceCallerBase<IO, GM, ACC, CombiLPCaller<IO, GM, ACC> > BaseClass;
	using typename BaseClass::OutputBase;

	virtual void runImpl(GM &model, OutputBase &output, const bool verbose);

	std::string parameter_lp_type;
	std::string parameter_ilp_type;
	bool parameter_verbose;
	std::string parameter_loadReparametrizationFileName;
	std::string parameter_saveReparametrizationFileName;
	bool parameter_saveProblemMasks;
	std::string parameter_maskFileNamePre;
	size_t parameter_maxNumberOfILPCycles;
	size_t parameter_lp_iterations;
	bool parameter_dense;
	bool parameter_redistribute;

public:
   const static std::string name_;
   CombiLPCaller(IO &ioIn);

   template<class> friend class combilp::CallerHelper;
   template<class> friend class combilp::CallerHelper1;
   template<class> friend class combilp::CallerHelper2;
   template<class> friend class combilp::Parameter_LP_Helper;
   template<class> friend class combilp::Parameter_ILP_Helper;
};

template <class IO, class GM, class ACC>
CombiLPCaller<IO, GM, ACC>::CombiLPCaller(IO &ioIn)
: BaseClass(name_, "detailed description of the internal CombiLP caller...", ioIn)
{
	std::vector<std::string> lp_types;
	lp_types.push_back(combilp::LP_TYPE_ADSAL);
	lp_types.push_back(combilp::LP_TYPE_SRMP);
	lp_types.push_back(combilp::LP_TYPE_TRWS);
	lp_types.push_back(combilp::LP_TYPE_TRWSi);

	std::vector<std::string> ilp_types;
	ilp_types.push_back(combilp::ILP_TYPE_CPLEX);
	ilp_types.push_back(combilp::ILP_TYPE_TOULBAR2);

	this->addArgument(StringArgument<>(parameter_lp_type, "", "lp", "select local polytope solver", lp_types[1], lp_types));
	this->addArgument(StringArgument<>(parameter_ilp_type, "", "ilp", "select combinatorial solver", ilp_types[0], ilp_types));
	this->addArgument(BoolArgument(parameter_verbose, "", "debugverbose", "If set the solver will output debug information to the stdout"));
	this->addArgument(StringArgument<>(parameter_loadReparametrizationFileName, "", "loadrepa", "If set to a valid filename the reparametrization will be load", std::string("")));
	this->addArgument(StringArgument<>(parameter_saveReparametrizationFileName, "", "saverepa", "If set to a valid filename the reparametrization will be saved", std::string("")));
	this->addArgument(BoolArgument(parameter_saveProblemMasks, "", "saveProblemMasks", "Saves masks of the subproblems passed to the ILP solver"));
	this->addArgument(StringArgument<>(parameter_maskFileNamePre, "", "maskFileNamePre", "Path and filename prefix of the subproblem masks, see parameter saveProblemMasks", std::string("")));
	this->addArgument(Size_TArgument<>(parameter_maxNumberOfILPCycles, "", "maxNumberOfILPCycles", "Max number of ILP solver cycles", static_cast<size_t>(1000)));
	this->addArgument(Size_TArgument<>(parameter_lp_iterations, "i", "lp-iterations", "Max number of LP solver cycles", static_cast<size_t>(1000)));
	this->addArgument(BoolArgument(parameter_dense, "", "dense", "Enable dense version of CombiLP algorithm"));
	this->addArgument(BoolArgument(parameter_redistribute, "", "redistribute", "Enable potential redistribution"));
}

template <class IO, class GM, class ACC>
void CombiLPCaller<IO, GM, ACC>::runImpl(GM &model, OutputBase &output, const bool verbose)
{
	std::cout << "running internal CombiLP caller" << std::endl;
	combilp::make_caller_helper(this).run(model, output, verbose);
}

template <class IO, class GM, class ACC>
const std::string CombiLPCaller<IO, GM, ACC>::name_ = "CombiLP";

} // namespace interface

} // namespace opengm

#endif /* COMBILP_CALLER_HXX_ */
