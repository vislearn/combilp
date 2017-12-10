#ifndef OPENGM_TOULBAR2_CALLER_HXX_
#define OPENGM_TOULBAR2_CALLER_HXX_

#include <opengm/opengm.hxx>
#include <opengm/inference/external/toulbar2.hxx>

#include "inference_caller_base.hxx"
#include "../argument/argument.hxx"

namespace opengm {
namespace interface {

template <class IO, class GM, class ACC>
class ToulBar2Caller : public InferenceCallerBase<IO, GM, ACC, ToulBar2Caller<IO, GM, ACC> >
{
public:
	typedef external::ToulBar2<GM, ACC> ToulBar2Type;
	typedef InferenceCallerBase<IO, GM, ACC, ToulBar2Caller<IO, GM, ACC> > BaseClass;
	typedef typename ToulBar2Type::VerboseVisitorType VerboseVisitorType;
	typedef typename ToulBar2Type::EmptyVisitorType EmptyVisitorType;
	typedef typename ToulBar2Type::TimingVisitorType TimingVisitorType;

	const static std::string name_;

	ToulBar2Caller
	(
		IO& ioIn
	)
	: BaseClass(name_, "ToulBar2", ioIn)
	{ }

	virtual void runImpl
	(
		GM &model,
		typename BaseClass::OutputBase &output,
		const bool verbose
	)
	{
		typename ToulBar2Type::Parameter parameter;
		this->template infer<ToulBar2Type, TimingVisitorType, typename ToulBar2Type::Parameter>(model, output, verbose, parameter);
	}

protected:
	using BaseClass::addArgument;
	using BaseClass::io_;
	using BaseClass::infer;
};

template<class IO, class GM, class ACC>
const std::string ToulBar2Caller<IO, GM, ACC>::name_ = "ToulBar2";

} // namespace interface
} // namespace opengm

#endif
