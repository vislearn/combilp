//
// File: combilp_default.hxx
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
#ifndef OPENGM_COMBILP_DEFAULT_HXX
#define OPENGM_COMBILP_DEFAULT_HXX

#include <opengm/inference/combilp.hxx>
#include <opengm/inference/lpcplex.hxx>
#include <opengm/inference/trws/trws_adsal.hxx>
#include <opengm/inference/trws/trws_trws.hxx>
#include <opengm/inference/external/srmp.hxx>
#include <opengm/inference/external/toulbar2.hxx>
#include <opengm/inference/external/trws_shekhovtsov.hxx>

namespace opengm {

//
// PUBLIC INTERFACE
//

enum CombiLP_LP_Type {
	CombiLP_LP_ADSal,
	CombiLP_LP_SRMP,
	CombiLP_LP_TRWSi,
	CombiLP_LP_TRWS_Shekhovtsov
};

enum CombiLP_ILP_Type {
	CombiLP_ILP_Cplex,
	CombiLP_ILP_ToulBar
};

// Forward declaration only, not part of public interface.
template<class GM, class ACC, CombiLP_LP_Type LP>
struct CombiLP_LP_TypeGen;

// Forward declaration only, not part of public interface.
template<class GM, class ACC, CombiLP_ILP_Type ILP>
struct CombiLP_ILP_TypeGen;

template<class GM, class ACC, CombiLP_LP_Type LP, CombiLP_ILP_Type ILP>
struct CombiLP_TypeGen {
	typedef CombiLP<GM, ACC, typename CombiLP_LP_TypeGen<GM, ACC, LP>::Type,
			typename CombiLP_ILP_TypeGen<GM, ACC, ILP>::Type> Type;
};

//
// INTERNAL TEMPLATE MAGIC
//
// You probably do not want to use these internal types.
//

// CombiLP_LP_TypeGen


template<class GM, class ACC>
struct CombiLP_LP_TypeGen<GM, ACC, CombiLP_LP_ADSal> {
	typedef ADSal<GM, ACC> Type;
};

template<class GM>
struct CombiLP_LP_TypeGen<GM, opengm::Minimizer, CombiLP_LP_SRMP> {
	typedef external::SRMP<GM> Type;
};

template<class GM, class ACC>
struct CombiLP_LP_TypeGen<GM, ACC, CombiLP_LP_TRWSi> {
	typedef TRWSi<GM, ACC> Type;
};

template<class GM, class ACC>
struct CombiLP_LP_TypeGen<GM, ACC, CombiLP_LP_TRWS_Shekhovtsov> {
	typedef external::TrwsShekhovtsov<GM, ACC> Type;
};

// CombiLP_ILP_TypeGen

template<class GM, class ACC>
struct CombiLP_ILP_TypeGen<GM, ACC, CombiLP_ILP_Cplex> {
	typedef LPCplex<typename CombiLP_ILP_GraphicalModelTypeGen<GM>::GraphicalModelType, ACC> Type;
};

template<class GM, class ACC>
struct CombiLP_ILP_TypeGen<GM, ACC, CombiLP_ILP_ToulBar> {
	typedef external::ToulBar2<typename CombiLP_ILP_GraphicalModelTypeGen<GM>::GraphicalModelType, ACC> Type;
};

} // namespace opengm

#endif
