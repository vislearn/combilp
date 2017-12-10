/*
 * lp_reparametrization_storage.hxx
 *
 *  Created on: Sep 16, 2013
 *      Author: bsavchyn
 */

#ifndef LP_REPARAMETRIZATION_STORAGE_HXX_
#define LP_REPARAMETRIZATION_STORAGE_HXX_
#include <opengm/inference/trws/utilities2.hxx>
#include <opengm/graphicalmodel/graphicalmodel_factor_accumulator.hxx>
#include <opengm/datastructures/marray/marray_hdf5.hxx>


//#ifdef WITH_HDF5
//#include <opengm/inference/auxiliary/lp_reparametrization_hdf5.hxx>
//#endif


namespace opengm{

#ifdef TRWS_DEBUG_OUTPUT
using OUT::operator <<;
#endif


template<class GM>
class LPReparametrisationStorage{
public:
	typedef GM GraphicalModelType;
	typedef typename GM::ValueType ValueType;
	typedef typename GM::FactorType FactorType;
	typedef typename GM::IndexType IndexType;
	typedef typename GM::LabelType LabelType;

	//typedef std::valarray<ValueType> UnaryFactor;
	typedef std::vector<ValueType> UnaryFactor;
	typedef ValueType* uIterator;
	typedef std::vector<UnaryFactor> VecUnaryFactors;
	typedef std::map<IndexType,IndexType> VarIdMapType;
	LPReparametrisationStorage(const GM& gm);

	const UnaryFactor& get(IndexType factorIndex,IndexType relativeVarIndex) const
	{
		OPENGM_ASSERT(factorIndex < _gm->numberOfFactors());
		OPENGM_ASSERT(relativeVarIndex < _dualVariables[factorIndex].size());
		return _dualVariables[factorIndex][relativeVarIndex];
	}

	UnaryFactor& get(IndexType factorIndex,IndexType relativeVarIndex)
	{
		return const_cast<UnaryFactor&>(static_cast<const LPReparametrisationStorage*>(this)->get(factorIndex, relativeVarIndex));
	}

	std::pair<uIterator,uIterator> getIterators(IndexType factorIndex,IndexType relativeVarIndex)
				{
		OPENGM_ASSERT(factorIndex < _gm->numberOfFactors());
		OPENGM_ASSERT(relativeVarIndex < _dualVariables[factorIndex].size());
		UnaryFactor& uf=_dualVariables[factorIndex][relativeVarIndex];
		uIterator begin=&uf[0];
		return std::make_pair(begin,begin+uf.size());
				}

	template<class OUTPUT_ITERATOR>
	void copyFactorValues(IndexType findex, OUTPUT_ITERATOR it) const
	{
		OPENGM_ASSERT(findex < _gm->numberOfFactors());
		const typename GM::FactorType& factor = (*_gm)[findex];

		typedef ShapeWalker<typename GraphicalModelType::FactorType::ShapeIteratorType> ShapeWalkerType;
		ShapeWalkerType walker(factor.shapeBegin(), factor.numberOfVariables());
		for (IndexType i = 0; i < factor.size(); ++i, ++walker, ++it) {
			*it = getFactorValue(findex, walker.coordinateTuple().begin());
		}
	}

	template<class ITERATOR>
	ValueType getFactorValue(IndexType findex,ITERATOR it)const
	{
		OPENGM_ASSERT(findex < _gm->numberOfFactors());
		const typename GM::FactorType& factor=(*_gm)[findex];

		ValueType res=0;//factor(it);
		if (factor.numberOfVariables()>1)
		{
			res=factor(it);
			for (IndexType varId=0;varId<factor.numberOfVariables();++varId)
			{
				OPENGM_ASSERT(varId < _dualVariables[findex].size());
				OPENGM_ASSERT(*(it+varId) < _dualVariables[findex][varId].size());
				res+=_dualVariables[findex][varId][*(it+varId)];
			}
		}else
		{
			res=getVariableValue(factor.variableIndex(0),*it);
		}
		return res;
	}

	ValueType getVariableValue(IndexType varIndex,LabelType label)const
	{
		OPENGM_ASSERT(varIndex < _gm->numberOfVariables());
		ValueType res=0.0;
		for (IndexType i=0;i<_gm->numberOfFactors(varIndex);++i)
		{
			IndexType factorId=_gm->factorOfVariable(varIndex,i);
			OPENGM_ASSERT(factorId < _gm->numberOfFactors());
			if ((*_gm)[factorId].numberOfVariables()==1)
			{
				res+=(*_gm)[factorId](&label);
				continue;
			}

			OPENGM_ASSERT( factorId < _dualVariables.size() );
			OPENGM_ASSERT(label < _dualVariables[factorId][localId(factorId,varIndex)].size());
			res-=_dualVariables[factorId][localId(factorId,varIndex)][label];
		}

		return res;
	}
#ifdef TRWS_DEBUG_OUTPUT
	void PrintTestData(std::ostream& fout)const;
#endif
	IndexType localId(IndexType factorId,IndexType varIndex)const{
		typename VarIdMapType::const_iterator it = _localIdMap[factorId].find(varIndex);
		trws_base::exception_check(it!=_localIdMap[factorId].end(),"LPReparametrisationStorage:localId() - factor and variable are not connected!");
		return it->second;};

	const GM& graphicalModel()const{return *_gm;}

	template<class VECTOR>
	void serialize(VECTOR* pserialization)const;
	template<class VECTOR>
	void deserialize(const VECTOR& serialization);
private:
	const GM* _gm;
	std::vector<VecUnaryFactors> _dualVariables;
	std::vector<VarIdMapType> _localIdMap;
};

template<class GM>
LPReparametrisationStorage<GM>::LPReparametrisationStorage(const GM& gm)
:_gm(&gm),_localIdMap(gm.numberOfFactors())
 {
	_dualVariables.resize(_gm->numberOfFactors());
	//for all factors with order > 1
	for (IndexType findex=0;findex<_gm->numberOfFactors();++findex)
	{
		IndexType numVars=(*_gm)[findex].numberOfVariables();
		VarIdMapType& mapFindex=_localIdMap[findex];
		if (numVars>=2)
		{

			_dualVariables[findex].resize(numVars);
			//std::valarray<IndexType> v(numVars);
			std::vector<IndexType> v(numVars);
			(*_gm)[findex].variableIndices(&v[0]);
			for (IndexType n=0;n<numVars;++n)
			{
				//_dualVariables[findex][n].assign(_gm->numberOfLabels(v[n]),0.0);//TODO. Do it like this
				_dualVariables[findex][n].resize(_gm->numberOfLabels(v[n]));
				mapFindex[v[n]]=n;
			}
		}
	}

 }

#ifdef TRWS_DEBUG_OUTPUT
template<class GM>
void LPReparametrisationStorage<GM>::PrintTestData(std::ostream& fout)const
{
	fout << "_dualVariables.size()=" << _dualVariables.size()<<std::endl;
	for (IndexType factorIndex=0;factorIndex<_dualVariables.size();++factorIndex )
	{
		fout <<"factorIndex="<<factorIndex<<": ---------------------------------"<<std::endl;
		for (IndexType varId=0;varId<_dualVariables[factorIndex].size();++varId)
			fout <<"varId="<<varId<<": "<< _dualVariables[factorIndex][varId]<<std::endl;
	}
}
#endif

template<class GM>
template<class VECTOR>
void LPReparametrisationStorage<GM>::serialize(VECTOR* pserialization)const
{
//computing total space needed:
size_t i=0;
for (IndexType factorId=0;factorId<_dualVariables.size();++factorId)
 for (IndexType localId=0;localId<_dualVariables[factorId].size();++localId)
  for (LabelType label=0;label<_dualVariables[factorId][localId].size();++label)
	  ++i;

 pserialization->resize(i);
 //serializing....
 i=0;
 for (IndexType factorId=0;factorId<_dualVariables.size();++factorId)
	 for (IndexType localId=0;localId<_dualVariables[factorId].size();++localId)
		for (LabelType label=0;label<_dualVariables[factorId][localId].size();++label)
		 (*pserialization)[i++]=_dualVariables[factorId][localId][label];
}

template<class GM>
template<class VECTOR>
void LPReparametrisationStorage<GM>::deserialize(const VECTOR& serialization)
{
	size_t i=0;
	 for (IndexType factorId=0;factorId<_gm->numberOfFactors();++factorId)
	 {
		 OPENGM_ASSERT(factorId<_dualVariables.size());
		 if ((*_gm)[factorId].numberOfVariables()==1) continue;
		 for (IndexType localId=0;localId<(*_gm)[factorId].numberOfVariables();++localId)
		 {
			OPENGM_ASSERT(localId<_dualVariables[factorId].size());
			for (LabelType label=0;label<_dualVariables[factorId][localId].size();++label)
			{
			 OPENGM_ASSERT(label<_dualVariables[factorId][localId].size());
			 if (i>=serialization.size())
				 throw std::runtime_error("LPReparametrisationStorage<GM>::deserialize(): Size of serialization is less than required for the graphical model! Deserialization failed.");
			 _dualVariables[factorId][localId][label]=serialization[i++];
			}
		 }
	 }
	 if (i!=serialization.size())
		 throw std::runtime_error("LPReparametrisationStorage<GM>::deserialize(): Size of serialization is greater than required for the graphical model! Deserialization failed.");
}

#ifdef WITH_HDF5
namespace hdf5 {

template<class GM>
void
save
(
	const LPReparametrisationStorage<GM> &repa,
	const std::string& filename,
	const std::string& modelname = "gm"
)
{
	const GM &gm = repa.graphicalModel();
	hid_t file = marray::hdf5::createFile(filename);
	hid_t group = marray::hdf5::createGroup(file, modelname);
	for (typename GM::IndexType i = 0; i < gm.numberOfFactors(); ++i) {
		if (gm[i].numberOfVariables() <= 1)
			continue;

		for (typename GM::IndexType j = 0; j < gm[i].numberOfVariables(); ++j) {
			std::stringstream s;
			s << i << "-" << j;
			marray::hdf5::save(group, s.str(), repa.get(i, j));
		}
	}
	marray::hdf5::closeGroup(group);
	marray::hdf5::closeFile(file);
}

template<class GM>
void
load
(
	LPReparametrisationStorage<GM> &repa,
	const std::string& filename,
	const std::string& modelname
)
{
	const GM &gm = repa.graphicalModel();
	hid_t file = marray::hdf5::openFile(filename);
	hid_t group = marray::hdf5::openGroup(file, modelname);
	for (typename GM::IndexType i = 0; i < gm.numberOfFactors(); ++i) {
		if (gm[i].numberOfVariables() <= 1)
			continue;

		for (typename GM::IndexType j = 0; j < gm[i].numberOfVariables(); ++j) {
			std::stringstream s;
			s << i << "-" << j;
			marray::hdf5::loadVec(group, s.str(), repa.get(i, j));
		}
	}
	marray::hdf5::closeGroup(group);
	marray::hdf5::closeFile(file);
};

}
#endif

template<class GM, class REPASTORAGE>
class ReparametrizationView : public opengm::FunctionBase<ReparametrizationView<GM,REPASTORAGE>,
typename GM::ValueType,typename GM::IndexType, typename GM::LabelType>
{
public:
	typedef typename GM::ValueType ValueType;
	typedef ValueType value_type;
	typedef typename GM::FactorType FactorType;
	typedef typename GM::OperatorType OperatorType;
	typedef typename GM::IndexType IndexType;
	typedef typename GM::LabelType LabelType;

	typedef GM GraphicalModelType;
	typedef REPASTORAGE ReparametrizationStorageType;

	ReparametrizationView(const FactorType & factor,const REPASTORAGE& repaStorage,IndexType factorId)
	:_pfactor(&factor),_prepaStorage(&repaStorage),_factorId(factorId)
	{};

	template<class Iterator>
	ValueType operator()(Iterator begin)const
	{
		switch (_pfactor->numberOfVariables())
		{
		case 1: return _prepaStorage->getVariableValue(_pfactor->variableIndex(0),*begin);
		default: return _prepaStorage->getFactorValue(_factorId,begin);
		};
	}

	LabelType shape(const IndexType& index)const{return _pfactor->numberOfLabels(index);};
	IndexType dimension()const{return _pfactor->numberOfVariables();};
	IndexType size()const{return _pfactor->size();};

private:
	const FactorType* _pfactor;
	const REPASTORAGE* _prepaStorage;
	IndexType _factorId;
};

struct LPReparametrizer_Parameter
{
	LPReparametrizer_Parameter(){};
};

template<class GM>
class LPReparametrizer
{
public:
	typedef GM GraphicalModelType;
	typedef typename GraphicalModelType::ValueType ValueType;
	typedef typename GraphicalModelType::IndexType IndexType;
	typedef typename GraphicalModelType::LabelType LabelType;
	typedef LPReparametrisationStorage<GM> RepaStorageType;
	typedef opengm::GraphicalModel<ValueType,opengm::Adder,opengm::ReparametrizationView<GM,RepaStorageType>,
					 opengm::DiscreteSpace<IndexType,LabelType> > ReparametrizedGMType;
	typedef LPReparametrizer_Parameter Parameter;

	LPReparametrizer(const GM& gm):_gm(gm),_repastorage(_gm){};
	virtual ~LPReparametrizer(){};
	RepaStorageType& Reparametrization(){return _repastorage;};
	const RepaStorageType& Reparametrization() const {return _repastorage;};
	virtual void reparametrize(){};
	virtual void getReparametrizedModel(ReparametrizedGMType& gm)const;
	const GM& graphicalModel()const{return _gm;}
protected:
	const GM& _gm;
	RepaStorageType _repastorage;
};

template<class GM>
void LPReparametrizer<GM>::getReparametrizedModel(ReparametrizedGMType& gm)const
{
	gm=ReparametrizedGMType(_gm.space());
	//copying factors
	for (typename GM::IndexType factorID=0;factorID<_gm.numberOfFactors();++factorID)
	{
		const typename GM::FactorType& f=_gm[factorID];
		opengm::ReparametrizationView<GM,RepaStorageType> repaView(f,_repastorage,factorID);
		typename ReparametrizedGMType::FunctionIdentifier fId=gm.addFunction(repaView);
		gm.addFactor(fId,f.variableIndicesBegin(), f.variableIndicesEnd());
	}
}

}//namespace


#endif /* LP_REPARAMETRIZATION_STORAGE_HXX_ */
