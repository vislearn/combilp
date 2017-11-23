cmake_minimum_required(VERSION 3.7)
project(combip)

enable_language(C CXX)
set(CMAKE_CXX_STANDARD 14)

set(VERSION_MAJOR 0)
set(VERSION_MINOR 0)
set(VERSION_PATCH 0)

#
# combilp_opengm_stub
#

add_library(combilp_opengm_stub SHARED lib/combilp_opengm_stub.cxx)
target_include_directories(combilp_opengm_stub PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/external/opengm/include /usr/include/python3.6m)
target_link_libraries(combilp_opengm_stub PUBLIC hdf5)

#
# combilp_trws_stub
#

add_library(combilp_trws_stub SHARED
	${CMAKE_CURRENT_SOURCE_DIR}/lib/combilp_trws_stub.cxx
	${CMAKE_CURRENT_SOURCE_DIR}/external/part_opt/code/data/dataset.cpp
	${CMAKE_CURRENT_SOURCE_DIR}/external/part_opt/code/debug/logs.cpp
	${CMAKE_CURRENT_SOURCE_DIR}/external/part_opt/code/debug/performance.cpp
	${CMAKE_CURRENT_SOURCE_DIR}/external/part_opt/code/dynamic/block_allocator.cpp
	${CMAKE_CURRENT_SOURCE_DIR}/external/part_opt/code/dynamic/options.cpp
	${CMAKE_CURRENT_SOURCE_DIR}/external/part_opt/code/exttype/pvect.cpp
	${CMAKE_CURRENT_SOURCE_DIR}/external/part_opt/code/files/xfs.cpp
	${CMAKE_CURRENT_SOURCE_DIR}/external/part_opt/code/geom/vectn.cpp
	${CMAKE_CURRENT_SOURCE_DIR}/external/part_opt/code/maxflow/graph.cpp
	${CMAKE_CURRENT_SOURCE_DIR}/external/part_opt/code/maxflow/maxflow.cpp
	${CMAKE_CURRENT_SOURCE_DIR}/external/part_opt/code/optim/graph/mgraph.cpp
	${CMAKE_CURRENT_SOURCE_DIR}/external/part_opt/code/optim/part_opt/dee2.cpp
	${CMAKE_CURRENT_SOURCE_DIR}/external/part_opt/code/optim/part_opt/dee.cpp
	${CMAKE_CURRENT_SOURCE_DIR}/external/part_opt/code/optim/part_opt/energy.cpp
	${CMAKE_CURRENT_SOURCE_DIR}/external/part_opt/code/optim/part_opt/energy.cpp
	${CMAKE_CURRENT_SOURCE_DIR}/external/part_opt/code/optim/part_opt/msg_alg.cpp
	${CMAKE_CURRENT_SOURCE_DIR}/external/part_opt/code/optim/part_opt/part_opt_interface.cpp
	${CMAKE_CURRENT_SOURCE_DIR}/external/part_opt/code/optim/part_opt/part_opt_TRWS.cpp
	${CMAKE_CURRENT_SOURCE_DIR}/external/part_opt/code/optim/part_opt/trws_machine.cpp
	${CMAKE_CURRENT_SOURCE_DIR}/external/part_opt/code/optim/part_opt/vectorizers.cpp
	${CMAKE_CURRENT_SOURCE_DIR}/external/part_opt/code/optim/trws/stream_graph.cpp
	${CMAKE_CURRENT_SOURCE_DIR}/external/part_opt/code/streams/binary_stream.cpp
	${CMAKE_CURRENT_SOURCE_DIR}/external/part_opt/code/streams/file_stream.cpp
	${CMAKE_CURRENT_SOURCE_DIR}/external/part_opt/code/streams/text_stream.cpp
	${CMAKE_CURRENT_SOURCE_DIR}/external/part_opt/code/streams/xstringstream.cpp
)
target_include_directories(combilp_trws_stub PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/external/part_opt/code)

#
# combilp_srmp_stub
#

add_library(combilp_srmp_stub SHARED
	${CMAKE_CURRENT_SOURCE_DIR}/lib/combilp_srmp_stub.cxx
	${CMAKE_CURRENT_SOURCE_DIR}/external/srmp/src/Algs/AddEdges.cpp
	${CMAKE_CURRENT_SOURCE_DIR}/external/srmp/src/Algs/CMP.cpp
	${CMAKE_CURRENT_SOURCE_DIR}/external/srmp/src/Algs/debug.cpp
	${CMAKE_CURRENT_SOURCE_DIR}/external/srmp/src/Algs/energy.cpp
	${CMAKE_CURRENT_SOURCE_DIR}/external/srmp/src/Algs/MPLP.cpp
	${CMAKE_CURRENT_SOURCE_DIR}/external/srmp/src/Algs/order.cpp
	${CMAKE_CURRENT_SOURCE_DIR}/external/srmp/src/Algs/SRMP.cpp
	${CMAKE_CURRENT_SOURCE_DIR}/external/srmp/src/Algs/triplets.cpp
	${CMAKE_CURRENT_SOURCE_DIR}/external/srmp/src/FactorTypes/GeneralType.cpp
	${CMAKE_CURRENT_SOURCE_DIR}/external/srmp/src/FactorTypes/PairwiseDualType.cpp
	${CMAKE_CURRENT_SOURCE_DIR}/external/srmp/src/FactorTypes/PairwiseType.cpp
	${CMAKE_CURRENT_SOURCE_DIR}/external/srmp/src/FactorTypes/PatternType.cpp
	${CMAKE_CURRENT_SOURCE_DIR}/external/srmp/src/FactorTypes/PottsType.cpp
	${CMAKE_CURRENT_SOURCE_DIR}/external/srmp/src/FactorTypes/SharedPairwiseType.cpp
)
target_include_directories(combilp_srmp_stub PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/external/srmp/include)

#
# combilp_toulbar2_stub
#

add_library(combilp_toulbar2_stub SHARED
	${CMAKE_CURRENT_SOURCE_DIR}/lib/combilp_toulbar2_stub.cxx
	${CMAKE_CURRENT_SOURCE_DIR}/external/toulbar2/toulbar2/src/incop/autotuning2.cpp
	${CMAKE_CURRENT_SOURCE_DIR}/external/toulbar2/toulbar2/src/incop/csproblem.cpp
	${CMAKE_CURRENT_SOURCE_DIR}/external/toulbar2/toulbar2/src/incop/incopalgo.cpp
	${CMAKE_CURRENT_SOURCE_DIR}/external/toulbar2/toulbar2/src/incop/incoputil.cpp
	${CMAKE_CURRENT_SOURCE_DIR}/external/toulbar2/toulbar2/src/incop/narycsproblem.cpp
	${CMAKE_CURRENT_SOURCE_DIR}/external/toulbar2/toulbar2/src/incop/timer.cpp
	${CMAKE_CURRENT_SOURCE_DIR}/external/toulbar2/toulbar2/src/tb2abstractconstr.cpp
	${CMAKE_CURRENT_SOURCE_DIR}/external/toulbar2/toulbar2/src/tb2alldiffconstr.cpp
	${CMAKE_CURRENT_SOURCE_DIR}/external/toulbar2/toulbar2/src/tb2amongconstr.cpp
	${CMAKE_CURRENT_SOURCE_DIR}/external/toulbar2/toulbar2/src/tb2arithmetic.cpp
	${CMAKE_CURRENT_SOURCE_DIR}/external/toulbar2/toulbar2/src/tb2automaton.cpp
	${CMAKE_CURRENT_SOURCE_DIR}/external/toulbar2/toulbar2/src/tb2bep.cpp
	${CMAKE_CURRENT_SOURCE_DIR}/external/toulbar2/toulbar2/src/tb2binconstr.cpp
	${CMAKE_CURRENT_SOURCE_DIR}/external/toulbar2/toulbar2/src/tb2boostgraph.cpp
	${CMAKE_CURRENT_SOURCE_DIR}/external/toulbar2/toulbar2/src/tb2btd.cpp
	${CMAKE_CURRENT_SOURCE_DIR}/external/toulbar2/toulbar2/src/tb2btqueue.cpp
	${CMAKE_CURRENT_SOURCE_DIR}/external/toulbar2/toulbar2/src/tb2clusters.cpp
	${CMAKE_CURRENT_SOURCE_DIR}/external/toulbar2/toulbar2/src/tb2constraint.cpp
	${CMAKE_CURRENT_SOURCE_DIR}/external/toulbar2/toulbar2/src/tb2domain.cpp
	${CMAKE_CURRENT_SOURCE_DIR}/external/toulbar2/toulbar2/src/tb2dpglobalconstr.cpp
	${CMAKE_CURRENT_SOURCE_DIR}/external/toulbar2/toulbar2/src/tb2enumvar.cpp
	${CMAKE_CURRENT_SOURCE_DIR}/external/toulbar2/toulbar2/src/tb2flowbasedconstr.cpp
	${CMAKE_CURRENT_SOURCE_DIR}/external/toulbar2/toulbar2/src/tb2globalcardinalityconstr.cpp
	${CMAKE_CURRENT_SOURCE_DIR}/external/toulbar2/toulbar2/src/tb2globalconstr.cpp
	${CMAKE_CURRENT_SOURCE_DIR}/external/toulbar2/toulbar2/src/tb2globaldecomposable.cpp
	${CMAKE_CURRENT_SOURCE_DIR}/external/toulbar2/toulbar2/src/tb2grammarconstr.cpp
	${CMAKE_CURRENT_SOURCE_DIR}/external/toulbar2/toulbar2/src/tb2grammarutils.cpp
	${CMAKE_CURRENT_SOURCE_DIR}/external/toulbar2/toulbar2/src/tb2graph.cpp
	${CMAKE_CURRENT_SOURCE_DIR}/external/toulbar2/toulbar2/src/tb2haplotype.cpp
	${CMAKE_CURRENT_SOURCE_DIR}/external/toulbar2/toulbar2/src/tb2intervar.cpp
	${CMAKE_CURRENT_SOURCE_DIR}/external/toulbar2/toulbar2/src/tb2linearconstr.cpp
	${CMAKE_CURRENT_SOURCE_DIR}/external/toulbar2/toulbar2/src/tb2lpsconstr.cpp
	${CMAKE_CURRENT_SOURCE_DIR}/external/toulbar2/toulbar2/src/tb2maxconstr.cpp
	${CMAKE_CURRENT_SOURCE_DIR}/external/toulbar2/toulbar2/src/tb2mipsolver.cpp
	${CMAKE_CURRENT_SOURCE_DIR}/external/toulbar2/toulbar2/src/tb2naryconstr.cpp
	${CMAKE_CURRENT_SOURCE_DIR}/external/toulbar2/toulbar2/src/tb2pedigree.cpp
	${CMAKE_CURRENT_SOURCE_DIR}/external/toulbar2/toulbar2/src/tb2queue.cpp
	${CMAKE_CURRENT_SOURCE_DIR}/external/toulbar2/toulbar2/src/tb2randomgen.cpp
	${CMAKE_CURRENT_SOURCE_DIR}/external/toulbar2/toulbar2/src/tb2reader.cpp
	${CMAKE_CURRENT_SOURCE_DIR}/external/toulbar2/toulbar2/src/tb2regulardpconstr.cpp
	${CMAKE_CURRENT_SOURCE_DIR}/external/toulbar2/toulbar2/src/tb2regularflowconstr.cpp
	${CMAKE_CURRENT_SOURCE_DIR}/external/toulbar2/toulbar2/src/tb2sameconstr.cpp
	${CMAKE_CURRENT_SOURCE_DIR}/external/toulbar2/toulbar2/src/tb2solver.cpp
	${CMAKE_CURRENT_SOURCE_DIR}/external/toulbar2/toulbar2/src/tb2system.cpp
	${CMAKE_CURRENT_SOURCE_DIR}/external/toulbar2/toulbar2/src/tb2ternaryconstr.cpp
	${CMAKE_CURRENT_SOURCE_DIR}/external/toulbar2/toulbar2/src/tb2treeconstr.cpp
	${CMAKE_CURRENT_SOURCE_DIR}/external/toulbar2/toulbar2/src/tb2vac.cpp
	${CMAKE_CURRENT_SOURCE_DIR}/external/toulbar2/toulbar2/src/tb2vacutils.cpp
	${CMAKE_CURRENT_SOURCE_DIR}/external/toulbar2/toulbar2/src/tb2variable.cpp
	${CMAKE_CURRENT_SOURCE_DIR}/external/toulbar2/toulbar2/src/tb2wcsp.cpp
)
target_include_directories(combilp_toulbar2_stub PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/external/toulbar2/toulbar2/src)
set_target_properties(combilp_toulbar2_stub PROPERTIES COMPILE_DEFINITIONS "LONGDOUBLE_PROB;LONGLONG_COST;NARYCHAR;WCSPFORMATONLY;WIDE_STRING;LINUX")
target_link_libraries(combilp_toulbar2_stub gmp)
