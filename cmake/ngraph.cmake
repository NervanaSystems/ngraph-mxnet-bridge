#*******************************************************************************
# Copyright 2018 Intel Corporation
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#*******************************************************************************

include(ExternalProject)

set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wno-comment")
set(CMAKE_SHARED_LINKER_FLAGS "${CMAKE_SHARED_LINKER_FLAGS} -Wl,--as-needed")

if(NGRAPH_EXTRA_CMAKE_FLAGS)
  string(REPLACE " " ";" NGRAPH_EXTRA_CMAKE_FLAGS ${NGRAPH_EXTRA_CMAKE_FLAGS})
else()
  set(NGRAPH_EXTRA_CMAKE_FLAGS "")
endif()

list(APPEND NGRAPH_EXTRA_CMAKE_FLAGS "-DCMAKE_BUILD_TYPE=${CMAKE_BUILD_TYPE}")
list(APPEND NGRAPH_EXTRA_CMAKE_FLAGS "-DCMAKE_C_COMPILER=${CMAKE_C_COMPILER}")
list(APPEND NGRAPH_EXTRA_CMAKE_FLAGS "-DCMAKE_CXX_COMPILER=${CMAKE_CXX_COMPILER}")
list(APPEND NGRAPH_EXTRA_CMAKE_FLAGS "-DCMAKE_INSTALL_PREFIX=${NGRAPH_INSTALL_PREFIX}")
list(APPEND NGRAPH_EXTRA_CMAKE_FLAGS "-DNGRAPH_DEX_ONLY=1")
list(APPEND NGRAPH_EXTRA_CMAKE_FLAGS "-DNGRAPH_UNIT_TEST_ENABLE=0")
list(APPEND NGRAPH_EXTRA_CMAKE_FLAGS "-DNGRAPH_TOOLS_ENABLE=0")

if (NGRAPH_TARGET_ARCH)
  list(APPEND NGRAPH_EXTRA_CMAKE_FLAGS "-DNGRAPH_TARGET_ARCH=${NGRAPH_TARGET_ARCH}")
endif()

if (NGRAPH_TUNE_ARCH)
  list(APPEND NGRAPH_EXTRA_CMAKE_FLAGS "-DNGRAPH_TUNE_ARCH=${NGRAPH_TUNE_ARCH}")
endif()

if (MKLDNN_INCLUDE_DIR AND MKLDNN_LIB_DIR)
  list(APPEND NGRAPH_EXTRA_CMAKE_FLAGS "-DMKLDNN_INCLUDE_DIR=${MKLDNN_INCLUDE_DIR}")
  list(APPEND NGRAPH_EXTRA_CMAKE_FLAGS "-DMKLDNN_LIB_DIR=${MKLDNN_LIB_DIR}")
  MESSAGE(STATUS "nGraph will use the MKLDNN library provided by MXnet")
  MESSAGE(STATUS "   MKLDNN_INCLUDE_DIR='${MKLDNN_INCLUDE_DIR}'")
  MESSAGE(STATUS "   MKLDNN_LIB_DIR='${MKLDNN_LIB_DIR}'")
elseif(MKLDNN_INCLUDE_DIR OR MKLDNN_LIB_DIR)
  MESSAGE(WARNING
"Just one of MKLDNN_INCLUDE_DIR or MKLDNN_LIB_DIR is set, so \
nGraph will build its own instance of libmkldnn.")

else()
  MESSAGE(STATUS "nGraph will NOT use the MKLDNN library provided by MXnet")
endif()

if(USE_NGRAPH_DISTRIBUTED)
  list(APPEND NGRAPH_EXTRA_CMAKE_FLAGS "-DNGRAPH_DISTRIBUTED_ENABLE=1")
endif(USE_NGRAPH_DISTRIBUTED)

if (USE_NGRAPH_GPU)
  list(APPEND NGRAPH_EXTRA_CMAKE_FLAGS "-DNGRAPH_GPU_ENABLE=TRUE")
endif()

ExternalProject_Add(
	ext_ngraph
	GIT_REPOSITORY https://github.com/NervanaSystems/ngraph.git
	GIT_TAG v0.12.0-rc.0
	PREFIX ngraph
	UPDATE_COMMAND ""
	CMAKE_ARGS ${NGRAPH_EXTRA_CMAKE_FLAGS}
	BUILD_ALWAYS 1
)

set(NGRAPH_INCLUDE_DIR ${NGRAPH_INSTALL_PREFIX}/include)
find_library(NGRAPH_LIB_DIR
    NAMES ngraph
    PATHS
    ${NGRAPH_INSTALL_PREFIX}/lib
    ${NGRAPH_INSTALL_PREFIX}/lib64
    NO_DEFAULT_PATH
)
include_directories(${NGRAPH_INCLUDE_DIR})
