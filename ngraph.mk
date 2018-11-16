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

####################################################################################################
# USAGE / OVERVIEW:
####################################################################################################
# Including this .mk file has the following effects:
#
# - If any problem is noticed, calls $(error ...).
#
# - Assuming no error is noticed:
#   - Sets the following variables (perhaps to the empty string):
#
#     NGRAPH_CFLAGS  - Compiler args needed to build the MXnet-nGraph bridge code.
#
#     NGRAPH_LDFLAGS_FOR_SHARED_LIBS - Static-linker flags for MXnet shared object(s) that
#        potentially require libngraph.so/.dylib at runtime, and which reside in MXnet's 'lib' directory.
#
#     NGRAPH_LDFLAGS_FOR_PROGS_IN_BIN - Static-linker flags for MXnet executables that
#        potentially require libngraph.so/.dylib at runtime, and which reside in MXnet's 'bin' directory.
#
#     NGRAPH_LDFLAGS_FOR_CPP_UNIT_TESTS_PROG - Static-linker flags for the program
#        build/tests/cpp/mxnet_unit_tests
#
#   - In whatever way is appropriate, defines new goals and adds them as
#     dependencies of the 'all' and/or 'clean' targets.
#
#   - Defines a target named 'ngraph' that represents *required* steps for the local build and
#     local installation of nGraph have been completed.  If USE_NGRAPH != 1, the 'ngraph'
#     target does nothing and is trivially satisfied.
#
# Input variables:
#   USE_NGRAPH
#   USE_NGRAPH_DISTRIBUTED
#   USE_NGRAPH_IE
#   MKLDNN_INCLUDE_DIR
#   MKLDNN_LIB_DIR
#   NGRAPH_EXTRA_CMAKE_FLAGS
#   NGRAPH_EXTRA_MAKE_FLAGS
#   ROOTDIR
#   USE_BLAS - One of MXnet's supported BLAS implementations:
#     "mkl", "blas", "atlas", "openblas", or "apple".
#   USE_CUDA - "0" or "1"

#===================================================================================================

# We use Bash to have access to certain safer constructs, such as '[[' vs. '['.
SHELL = /bin/bash

# Check for some configuration problems...
ifneq ($(NGRAPH_DIR),)
  $(warning "WARNING: MXnet's build system ignores the value of NGRAPH_DIR.")
endif

#===================================================================================================
ADD_NGRAPH_LIBDIR_TO_MXNET_RPATH=1
ifeq ($(DEBUG), 1)
	override NGRAPH_EXTRA_CMAKE_FLAGS += -DCMAKE_BUILD_TYPE=Debug
	override NGRAPH_EXTRA_CMAKE_FLAGS += -DNGRAPH_DEBUG_ENABLE=TRUE
endif
override NGRAPH_EXTRA_CMAKE_FLAGS += -DNGRAPH_UNIT_TEST_ENABLE=0 -DNGRAPH_TOOLS_ENABLE=0

ifneq ($(MKLDNN_INCLUDE_DIR),)
	override NGRAPH_EXTRA_CMAKE_FLAGS += -DMKLDNN_INCLUDE_DIR=$(MKLDNN_INCLUDE_DIR)
endif

ifneq ($(MKLDNN_LIB_DIR),)
	override NGRAPH_EXTRA_CMAKE_FLAGS += -DMKLDNN_LIB_DIR=$(MKLDNN_LIB_DIR)
endif

NGRAPH_EXTRA_MAKE_FLAGS="VERBOSE=1"

NGRAPH_SRC_DIR := $(ROOTDIR)/3rdparty/ngraph-mxnet-bridge
NGRAPH_BUILD_DIR := $(ROOTDIR)/3rdparty/ngraph-mxnet-bridge/build
NGRAPH_INSTALL_DIR := $(ROOTDIR)/3rdparty/ngraph-mxnet-bridge/build
MXNET_LIB_DIR := $(ROOTDIR)/lib

UNAME_S := $(shell uname -s)
ifeq ($(UNAME_S),Linux)
	OS := $(shell cat /etc/os-release)
	ifeq ($(findstring CentOS,$(OS)), CentOS)
		OS := "CentOS"
	endif
endif

ifeq ($(OS), "CentOS")
	NGRAPH_LIB_DIR := $(NGRAPH_INSTALL_DIR)/lib64
else
	NGRAPH_LIB_DIR := $(NGRAPH_INSTALL_DIR)/lib
endif

# The 'clean' target should remove nGraph-related generated files, regardless of whether or not
# the current Make invocation has USE_NGRAPH=1 ...
.PHONY: ngraph_clean
ngraph_clean:
	$(RM) -r "$(NGRAPH_BUILD_DIR)"
	$(RM) -r "$(NGRAPH_INSTALL_DIR)"

ifeq ($(USE_NGRAPH), 1)

.PHONY: ngraph
all: ngraph
ngraph: mkldnn
	mkdir -p "$(NGRAPH_BUILD_DIR)"
	@echo
	@echo ABOUT TO CONFIGURE AND BUILD 3rdparty/ngraph...
	@echo

	@# All of the following commands must run in the same subshell, because we want the 'cd'
	@# directory to be in effect for all of them.  Thus our use of ';' ...
	@if echo "$(NGRAPH_EXTRA_CMAKE_FLAGS)" | grep -q -e '-DCMAKE_INSTALL_PREFIX='; then \
	  echo; \
	  echo "It looks like NGRAPH_EXTRA_CMAKE_FLAGS is specifying a CMAKE_INSTALL_PREFIX value." \
	       " This is not supported." >&2; \
	  echo; \
	  exit 1; \
	  fi
	cd "$(NGRAPH_BUILD_DIR)"; \
	cmake "$(NGRAPH_SRC_DIR)" \
	  "-DCMAKE_MODULE_PATH=$(ROOTDIR)/cmake/Modules" \
	  "-DCMAKE_INSTALL_PREFIX=$(NGRAPH_INSTALL_DIR)" \
	  "-DUSE_CUDA=$(USE_CUDA)" \
	  "-DBLAS=$(USE_BLAS)" \
	  "-DUSE_NGRAPH_DISTRIBUTED=$(USE_NGRAPH_DISTRIBUTED)" \
	  "-DUSE_NGRAPH_GPU=$(USE_NGRAPH_GPU)" \
          "-DCUDA_INCLUDE_DIRS=$(USE_CUDA_PATH)/include" \
	  "-DNGRAPH_EXTRA_CMAKE_FLAGS='$(NGRAPH_EXTRA_CMAKE_FLAGS)'" \
	  $(NGRAPH_EXTRA_CMAKE_FLAGS); \
	  $(MAKE) all ngraph-mxnet-bridge $(NGRAPH_EXTRA_MAKE_FLAGS)

	# Copy contents of nGraph's 'lib' directory into MXnet's lib directory, taking care to
	# preserve the relative symlinks used to support Linux shared-object versioning.
	@echo
	@echo "COPYING 3rdparty/ngraph -SUPPLIED .SO/.DYLIB FILES INTO MXNET LIBS DIRECTORY..."
	@echo
	mkdir -p "$(MXNET_LIB_DIR)"
	@if [[ "$$(uname)" == "Darwin" ]]; then \
	    cd "$(NGRAPH_INSTALL_DIR)/lib"; tar -c -f - . | tar -x -f - -C "$(MXNET_LIB_DIR)"; \
	else \
	    cd "$(NGRAPH_LIB_DIR)"; tar c --to-stdout . | tar x --dereference --directory "$(MXNET_LIB_DIR)"; \
	fi

	@if [[ "$$(uname)" == "Darwin" ]]; then \
	    echo; \
	    echo "PATCHING MACH-O INSTALL NAMES, RPATH, ETC. FOR 3rdparty/ngraph -SUPPLIED .DYLIB FILES"; \
	    echo; \
	    $(ROOTDIR)/3rdparty/ngraph-mxnet-bridge/build-scripts/fixup-ngraph-osx-install-names.sh \
		"$(MXNET_LIB_DIR)"; \
	fi

	@echo
	@echo "3rdparty/ngraph INTERNAL BUILD/INSTALLATION SUCCESSFUL"
	@echo

NGRAPH_BRIDGE_SRC = $(wildcard $(ROOTDIR)/3rdparty/ngraph-mxnet-bridge/src/*.cc)
NGRAPH_BRIDGE_SRC += $(wildcard $(ROOTDIR)/3rdparty/ngraph-mxnet-bridge/src/ops/*.cc)
NGRAPH_BRIDGE_OBJ = $(patsubst %.cc,%.cc.o,$(patsubst $(ROOTDIR)/3rdparty/ngraph-mxnet-bridge/src/%,$(ROOTDIR)/3rdparty/ngraph-mxnet-bridge/build/src/CMakeFiles/ngraph-mxnet-bridge.dir/%,$(NGRAPH_BRIDGE_SRC)))
$(NGRAPH_BRIDGE_OBJ): %.o: ngraph $(NGRAPH_BRIDGE_SRC)

  # Set NGRAPH_CFLAGS ...
  NGRAPH_CFLAGS = \
    "-I$(NGRAPH_INSTALL_DIR)/include" \
    "-I$(ROOTDIR)/3rdparty/ngraph-mxnet-bridge/src" \
    -DMXNET_USE_NGRAPH=1

  ifeq ($(USE_NGRAPH_IE),1)
    NGRAPH_CFLAGS += -DMXNET_USE_NGRAPH_IE=1
  endif

  ifeq ($(USE_CUDA), 1)
    NGRAPH_CFLAGS += -DMXNET_USE_CUDA=1
  endif

  ifeq ($(USE_NGRAPH_DISTRIBUTED), 1)
    NGRAPH_CFLAGS += -DMXNET_USE_NGRAPH_DISTRIBUTED=1
  endif

  # nGraph provides some libraries that may compete with other libraries already installed
  # on the system. This provides the link-flags that, if provided early enough on the static-linker
  # command line, should ensure that the nGraph-supplied version is preferred.
  ifeq ("$(shell uname)","Darwin")
    NGRAPH_MKLML_LIB_ := mklml
  else
    NGRAPH_MKLML_LIB_ := mklml_intel
  endif

  NGRAPH_COMMON_LIBRARY_LDFLAGS_ := \
    -ltbb \
    -liomp5 \
    -l$(NGRAPH_MKLML_LIB_) \
    -lmkldnn

  ifeq ("$(shell uname)","Darwin")
    NGRAPH_LDFLAGS_ := \
      -L$(MXNET_LIB_DIR) \
      $(NGRAPH_COMMON_LIBRARY_LDFLAGS_) \
      -lngraph \
      -lcpu_backend

    ifeq ($(USE_NGRAPH_DISTRIBUTED), 1)
      $(error "USE_NGRAPH_DISTRIBUTED=1 is not yet supported on OS X.")
    endif

    # The static-link-time flags for creating .so/.dylib files that will dynamically link to nGraph's libs
    # at runtime.
    NGRAPH_LDFLAGS_FOR_SHARED_LIBS := \
      $(NGRAPH_LDFLAGS_) \
      -Wl,-rpath,@loader_path

    NGRAPH_LDFLAGS_FOR_PROGS_IN_BIN := \
      $(NGRAPH_LDFLAGS_) \
      -Wl,-rpath,@loader_path/../lib

    NGRAPH_LDFLAGS_FOR_CPP_UNIT_TESTS_PROG := \
      $(NGRAPH_LDFLAGS_) \
      -Wl,-rpath,@loader_path/../../../lib
  else
    NGRAPH_LDFLAGS_ := \
      -L$(MXNET_LIB_DIR) \
      -Wl,-rpath-link=$(MXNET_LIB_DIR) \
      $(NGRAPH_COMMON_LIBRARY_LDFLAGS_) \
      -lngraph \
      -lcpu_backend

    ifeq ($(USE_NGRAPH_DISTRIBUTED), 1)
      NGRAPH_LDFLAGS_ += $(shell mpicxx --showme:link)
    endif

    # The static-link-time flags for creating .so/.dylib files that will dynamically link to nGraph's libs
    # at runtime.
    NGRAPH_LDFLAGS_FOR_SHARED_LIBS := \
      $(NGRAPH_LDFLAGS_) \
      -Wl,-rpath='$${ORIGIN}' \
      -Wl,--as-needed

    NGRAPH_LDFLAGS_FOR_PROGS_IN_BIN := \
      $(NGRAPH_LDFLAGS_) \
      -Wl,-rpath='$${ORIGIN}/../lib' \
      -Wl,--as-needed

    NGRAPH_LDFLAGS_FOR_CPP_UNIT_TESTS_PROG := \
      $(NGRAPH_LDFLAGS_) \
      -Wl,-rpath='$${ORIGIN}/../../../lib' \
      -Wl,--as-needed
  endif

else
  #-------------------------------------------------------------------------------------------------
  # USE_NGRAPH != 1
  #-------------------------------------------------------------------------------------------------
  NGRAPH_CFLAGS :=
  NGRAPH_LDFLAGS_FOR_SHARED_LIBS :=
  NGRAPH_LDFLAGS_FOR_PROGS_IN_BIN :=
  NGRAPH_LDFLAGS_FOR_CPP_UNIT_TESTS_PROG :=

  # The 'ngraph' target is a no-op.
ngraph:

endif

