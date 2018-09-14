#!/bin/bash

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

# This script is OS X -specific.  For the shared libraries provided by nGraph,
# this script adjusts their install-names to follow OS X norms.
#
# This is required allow nGraph to successfully load certain plugins at runtime.
# It's intended to be run by the build system, not manually.

set -o errexit
set -o nounset

if [[ $# -ne 1 ]]; then
    echo "Usage: $(basename ${0}) <ngraph-lib-dir>" >&2
    exit 1
fi

# We'll work with library names that don't include the version number.  E.g., the
# symlink named 'libngraph.dylib' rather than the file 'libngraph.0.6.0.dylib'.
# This makes the script less brittle.
declare -a EXPECTED_LIBS=(
    "libcpu_backend.dylib"
    "libinterpreter_backend.dylib"
    "libiomp5.dylib"
    "libmkldnn.dylib"
    "libmklml.dylib"
    "libngraph.dylib"
    "libngraph_test_util.dylib"
    "libtbb.dylib"
    "libtbb_debug.dylib"
    )

#-------------------------------------------------------------------------------
# First, make sure everything looks as expected...
#-------------------------------------------------------------------------------
declare NGRAPH_LIB_DIR="${1}"
if [[ ! -d "${NGRAPH_LIB_DIR}" ]]; then
    echo "ERROR: The specified directory does not exist: '${NGRAPH_LIB_DIR}'" >&2
    exit 1
fi

declare IS_OK=1
declare F
for F in ${EXPECTED_LIBS[*]}; do
    declare F_PATH="${NGRAPH_LIB_DIR}/${F}"
    if [[ ! -f "${F_PATH}" ]]; then
        echo "ERROR: Expected library not found: '${F_PATH}'" >&2
        IS_OK=0
    fi
done
if [[ "${IS_OK}" != "1" ]]; then
    exit 1
fi

#-------------------------------------------------------------------------------
# Change each library as needed:
#    - The library's install name
#    - The library's recorded install name of each dependent-library
#    - The library's rpath value
#-------------------------------------------------------------------------------
for F in ${EXPECTED_LIBS[*]}; do
    declare F_PATH="${NGRAPH_LIB_DIR}/${F}"

    declare F_OLD_ID
    F_OLD_ID="$(/usr/bin/objdump -macho -dylib-id "${F_PATH}" | tail -1)"

    declare F_NEW_ID="@rpath/${F}"

    # Change the install name of this library.
    #
    # NOTE: It's theoretically possible for this to fail because the new id
    # is longer than the header can handle.  If we encounter that we'll need to
    # change our approach.
    /usr/bin/install_name_tool -id "${F_NEW_ID}" "${F_PATH}"

    # Fix up all of the shared libraries that depend upon F.
    # We don't have to be very specific, because 'install_name_tool -change'
    # simply does nothing when the "old" ID is absent...
    for OTHER_F in ${EXPECTED_LIBS[*]}; do
        if [[ "${F}" == "${OTHER_F}" ]]; then
            continue
        fi

        declare OTHER_F_PATH="${NGRAPH_LIB_DIR}/${OTHER_F}"
        /usr/bin/install_name_tool -change "${F_OLD_ID}" "${F_NEW_ID}" "${OTHER_F_PATH}"
    done
done

echo
echo "SUCCESS"
echo

