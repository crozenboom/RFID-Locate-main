#!/bin/bash -eu

#
# Example build script for use with Impinj ETK
#
# The target libraries are contained inside the sysroot at
# "$ETK_ROOT/arm-buildroot-linux-gnueabihf/sysroot/lib".  The host libraries
# are contained in "$ETK_ROOT/lib".
#
# For a host build, CMake must be configured to correctly search for host
# headers and libraries in the correct location inside the ETK.
#
# For a target build, CMake must be configured to use the correct cross
# compiler and informed about the sysroot location in the ETK.
#
# The script allows overriding some variables when building, e.g.:
#
#   [user@machine]$ BUILD_TYPE=Release VERBOSE=ON ./build.sh
#

SCRIPT_PATH=$(dirname $(realpath $0))
OUTPUT_PATH=${SCRIPT_PATH}/output

# defaults when building with ETK; override on command line for custom setup
ETK_PATH=${ETK_PATH:-$(realpath ${SCRIPT_PATH}/../../arm-toolchain)}
BUILD_TYPE=${BUILD_TYPE:-Debug}
VERBOSE=${VERBOSE:-OFF}

# host build; override on command line for custom setup
INC=${INC:-${ETK_PATH}/include}
LIB=${LIB:-${ETK_PATH}/lib}
cmake ${SCRIPT_PATH} -B"${OUTPUT_PATH}/host" \
    -DCMAKE_VERBOSE_MAKEFILE:BOOL=${VERBOSE} \
    -DCMAKE_BUILD_TYPE=${BUILD_TYPE} \
    -DCMAKE_CXX_FLAGS="-isystem ${INC}" \
    -DCMAKE_LIBRARY_PATH=${LIB}
cmake --build "${OUTPUT_PATH}/host" -j

# target build
CC=$(find ${ETK_PATH}/bin -name arm-*-linux-*-gcc)
CC=${CC%gcc}
SYSROOT_PATH="$(${CC}gcc -print-sysroot)"
cmake ${SCRIPT_PATH} -B"${OUTPUT_PATH}/target" \
    -DCMAKE_VERBOSE_MAKEFILE:BOOL=${VERBOSE} \
    -DCMAKE_BUILD_TYPE=${BUILD_TYPE} \
    -DCMAKE_C_COMPILER=${CC}gcc \
    -DCMAKE_CXX_COMPILER=${CC}g++ \
    -DCMAKE_FIND_ROOT_PATH=${SYSROOT_PATH}
cmake --build "${OUTPUT_PATH}/target" -j
