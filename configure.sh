#!/bin/bash

set -e

# Default values
BUILD_TYPE="Debug"
BUILD_DIR="build"
BUILD_EXAMPLES=OFF
BUILD_TESTS=OFF
PYTHON_UDL_INTERFACE_PREFIX="${PYTHON_UDL_INTERFACE_PREFIX:-/usr/local}"

ENABLE_GPROF=OFF
ENABLE_ASAN=OFF
ENABLE_TSAN=OFF
ENABLE_FP=OFF

# Parse arguments
while [[ "$#" -gt 0 ]]; do
    case $1 in
    -m | --mode)
        BUILD_TYPE="$2"
        shift
        ;;
    -d | --dir)
        BUILD_DIR="$2"
        shift
        ;;
    -e | --examples)
        BUILD_EXAMPLES=ON
        ;;
    -t | --tests)
        BUILD_TESTS=ON
        ;;
    -p | --prefix)
        PYTHON_UDL_INTERFACE_PREFIX="$2"
        shift
        ;;
    --gprof)
        ENABLE_GPROF=ON
        ;;
    --asan)
        ENABLE_ASAN=ON
        ;;
    --tsan)
        ENABLE_TSAN=ON
        ;;
    --flame)
        ENABLE_FP=ON
        ;;
    -h | --help)
        echo "Usage: ./configure.sh [options]"
        echo "  -m | --mode      : Build type (Debug/Release/RelWithDebInfo)"
        echo "  -d | --dir       : Build directory (default: build)"
        echo "  -e | --examples  : Enable building examples"
        echo "  -t | --tests     : Enable building tests"
        echo "  -p | --prefix    : Install prefix"
        echo "  --gprof          : Enable gprof profiling"
        echo "  --asan           : Enable AddressSanitizer"
        echo "  --tsan           : Enable ThreadSanitizer"
        echo "  --flame          : Enable frame pointer for flamegraph generation"
        exit 0
        ;;
    *)
        echo "Unknown parameter: $1"
        exit 1
        ;;
    esac
    shift
done

# Summary
BUILD_DIR="${BUILD_DIR}-${BUILD_TYPE}"

echo "Configuring project..."
echo "  Build type     : $BUILD_TYPE"
echo "  Build dir      : $BUILD_DIR"
echo "  Build examples : $BUILD_EXAMPLES"
echo "  Build tests    : $BUILD_TESTS"
echo "  Install prefix : $PYTHON_UDL_INTERFACE_PREFIX"
echo "  ENABLE_GPROF   : $ENABLE_GPROF"
echo "  ENABLE_ASAN    : $ENABLE_ASAN"
echo "  ENABLE_TSAN    : $ENABLE_TSAN"
echo "  ENABLE_FP      : $ENABLE_FP"

# Create build directory
mkdir -p "$BUILD_DIR"

# Run CMake configuration
cmake -B "$BUILD_DIR" \
    -DCMAKE_BUILD_TYPE="$BUILD_TYPE" \
    -DCMAKE_INSTALL_PREFIX="$PYTHON_UDL_INTERFACE_PREFIX" \
    -DBUILD_EXAMPLES="$BUILD_EXAMPLES" \
    -DBUILD_TESTS="$BUILD_TESTS" \
    -DENABLE_GPROF="$ENABLE_GPROF" \
    -DENABLE_ASAN="$ENABLE_ASAN" \
    -DENABLE_TSAN="$ENABLE_TSAN" \
    -DENABLE_FP="$ENABLE_FP" \
    -S .

echo "Configuration complete. You can now run: cmake --build $BUILD_DIR"
