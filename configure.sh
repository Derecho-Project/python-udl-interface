#!/bin/bash

set -e

# Default values
BUILD_TYPE="Debug"
BUILD_DIR="build"
BUILD_EXAMPLES=OFF
BUILD_TESTS=OFF

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
    -h | --help)
        echo "Usage: ./configure.sh [-t Debug|Release|RelWithDebInfo|MinSizeRel] [-d build_dir] [-e]"
        echo "  -m | --mode      : Build type (default: Debug)"
        echo "  -d | --dir       : Build directory (default: build)"
        echo "  -e | --examples  : Enable building examples (default: OFF)"
        echo "  -t | --tests     : Enable building tests (default: OFF)"
        exit 0
        ;;
    *)
        echo "Unknown parameter: $1"
        exit 1
        ;;
    esac
    shift
done

echo "Configuring project..."
echo "  Build type     : $BUILD_TYPE"
echo "  Build dir      : $BUILD_DIR"
echo "  Build examples : $BUILD_EXAMPLES"
echo "  Build tests    : $BUILD_TESTS"

# Create build directory
mkdir -p "$BUILD_DIR"

# Run CMake configuration
cmake -B "$BUILD_DIR" \
    -DCMAKE_BUILD_TYPE="$BUILD_TYPE" \
    -DBUILD_EXAMPLES="$BUILD_EXAMPLES" \
    -DBUILD_TESTS="$BUILD_TESTS" \
    -S .

echo "Configuration complete. You can now run: cmake --build $BUILD_DIR"
