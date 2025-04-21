#!/bin/bash

set -e

# Default values
BUILD_TYPE="Debug"
BUILD_DIR="build"

# Parse arguments
while [[ "$#" -gt 0 ]]; do
    case $1 in
    -t | --type)
        BUILD_TYPE="$2"
        shift
        ;;
    -d | --dir)
        BUILD_DIR="$2"
        shift
        ;;
    -h | --help)
        echo "Usage: ./configure.sh [-t Debug|Release|RelWithDebInfo|MinSizeRel] [-d build_dir]"
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
echo "  Build type : $BUILD_TYPE"
echo "  Build dir  : $BUILD_DIR"

# Create build directory
mkdir -p "$BUILD_DIR"
cd "$BUILD_DIR"

# Run CMake configuration
cmake .. -DCMAKE_BUILD_TYPE="$BUILD_TYPE"

echo "Configuration complete. You can now run: cmake --build $BUILD_DIR"
