#!/usr/bin/env bash

set -euo pipefail

usage() {
    cat <<'EOF'
Usage: ./configure.sh [options]
  -m, --mode TYPE       Debug|Release|RelWithDebInfo (default: Debug)
  -t, --tests           BUILD_TESTS=ON
  -e, --examples        BUILD_EXAMPLES=ON
      --asan            ENABLE_ASAN=ON
      --tsan            ENABLE_TSAN=ON
      --flame           ENABLE_FP=ON
  -p, --prefix PATH     Install prefix (default: /usr/local or env)
      --build           Build after configure
      --install         Install after build/configure
  -j, --jobs N          Parallel jobs for build
  -h, --help            Show this help
EOF
}

die(){ echo "Error: $*" >&2; exit 1; }
to_preset(){ case "$1" in Debug) echo debug;; Release) echo release;; RelWithDebInfo) echo relwithdebinfo;; *) die "unsupported mode '$1'";; esac; }

command -v cmake >/dev/null 2>&1 || die "cmake not found in PATH"

MODE="debug" BUILD_EXAMPLES=OFF BUILD_TESTS=OFF
ENABLE_ASAN=OFF ENABLE_TSAN=OFF ENABLE_FP=OFF
RUN_BUILD=OFF RUN_INSTALL=OFF JOBS=""
PYTHON_UDL_INTERFACE_PREFIX="${PYTHON_UDL_INTERFACE_PREFIX:-/usr/local}"

while [[ $# -gt 0 ]]; do
    case "$1" in
        -m|--mode) [[ $# -lt 2 ]] && die "missing value for $1"; MODE="$2"; shift ;;
        -t|--tests) BUILD_TESTS=ON ;;
        -e|--examples) BUILD_EXAMPLES=ON ;;
        --asan) ENABLE_ASAN=ON ;;
        --tsan) ENABLE_TSAN=ON ;;
        --flame) ENABLE_FP=ON ;;
        -p|--prefix) [[ $# -lt 2 ]] && die "missing value for $1"; PYTHON_UDL_INTERFACE_PREFIX="$2"; shift ;;
        --build) RUN_BUILD=ON ;;
        --install) RUN_INSTALL=ON ;;
        -j|--jobs) [[ $# -lt 2 ]] && die "missing value for $1"; JOBS="$2"; shift ;;
        -h|--help) usage; exit 0 ;;
        *) die "unknown parameter '$1'" ;;
    esac
    shift
done

[[ "$ENABLE_ASAN" == ON && "$ENABLE_TSAN" == ON ]] && die "--asan and --tsan cannot be used together"
[[ -n "$JOBS" && ! "$JOBS" =~ ^[0-9]+$ ]] && die "--jobs must be a non-negative integer"

pybind_arg=()
if command -v python3 >/dev/null 2>&1; then
    pybind_dir="$(python3 -m pybind11 --cmakedir 2>/dev/null || true)"
    [[ -n "$pybind_dir" && -d "$pybind_dir" ]] && pybind_arg=("-Dpybind11_DIR=$pybind_dir")
fi

echo "Configuring preset: $MODE"
cmake --preset "$MODE" \
    --install-prefix "$PYTHON_UDL_INTERFACE_PREFIX" \
    -DBUILD_EXAMPLES="$BUILD_EXAMPLES" \
    -DBUILD_TESTS="$BUILD_TESTS" \
    -DENABLE_ASAN="$ENABLE_ASAN" \
    -DENABLE_TSAN="$ENABLE_TSAN" \
    -DENABLE_FP="$ENABLE_FP" \
    "${pybind_arg[@]}"

if [[ "$RUN_BUILD" == ON || "$RUN_INSTALL" == ON ]]; then
    build_cmd=(cmake --build --preset "$MODE")
    [[ -n "$JOBS" ]] && build_cmd+=(--parallel "$JOBS")
    echo "${build_cmd[@]}"
    "${build_cmd[@]}"
fi

if [[ "$RUN_INSTALL" == ON ]]; then
    cmake --install "build/$MODE"
    echo "Done. Installed from build/$MODE"
elif [[ "$RUN_BUILD" == ON ]]; then
    echo "Done. Built preset $MODE"
else
    echo "Configuration complete. You can now run: cmake --build --preset $MODE"
fi
