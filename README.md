# Pyscheduler

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

A C++ library that embeds a single Python interpreter, providing thread‑pooled, asynchronous, and synchronous invocation of Python functions from C++ with optimized GIL usage.

Optimized as an execution engine for Machine Learning pipelines. 

## Table of Contents

- [Features](#features)  
- [Requirements](#requirements)  
- [Installation](#installation)  
<!-- - [Quick Start](#quick-start)  
- [API Reference](#api-reference)  
  - [PyManager](#pyschedulerpymanager)  
  - [InvokeHandler](#pyschedulerpymanagerinvokehandler)  
- [Examples](#examples)  
- [Configuration](#configuration)  
- [Contributing](#contributing)  
- [License](#license)   -->

## Features

- **Thread Safe Implementation** 
  Ensures only one Python interpreter is ever initialized per process for pre python3.13. 
    - [x] Pre Python 3.13 
    - [ ] Python 3.13+ no-gil sub interpreter 
- **Thread‑Pooled Execution**  
  Uses a high‑performance round robin queue to minimize latency during high throughput workloads
- **Optimized GIL Management**  
  Acquires/releases the Global Interpreter Lock only around actual Python execution.
- **Synchronous & Asynchronous APIs**  
  Easily call Python functions synchronously or schedule them with callbacks returning `std::future`.
- **Opportunistic Batching** 
  Batches similar workloads together to minimize up-call latencies into Python.

## Requirements
System Dependencies
- **CMake** ≥ 3.27  
- **C++20** (or later)  
- **Python** ≥ 3.x development headers  
- **CUDAToolkit**

## Installation
Can add to your project as a submodule or install as a system library.

1. Clone repository: `git clone --recurse-submodules`
2. Configure and build with `configure.sh` (use `-h | --help` for all options).

```bash
# Configure + build default debug build
./configure.sh --build

# Configure + build tests with ASan only
./configure.sh --tests --asan --build

# Configure + build tests with TSan only
./configure.sh --tests --tsan --build

# Configure + build release
./configure.sh --mode release --build
```

3. Install with either:
```bash
./configure.sh --mode Release --build --install
# or
cmake --install build-Release
```
4. Uninstall `sudo xargs rm < build-Release/install_manifest.txt`

The configure script also supports custom install prefix. You can either use the `-p | --prefix` flag, or specify the `PYTHON_UDL_INTERFACE_PREFIX` environment variable.

## Architecture

Pyscheduler is designed to manage the lifecycle of embedding Python within a C++ multithreaded environment. At its core, the library aims to reduce the overhead imposed by Python's Global Interpreter Lock (GIL), allowing for concurrent C++ execution before transitioning data to the Python runtime.

### Core Components

**`PyManager`**  
The primary lifecycle manager for the embedded Python environment. It safely scopes the interpreter's initialization and finalization, ensuring that Python routines are accessible across the C++ application. `PyManager` provides thread-safety when starting the Python runtime and serves as the factory for creating specialized function handlers.

**`InvokeHandler`**  
An execution interface bound to a specific Python function. Handlers isolate execution pipelines, allowing the library to apply localized optimizations like per-handler queuing, opportunistic batching, and data prefetching without impacting other Python tasks.

### Execution Lifecycle and Optimization

A key design principle of Pyscheduler is decoupling C++ state preparation from Python execution. 

1. **Commit Phase**: When work is dispatched via `queue_invoke`, Pyscheduler requests a user-provided commit function. The commit function is called ahead of execution to provide the user with an opportunity to pre-process data e.g. load memory onto the GPU.
2. **Execution Phase**: Once the commit phase structures the C++ arguments into Python-accessible objects (using `pybind11`), the handler acquires the GIL and submits the batched payload to the underlying Python interpreter.
3. **Callback Phase**: Results yielded from the Python function are handled by a C++ callback, returning the computed outcomes to the caller asynchronously via a standard `std::future`.

### Workflow Example

A typical pipeline consists of initializing the runtime, establishing a handler, and dispatching work:

```cpp
// 1. Initialize the global Python manager
PyManager manager;

// 2. Bind a specific Python function to a handler
PyManager::InvokeHandler add_handler = manager.loadPythonModule("math_ops", "add");

// 3. Synchronous invocation
int sync_result = add_handler.invoke<int>(arg1, arg2);

// 4. Asynchronous invocation with GIL-free staging
auto commit_fn = [](int a, int b) { 
    // Prepared without the GIL lock
    return pybind11::make_tuple(a, b); 
};
auto callback_fn = [](pybind11::object&& result) { 
    // Resolves output from Python execution
    return pybind11::cast<int>(result); 
};

std::future<int> async_result = add_handler.queue_invoke(commit_fn, callback_fn, arg1, arg2);
```

## Examples

### Synchronous invoke

Call a Python function directly and cast the result.

**Python** (`my_module.py`):
```python
def invoke(a, b):
    return a + b
```

**C++**:
```cpp
pyscheduler::PyManager manager;
auto add = manager.loadPythonModule("my_module", "invoke");

int64_t result = add.invoke<int64_t>(3000, -1234);  // 1766
```

### Synchronous invoke with callback

Process the raw `pybind11::object` before it leaves the GIL scope.

```cpp
auto to_string = [](const pybind11::object& obj) { return obj.cast<std::string>(); };

std::string result = add.invoke(to_string, "hello", " world");  // "hello world"
```

### Asynchronous invoke with batching

Queue work items that get batched into a single Python call. The `commit` function converts C++ arguments into a `pybind11::object`, and the `callback` processes each result.

**Python** (`svd.py`):
```python
import numpy as np

def generate(items):
    return [np.random.rand(n, n) for n in items]

def compute_svd_rank(matrices):
    def svd_one(M):
        S = np.linalg.svd(M, compute_uv=False)
        tol = np.max(M.shape) * np.spacing(S.max())
        return int(np.sum(S > tol))
    return [svd_one(M) for M in matrices]
```

**C++**:
```cpp
pyscheduler::PyManager manager;

// batch_size=1 (default) — one matrix per Python call
auto generate = manager.loadPythonModule("svd", "generate");

// batch_size=32, prefetch_depth=3 — up to 32 matrices per call, 96 pre-committed
auto compute = manager.loadPythonModule("svd", "compute_svd_rank", 32, 3);

// Step 1: generate 1000 random matrices
auto commit = [](int n) -> pybind11::object { return pybind11::cast(n); };
auto extract = [](pybind11::object&& obj) { return obj.release().ptr(); };

std::vector<std::future<PyObject*>> gen_futures;
for (int i = 0; i < 1000; i++)
    gen_futures.push_back(generate.queue_invoke(commit, extract, 100));

// Collect raw PyObject* pointers (ownership transferred out of pybind11)
std::vector<PyObject*> matrices;
for (auto& f : gen_futures)
    matrices.push_back(f.get());

// Step 2: compute SVD rank of each matrix (batched 32 at a time)
auto commit2 = [](PyObject* p) -> pybind11::object {
    return pybind11::reinterpret_steal<pybind11::object>(p);
};
auto to_int = [](pybind11::object&& obj) { return obj.cast<int>(); };

std::vector<std::future<int>> rank_futures;
for (int i = 0; i < 1000; i++)
    rank_futures.push_back(compute.queue_invoke(commit2, to_int, matrices[i]));

for (auto& f : rank_futures)
    std::cout << f.get() << "\n";
```
