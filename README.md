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
- **CMake** ≥ 3.14  
- **C++17** (or later)  
- **Python** ≥ 3.x development headers  
- **CUDAToolkit**

Package Dependencies
- [pybind11](https://github.com/pybind/pybind11)  
- [moodycamel::BlockingConcurrentQueue](https://github.com/cameron314/concurrentqueue)  
- [dmlc::dlpack](https://github.com/dmlc/dlpack.git)

Can be installed ob Ubuntu using `sudo apt install pybind11-dev libconcurrentqueue-dev libdlpack-dev`

## Installation
1. Initialize submodules `git submodule update --init --recursive`
2. We provide a simple `configure.sh` script to invoke CMake configuration. Use the `-h | --help` flag to see the possible options.
```bash
# Configure a Release build, including examples
./configure.sh -t Release -e
```
3. Install `sudo cmake --install build`
3. Uninstall `sudo xargs rm < build/install_manifest.txt`

The configure script also supports custom install prefix. You can either use the `-p | --prefix` flag, or specify the `PYTHON_UDL_INTERFACE_PREFIX` environment variable.
