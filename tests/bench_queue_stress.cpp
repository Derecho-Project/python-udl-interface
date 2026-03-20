#include "pyscheduler/pyscheduler.hpp"
#include <benchmark/benchmark.h>

#include <cstdint>
#include <future>
#include <random>
#include <vector>

#if defined(PYSCHEDULER_TEST_HAS_CUDA) && PYSCHEDULER_TEST_HAS_CUDA &&                             \
	__has_include(<cuda_runtime.h>) && __has_include(<dlpack/dlpack.h>)
#	include "pyscheduler/tensor.hpp"
#	include <cuda_runtime.h>
#	include <dlpack/dlpack.h>
#	include <pybind11/pytypes.h>
#endif

using namespace pyscheduler;

namespace {
PyManager& getManager() {
	static PyManager manager;
	return manager;
}

#if defined(PYSCHEDULER_TEST_HAS_CUDA) && PYSCHEDULER_TEST_HAS_CUDA &&                             \
	__has_include(<cuda_runtime.h>) && __has_include(<dlpack/dlpack.h>)
std::vector<float> generateRandomMatrix(int rows, int cols, uint32_t seed) {
	std::mt19937 rng(seed);
	std::uniform_real_distribution<float> dist(-1.0f, 1.0f);
	std::vector<float> out(static_cast<size_t>(rows * cols));
	for(float& v : out) {
		v = dist(rng);
	}
	return out;
}

pybind11::capsule toDlpackCapsule(DLManagedTensor* managed_tensor) {
	return pybind11::capsule(managed_tensor, "dltensor", [](PyObject* capsule_ptr) {
		auto* tensor =
			reinterpret_cast<DLManagedTensor*>(PyCapsule_GetPointer(capsule_ptr, "dltensor"));
		if(tensor != nullptr && tensor->deleter != nullptr) {
			tensor->deleter(tensor);
		}
	});
}
#endif
} // namespace

static void BM_QS_CPU(benchmark::State& state) {
	const int64_t entries = state.range(0);
	const size_t batch_size = static_cast<size_t>(state.range(1));
	const size_t prefetch_depth = static_cast<size_t>(state.range(2));

	auto commit = [](int val) -> pybind11::object { return pybind11::cast(val); };
	auto callback = [](const pybind11::object& obj) { return obj.cast<int>(); };

	for(auto _ : state) {
		state.PauseTiming();
		PyManager::InvokeHandler reflect = getManager().loadPythonModule(
			"tests.test_modules.identity", "invoke", batch_size, prefetch_depth);
		state.ResumeTiming();

		std::vector<std::future<int>> futures;
		futures.reserve(static_cast<size_t>(entries));

		for(int64_t i = 0; i < entries; i++) {
			futures.push_back(reflect.queue_invoke(commit, callback, static_cast<int>(i)));
		}

		int64_t checksum = 0;
		for(auto& f : futures) {
			checksum += f.get();
		}

		benchmark::DoNotOptimize(checksum);
		benchmark::ClobberMemory();
	}

	state.SetItemsProcessed(state.iterations() * entries);
}

BENCHMARK(BM_QS_CPU)
	->ArgNames({ "n", "batch", "prefetch" })
	->Args({ 120000, 256, 64 })
	->Args({ 120000, 512, 128 })
	->Args({ 120000, 1024, 256 })
	->Unit(benchmark::kMillisecond)
	->Iterations(1);

#if defined(PYSCHEDULER_TEST_HAS_CUDA) && PYSCHEDULER_TEST_HAS_CUDA &&                             \
	__has_include(<cuda_runtime.h>) && __has_include(<dlpack/dlpack.h>)
static void BM_QS_GpuMatmul(benchmark::State& state) {
	const int64_t entries = state.range(0);
	const int dim = static_cast<int>(state.range(1));
	const size_t batch_size = static_cast<size_t>(state.range(2));
	const size_t prefetch_depth = static_cast<size_t>(state.range(3));

	int device_count = 0;
	if(cudaGetDeviceCount(&device_count) != cudaSuccess || device_count == 0) {
		state.SkipWithError("No CUDA device available");
		return;
	}

	PyManager::InvokeHandler has_cuda =
		getManager().loadPythonModule("tests.test_modules.dense_matmul", "has_cuda");
	if(!has_cuda.invoke<bool>()) {
		state.SkipWithError("PyTorch CUDA unavailable in Python environment");
		return;
	}

	auto commit = [dim](uint32_t seed) -> pybind11::object {
		auto a_host = generateRandomMatrix(dim, dim, seed);
		auto b_host = generateRandomMatrix(dim, dim, seed ^ 0x9e3779b9u);

		pybind11::gil_scoped_acquire gil;
		auto a_capsule = toDlpackCapsule(createCudaMatrixDlpack(a_host, dim, dim));
		auto b_capsule = toDlpackCapsule(createCudaMatrixDlpack(b_host, dim, dim));
		return pybind11::make_tuple(std::move(a_capsule), std::move(b_capsule));
	};

	auto callback = [](pybind11::object&& obj) { return obj.cast<double>(); };

	for(auto _ : state) {
		state.PauseTiming();
		PyManager::InvokeHandler matmul =
			getManager().loadPythonModule("tests.test_modules.dense_matmul",
										  "dense_matmul_from_dlpack_batch_sum",
										  batch_size,
										  prefetch_depth);
		state.ResumeTiming();

		std::vector<std::future<double>> futures;
		futures.reserve(static_cast<size_t>(entries));

		for(int64_t i = 0; i < entries; i++) {
			futures.push_back(matmul.queue_invoke(commit, callback, static_cast<uint32_t>(i + 1)));
		}

		double checksum = 0.0;
		for(auto& f : futures) {
			checksum += f.get();
		}

		benchmark::DoNotOptimize(checksum);
		benchmark::ClobberMemory();
	}

	state.SetItemsProcessed(state.iterations() * entries);
}

BENCHMARK(BM_QS_GpuMatmul)
	->ArgNames({ "n", "dim", "batch", "prefetch" })
	->Args({ 4000, 128, 32, 1 })
	->Args({ 4000, 128, 32, 2 })
	->Args({ 4000, 128, 32, 3 })
	->Args({ 4000, 128, 64, 1 })
	->Args({ 4000, 128, 64, 2 })
	->Args({ 4000, 128, 64, 3 })
	->Unit(benchmark::kMillisecond)
	->Iterations(1);
#endif
