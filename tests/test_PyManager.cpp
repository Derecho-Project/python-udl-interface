#include "pyscheduler/pyscheduler.hpp"
#include "pyscheduler/tensor.hpp"
#include <catch2/catch_all.hpp>
#include <pybind11/pytypes.h>
#include <pybind11/stl.h>

#include <chrono>
#include <cmath>
#include <thread>

#if defined(PYSCHEDULER_TEST_HAS_CUDA) && PYSCHEDULER_TEST_HAS_CUDA &&                             \
	__has_include(<cuda_runtime.h>) && __has_include(<dlpack/dlpack.h>)
#	if __has_include(<cblas.h>)
#		include <cblas.h>
#	elif __has_include(<openblas/cblas.h>)
#		include <openblas/cblas.h>
#	else
#		error "CBLAS header not found"
#	endif
#	include <cuda_runtime.h>
#	include <dlpack/dlpack.h>
#	include <cstring>
#	include <random>
#	include <vector>
#endif

using namespace pyscheduler;

struct Context {
	PyManager manager;

	Context() {
		manager.add_path(PYSCHEDULER_SOURCE_DIR);
	}
};

Context& getContext() {
	static Context context;
	return context;
}

TEST_CASE("Load module", "[basic]") {
	PyManager& manager = getContext().manager;
	PyManager::InvokeHandler reflect = manager.loadPythonModule("tests.test_modules.identity");
}

TEST_CASE("Throws exception if module dne", "[basic]") {
	PyManager& manager = getContext().manager;
	REQUIRE_THROWS_AS(manager.loadPythonModule("tests.test_modules.does_not_exist"),
					  std::invalid_argument);
}

TEST_CASE("Throws exception if handler dne", "[basic]") {
	PyManager& manager = getContext().manager;
	REQUIRE_THROWS_AS(manager.loadPythonModule("tests.test_modules.identity2"),
					  std::invalid_argument);
}

TEST_CASE("Synchronous invoke with cast", "[basic]") {
	PyManager& manager = getContext().manager;
	PyManager::InvokeHandler reflect = manager.loadPythonModule("tests.test_modules.identity");

	REQUIRE(reflect.invoke<std::string>("hello") == "hello");
	REQUIRE(reflect.invoke<double>(3.1415926535) == 3.1415926535);
}

TEST_CASE("Synchronous invoke with closure", "[basic]") {
	auto to_string = [](const pybind11::object& obj) { return obj.cast<std::string>(); };
	auto to_double = [](const pybind11::object& obj) { return obj.cast<double>(); };

	PyManager& manager = getContext().manager;
	PyManager::InvokeHandler reflect = manager.loadPythonModule("tests.test_modules.identity");

	REQUIRE(reflect.invoke(to_string, "hello") == "hello");
	REQUIRE(reflect.invoke(to_double, 3.1415926535) == 3.1415926535);
}

TEST_CASE("Asynchronous invoke", "[basic]") {
	auto commit = [](double val) -> pybind11::object { return pybind11::cast(val); };
	auto callback = [](const pybind11::object& obj) { return obj.cast<double>(); };

	PyManager& manager = getContext().manager;
	PyManager::InvokeHandler reflect = manager.loadPythonModule("tests.test_modules.identity");

	std::vector<std::future<double>> futures;
	for(size_t i = 0; i < 5; i++) {
		futures.push_back(reflect.queue_invoke(commit, callback, 3.1415926535));
	}

	for(auto& future : futures) {
		REQUIRE(future.get() == 3.1415926535);
	}
}

TEST_CASE("Swap only works for non-committed queued requests", "[queue-control]") {
	auto commit = [](int val, double sleep_s) -> pybind11::object {
		return pybind11::make_tuple(val, sleep_s);
	};
	auto callback = [](const pybind11::object& obj) { return obj.cast<int>(); };

	PyManager& manager = getContext().manager;
	PyManager::InvokeHandler handler =
		manager.loadPythonModule("tests.test_modules.black_hole", "invoke", 1, 1);

	const auto id1 = static_cast<PyManager::InvokeHandler::RequestId>(2001);
	const auto id2 = static_cast<PyManager::InvokeHandler::RequestId>(2002);
	const auto id3 = static_cast<PyManager::InvokeHandler::RequestId>(2003);
	auto f1 = handler.queue_invoke_with_id(id1, commit, callback, 1, 0.4);
	auto f2 = handler.queue_invoke_with_id(id2, commit, callback, 2, 0.4);
	auto f3 = handler.queue_invoke_with_id(id3, commit, callback, 3, 0.4);

	REQUIRE(handler.swap_requests(id2, id3));

	REQUIRE(f1.get() == 1);
	REQUIRE(f2.get() == 2);
	REQUIRE(f3.get() == 3);
}

TEST_CASE("Take returns moved queue entries for non-committed requests", "[queue-control]") {
	auto commit = [](int val, double sleep_s) -> pybind11::object {
		return pybind11::make_tuple(val, sleep_s);
	};
	auto callback = [](const pybind11::object& obj) { return obj.cast<int>(); };

	PyManager& manager = getContext().manager;
	PyManager::InvokeHandler handler =
		manager.loadPythonModule("tests.test_modules.black_hole", "invoke", 1, 1);

	const auto id1 = static_cast<PyManager::InvokeHandler::RequestId>(3001);
	const auto id2 = static_cast<PyManager::InvokeHandler::RequestId>(3002);
	const auto id3 = static_cast<PyManager::InvokeHandler::RequestId>(3003);
	auto f1 = handler.queue_invoke_with_id(id1, commit, callback, 11, 0.4);
	auto f2 = handler.queue_invoke_with_id(id2, commit, callback, 22, 0.4);
	auto f3 = handler.queue_invoke_with_id(id3, commit, callback, 33, 0.4);

	auto taken_one = handler.take_request(id2);
	REQUIRE(taken_one.has_value());
	REQUIRE(taken_one->request_id == id2);
	taken_one->on_error(std::make_exception_ptr(std::runtime_error("taken by test before commit")));

	auto taken_many = handler.take_requests({ id3, id1 });
	REQUIRE((taken_many.size() == 1 || taken_many.size() == 2));
	for(auto& entry : taken_many) {
		entry.on_error(std::make_exception_ptr(std::runtime_error("taken by test before commit")));
	}

	const bool id1_taken = std::any_of(
		taken_many.begin(), taken_many.end(), [id1](const auto& e) { return e.request_id == id1; });
	const bool id3_taken = std::any_of(
		taken_many.begin(), taken_many.end(), [id3](const auto& e) { return e.request_id == id3; });

	if(id1_taken) {
		REQUIRE_THROWS(f1.get());
	} else {
		REQUIRE(f1.get() == 11);
	}
	REQUIRE_THROWS(f2.get());
	if(id3_taken) {
		REQUIRE_THROWS(f3.get());
	} else {
		REQUIRE(f3.get() == 33);
	}
}

TEST_CASE("Duplicate request ids are rejected", "[queue-control]") {
	auto commit = [](int val, double sleep_s) -> pybind11::object {
		return pybind11::make_tuple(val, sleep_s);
	};
	auto callback = [](const pybind11::object& obj) { return obj.cast<int>(); };

	PyManager& manager = getContext().manager;
	PyManager::InvokeHandler handler =
		manager.loadPythonModule("tests.test_modules.black_hole", "invoke", 1, 1);

	const auto request_id = static_cast<PyManager::InvokeHandler::RequestId>(4001);
	auto f1 = handler.queue_invoke_with_id(request_id, commit, callback, 123, 0.4);

	REQUIRE_THROWS_AS(handler.queue_invoke_with_id(request_id, commit, callback, 456, 0.4),
					  std::invalid_argument);

	REQUIRE(f1.get() == 123);
}

TEST_CASE("Asynchronous invoke perfect forwarding", "[basic]") {
	PyManager& manager = getContext().manager;
	PyManager::InvokeHandler reflect = manager.loadPythonModule("tests.test_modules.identity");

	auto commit = [](int val) -> pybind11::object { return pybind11::cast(val); };

	std::vector<std::future<int>> futures;
	for(size_t i = 0; i < 5; i++) {
		std::unique_ptr<int> ptr = std::make_unique<int>(1);
		auto callback = [ptr = std::move(ptr)](const pybind11::object& obj) {
			return obj.cast<int>() + *ptr;
		};
		futures.push_back(reflect.queue_invoke(commit, std::move(callback), 1));
	}

	for(auto& future : futures) {
		REQUIRE(future.get() == 2);
	}
}

TEST_CASE("Multiple arguments", "[basic]") {
	auto commit = [](int a, int b) -> pybind11::object { return pybind11::make_tuple(a, b); };
	auto callback = [](const pybind11::object& obj) { return obj.cast<int>(); };

	PyManager& manager = getContext().manager;
	PyManager::InvokeHandler reflect = manager.loadPythonModule("tests.test_modules.add", "add");

	std::vector<std::future<int>> futures;
	for(size_t i = 0; i < 5; i++) {
		futures.push_back(reflect.queue_invoke(commit, callback, 1, 2));
	}

	for(auto& future : futures) {
		REQUIRE(future.get() == 3);
	}
}

TEST_CASE("SVD no requests lost", "[batch]") {
	/// (<16MB of ram)
	const int MAT_DIM = 100;
	const int N = 1000;
	auto commit = [](int n) -> pybind11::object { return pybind11::cast(n); };
	auto extract_raw = [](pybind11::object&& obj) { return obj.release().ptr(); };

	PyManager& manager = getContext().manager;
	PyManager::InvokeHandler generate =
		manager.loadPythonModule("tests.test_modules.svd", "generate");
	PyManager::InvokeHandler compute =
		manager.loadPythonModule("tests.test_modules.svd", "compute_svd_rank", 32, 3);

	// step 1: generate the N matrices
	std::vector<std::future<PyObject*>> matrix_futures;
	for(int i = 0; i < N; i++) {
		matrix_futures.emplace_back(generate.queue_invoke(commit, extract_raw, MAT_DIM));
	}
	std::vector<PyObject*> matrices;
	for(int i = 0; i < N; i++) {
		matrices.push_back(matrix_futures[i].get());
		REQUIRE(matrices.back() != nullptr);
	}

	// step 2: compute the SVD rank of each matrix
	auto commit_step2 = [](PyObject* ptr) -> pybind11::object {
		return pybind11::reinterpret_steal<pybind11::object>(ptr);
	};
	auto callback_step2 = [](pybind11::object&& obj) { return obj.cast<int>(); };
	std::vector<std::future<int>> rank_futures;
	for(int i = 0; i < N; i++) {
		rank_futures.emplace_back(compute.queue_invoke(commit_step2, callback_step2, matrices[i]));
	}
	int cnt = 0;
	for(int i = 0; i < N; i++) {
		int rank = rank_futures[i].get();
		cnt++;
	}
	REQUIRE(cnt == N);
}

TEST_CASE("Batched asynchronous invoke", "[batch]") {
	auto commit = [](int val) -> pybind11::object { return pybind11::cast(val); };
	auto callback = [](const pybind11::object& obj) { return obj.cast<int>(); };

	PyManager& manager = getContext().manager;
	// batch_size=5, prefetch_depth=1
	PyManager::InvokeHandler reflect =
		manager.loadPythonModule("tests.test_modules.identity", "invoke", 5, 1);

	std::vector<std::future<int>> futures;
	for(int i = 0; i < 10; i++) {
		futures.push_back(reflect.queue_invoke(commit, callback, i));
	}

	for(int i = 0; i < 10; i++) {
		REQUIRE(futures[i].get() == i);
	}
}

TEST_CASE("Batched with prefetch depth", "[batch]") {
	auto commit = [](int val) -> pybind11::object { return pybind11::cast(val); };
	auto callback = [](const pybind11::object& obj) { return obj.cast<int>(); };

	PyManager& manager = getContext().manager;
	// batch_size=3, prefetch_depth=2 → up to 6 items pre-committed
	PyManager::InvokeHandler reflect =
		manager.loadPythonModule("tests.test_modules.identity", "invoke", 3, 2);

	std::vector<std::future<int>> futures;
	for(int i = 0; i < 9; i++) {
		futures.push_back(reflect.queue_invoke(commit, callback, i * 10));
	}

	for(int i = 0; i < 9; i++) {
		REQUIRE(futures[i].get() == i * 10);
	}
}

TEST_CASE("Partial batch drain on shutdown", "[batch]") {
	auto commit = [](int val) -> pybind11::object { return pybind11::cast(val); };
	auto callback = [](const pybind11::object& obj) { return obj.cast<int>(); };

	PyManager& manager = getContext().manager;
	// batch_size=4, but we only enqueue 3 items — should drain on handler destruction
	PyManager::InvokeHandler reflect =
		manager.loadPythonModule("tests.test_modules.identity", "invoke", 4, 1);

	std::vector<std::future<int>> futures;
	for(int i = 0; i < 3; i++) {
		futures.push_back(reflect.queue_invoke(commit, callback, i + 100));
	}

	// Destroy handler — should drain the partial batch
	reflect = manager.loadPythonModule("tests.test_modules.identity", "invoke", 1, 1);

	for(int i = 0; i < 3; i++) {
		REQUIRE(futures[i].get() == i + 100);
	}
}

#if defined(PYSCHEDULER_TEST_HAS_CUDA) && PYSCHEDULER_TEST_HAS_CUDA &&                             \
	__has_include(<cuda_runtime.h>) && __has_include(<dlpack/dlpack.h>)
namespace {
std::vector<float> generateRandomMatrix(int rows, int cols, uint32_t seed) {
	std::mt19937 rng(seed);
	std::uniform_real_distribution<float> dist(-1.0f, 1.0f);
	std::vector<float> out(static_cast<size_t>(rows * cols));
	for(float& v : out) {
		v = dist(rng);
	}
	return out;
}

std::vector<std::vector<float>> matmulCpuReference(
	const std::vector<float>& a, int rows_a, int cols_a, const std::vector<float>& b, int cols_b) {
	std::vector<float> out_flat(static_cast<size_t>(rows_a * cols_b), 0.0f);
	cblas_sgemm(CblasRowMajor,
				CblasNoTrans,
				CblasNoTrans,
				rows_a,
				cols_b,
				cols_a,
				1.0f,
				a.data(),
				cols_a,
				b.data(),
				cols_b,
				0.0f,
				out_flat.data(),
				cols_b);

	std::vector<std::vector<float>> out(rows_a, std::vector<float>(cols_b));
	for(int r = 0; r < rows_a; r++) {
		for(int c = 0; c < cols_b; c++) {
			out[r][c] = out_flat[static_cast<size_t>(r * cols_b + c)];
		}
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
} // namespace

TEST_CASE("Construct CUDA tensor in C++ and dense matmul in Python", "[gpu][dlpack]") {
	int device_count = 0;
	if(cudaGetDeviceCount(&device_count) != cudaSuccess || device_count == 0) {
		SKIP("No CUDA device is available on this system");
	}

	PyManager& manager = getContext().manager;
	auto has_cuda = manager.loadPythonModule("tests.test_modules.dense_matmul", "has_cuda");
	if(!has_cuda.invoke<bool>()) {
		SKIP("PyTorch CUDA support is unavailable in the Python environment");
	}

	auto matmul =
		manager.loadPythonModule("tests.test_modules.dense_matmul", "dense_matmul_from_dlpack");

	std::vector<float> a_host = { 1.f, 2.f, 3.f, 4.f, 5.f, 6.f }; // 2x3
	std::vector<float> b_host = { 7.f, 8.f, 9.f, 10.f, 11.f, 12.f }; // 3x2

	auto parse_result = [](pybind11::object&& obj) {
		return obj.cast<std::vector<std::vector<float>>>();
	};

	std::vector<std::vector<float>> result;
	{
		// Keep capsule construction under the GIL to satisfy pybind11 refcount checks.
		pybind11::gil_scoped_acquire gil;
		auto a_capsule = toDlpackCapsule(createCudaMatrixDlpack(a_host, 2, 3));
		auto b_capsule = toDlpackCapsule(createCudaMatrixDlpack(b_host, 3, 2));
		result = matmul.invoke(parse_result, a_capsule, b_capsule);
	}

	REQUIRE(result.size() == 2);
	REQUIRE(result[0].size() == 2);
	REQUIRE(result[1].size() == 2);

	REQUIRE(std::fabs(result[0][0] - 58.f) < 1e-5f);
	REQUIRE(std::fabs(result[0][1] - 64.f) < 1e-5f);
	REQUIRE(std::fabs(result[1][0] - 139.f) < 1e-5f);
	REQUIRE(std::fabs(result[1][1] - 154.f) < 1e-5f);
}

TEST_CASE("Asynchronous randomized CUDA DLPack matmul", "[gpu][dlpack][async]") {
	int device_count = 0;
	if(cudaGetDeviceCount(&device_count) != cudaSuccess || device_count == 0) {
		SKIP("No CUDA device is available on this system");
	}

	PyManager& manager = getContext().manager;
	auto has_cuda = manager.loadPythonModule("tests.test_modules.dense_matmul", "has_cuda");
	if(!has_cuda.invoke<bool>()) {
		SKIP("PyTorch CUDA support is unavailable in the Python environment");
	}

	auto matmul = manager.loadPythonModule(
		"tests.test_modules.dense_matmul", "dense_matmul_from_dlpack_batch", 8, 1);

	auto commit = [](int a_rows, int a_cols, int b_cols, uint32_t seed) -> pybind11::object {
		auto a_host = generateRandomMatrix(a_rows, a_cols, seed);
		auto b_host = generateRandomMatrix(a_cols, b_cols, seed ^ 0x9e3779b9u);

		pybind11::gil_scoped_acquire gil;
		auto a_capsule = toDlpackCapsule(createCudaMatrixDlpack(a_host, a_rows, a_cols));
		auto b_capsule = toDlpackCapsule(createCudaMatrixDlpack(b_host, a_cols, b_cols));
		return pybind11::make_tuple(std::move(a_capsule), std::move(b_capsule));
	};

	std::vector<std::future<bool>> futures;
	std::mt19937 rng(1337);
	std::uniform_int_distribution<int> dim_dist(100, 200);
	std::uniform_int_distribution<uint32_t> seed_dist;

	for(int i = 0; i < 24; i++) {
		int a_rows = dim_dist(rng);
		int a_cols = dim_dist(rng);
		int b_cols = dim_dist(rng);
		uint32_t seed = seed_dist(rng);

		auto callback = [a_rows, a_cols, b_cols, seed](pybind11::object&& obj) {
			auto got = obj.cast<std::vector<std::vector<float>>>();
			auto a_host = generateRandomMatrix(a_rows, a_cols, seed);
			auto b_host = generateRandomMatrix(a_cols, b_cols, seed ^ 0x9e3779b9u);
			auto expect = matmulCpuReference(a_host, a_rows, a_cols, b_host, b_cols);

			if(got.size() != expect.size()) return false;
			for(size_t r = 0; r < got.size(); r++) {
				if(got[r].size() != expect[r].size()) return false;
				for(size_t c = 0; c < got[r].size(); c++) {
					if(std::fabs(got[r][c] - expect[r][c]) > 2e-3f) return false;
				}
			}
			return true;
		};

		futures.push_back(
			matmul.queue_invoke(commit, std::move(callback), a_rows, a_cols, b_cols, seed));
	}

	for(auto& f : futures) {
		REQUIRE(f.get());
	}
}
#else
TEST_CASE("Construct CUDA tensor in C++ and dense matmul in Python", "[gpu][dlpack]") {
	SKIP("CUDA toolkit or DLPack headers are unavailable at compile time");
}

TEST_CASE("Asynchronous randomized CUDA DLPack matmul", "[gpu][dlpack][async]") {
	SKIP("CUDA toolkit or DLPack headers are unavailable at compile time");
}
#endif
