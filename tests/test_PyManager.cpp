#include "pyscheduler/pyscheduler.hpp"
#include "pyscheduler/tensor.hpp"
#include <catch2/catch_all.hpp>
#include <catch2/reporters/catch_reporter_event_listener.hpp>
#include <catch2/reporters/catch_reporter_registrars.hpp>
#include <pybind11/pytypes.h>
#include <pybind11/stl.h>

#include <cstdlib>

namespace {
// Bypass libtorch_python's static destructor chain, which crashes inside
// `_PyThreadState_DeleteCurrent` when its embedded pybind11 acquires the GIL
// at process exit. Catch2 has already written its summary by the time
// `testRunEnded` fires, so terminating early is safe for tests.
struct CleanExitListener : Catch::EventListenerBase {
	using Catch::EventListenerBase::EventListenerBase;
	void testRunEnded(Catch::TestRunStats const& stats) override {
		std::_Exit(stats.totals.assertions.failed == 0
				   && stats.totals.testCases.failed == 0
					   ? 0
					   : 1);
	}
};
} // namespace
CATCH_REGISTER_LISTENER(CleanExitListener)

#include <chrono>
#include <cmath>
#include <cstdint>
#include <dlfcn.h>
#include <string>
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

TEST_CASE("Void-returning callback yields std::future<void>", "[basic][void]") {
	auto commit = [](int val) -> pybind11::object { return pybind11::cast(val); };

	std::atomic<int> sum{ 0 };
	auto callback = [&sum](const pybind11::object& obj) -> void {
		sum.fetch_add(obj.cast<int>(), std::memory_order_relaxed);
	};

	PyManager& manager = getContext().manager;
	PyManager::InvokeHandler reflect =
		manager.loadPythonModule("tests.test_modules.identity", "invoke", 4, 1);

	std::vector<std::future<void>> futures;
	const int N = 20;
	int expected = 0;
	for(int i = 0; i < N; i++) {
		expected += i;
		futures.push_back(reflect.queue_invoke(commit, callback, i));
	}

	for(auto& f : futures) {
		f.get(); // must not throw
	}
	REQUIRE(sum.load() == expected);
}

TEST_CASE("Void-returning callback propagates thrown exceptions", "[basic][void][error]") {
	auto commit = [](int val) -> pybind11::object { return pybind11::cast(val); };
	auto callback = [](const pybind11::object&) -> void {
		throw std::runtime_error("callback failure");
	};

	PyManager& manager = getContext().manager;
	PyManager::InvokeHandler reflect =
		manager.loadPythonModule("tests.test_modules.identity", "invoke", 1, 1);

	auto fut = reflect.queue_invoke(commit, callback, 1);
	REQUIRE_THROWS_AS(fut.get(), std::runtime_error);
}

TEST_CASE("Commit function throwing propagates to future", "[error]") {
	auto commit = [](int val) -> pybind11::object {
		if(val < 0) throw std::runtime_error("commit failure");
		return pybind11::cast(val);
	};
	auto callback = [](const pybind11::object& obj) { return obj.cast<int>(); };

	PyManager& manager = getContext().manager;
	PyManager::InvokeHandler reflect =
		manager.loadPythonModule("tests.test_modules.identity", "invoke", 2, 1);

	auto good = reflect.queue_invoke(commit, callback, 7);
	auto bad = reflect.queue_invoke(commit, callback, -1);
	auto good2 = reflect.queue_invoke(commit, callback, 9);

	REQUIRE(good.get() == 7);
	REQUIRE_THROWS_AS(bad.get(), std::runtime_error);
	REQUIRE(good2.get() == 9);
}

TEST_CASE("Python exception fans out to all batch callbacks", "[error][batch]") {
	auto commit = [](int val) -> pybind11::object { return pybind11::cast(val); };
	auto callback = [](const pybind11::object& obj) { return obj.cast<int>(); };

	PyManager& manager = getContext().manager;
	PyManager::InvokeHandler raises =
		manager.loadPythonModule("tests.test_modules.raises", "invoke", 4, 1);

	std::vector<std::future<int>> futures;
	for(int i = 0; i < 4; i++) {
		futures.push_back(raises.queue_invoke(commit, callback, i));
	}
	for(auto& f : futures) {
		REQUIRE_THROWS(f.get());
	}
}

TEST_CASE("Synchronous invoke surfaces Python exceptions", "[basic][error]") {
	PyManager& manager = getContext().manager;
	PyManager::InvokeHandler raises =
		manager.loadPythonModule("tests.test_modules.raises", "invoke");

	// pybind11::error_already_set must be destroyed under the GIL; hold it
	// across the catch site so the propagated exception unwinds safely.
	pybind11::gil_scoped_acquire gil;
	REQUIRE_THROWS(raises.invoke<int>(pybind11::list()));
}

TEST_CASE("get_queue_stats reports zero state, then accumulates", "[stats]") {
	auto commit = [](int val) -> pybind11::object { return pybind11::cast(val); };
	auto callback = [](const pybind11::object& obj) { return obj.cast<int>(); };

	PyManager& manager = getContext().manager;
	const size_t batch_size = 5;
	const size_t prefetch_depth = 2;
	PyManager::InvokeHandler reflect = manager.loadPythonModule(
		"tests.test_modules.identity", "invoke", batch_size, prefetch_depth);

	auto initial = reflect.get_queue_stats();
	REQUIRE(initial.total_enqueued == 0);
	REQUIRE(initial.commit_batch_size_ema == 0.0);
	REQUIRE(initial.execute_batch_size_ema == 0.0);
	REQUIRE(initial.commit_ns_per_batch_ema == 0.0);
	REQUIRE(initial.execute_ns_per_batch_ema == 0.0);

	const int N = 100;
	std::vector<std::future<int>> futures;
	for(int i = 0; i < N; i++) {
		futures.push_back(reflect.queue_invoke(commit, callback, i));
	}
	for(auto& f : futures) f.get();

	auto stats = reflect.get_queue_stats();
	REQUIRE(stats.total_enqueued == N);
	// After draining, queues are empty.
	REQUIRE(stats.commit_queue_size == 0);
	REQUIRE(stats.execute_queue_size == 0);
	// Worker did real work, so EMAs are populated.
	REQUIRE(stats.commit_batch_size_ema > 0.0);
	REQUIRE(stats.execute_batch_size_ema > 0.0);
	REQUIRE(stats.commit_ns_per_batch_ema > 0.0);
	REQUIRE(stats.execute_ns_per_batch_ema > 0.0);
	// Execute batches are bounded by configured batch_size.
	REQUIRE(stats.execute_batch_size_ema <= static_cast<double>(batch_size));
}

TEST_CASE("add_path rejects empty and is idempotent", "[basic][add_path]") {
	PyManager& manager = getContext().manager;
	REQUIRE_THROWS_AS(manager.add_path(""), std::invalid_argument);

	// Multiple calls with the same path should not duplicate the sys.path entry.
	const std::string p = PYSCHEDULER_SOURCE_DIR;
	manager.add_path(p);
	manager.add_path(p);
	manager.add_path(p);

	pybind11::gil_scoped_acquire gil;
	pybind11::module_ sys = pybind11::module_::import("sys");
	pybind11::list sys_path = sys.attr("path");
	int count = 0;
	for(auto item : sys_path) {
		if(pybind11::str(item).cast<std::string>() == p) count++;
	}
	REQUIRE(count == 1);
}

TEST_CASE("InvokeHandler move-construction preserves in-flight work", "[move]") {
	auto commit = [](int val) -> pybind11::object { return pybind11::cast(val); };
	auto callback = [](const pybind11::object& obj) { return obj.cast<int>(); };

	PyManager& manager = getContext().manager;
	PyManager::InvokeHandler src =
		manager.loadPythonModule("tests.test_modules.identity", "invoke", 4, 1);

	std::vector<std::future<int>> futures;
	for(int i = 0; i < 25; i++) {
		futures.push_back(src.queue_invoke(commit, callback, i));
	}

	PyManager::InvokeHandler moved(std::move(src));

	for(int i = 25; i < 50; i++) {
		futures.push_back(moved.queue_invoke(commit, callback, i));
	}

	for(int i = 0; i < 50; i++) {
		REQUIRE(futures[i].get() == i);
	}
}

TEST_CASE("Concurrent producers all complete and total_enqueued matches", "[concurrent]") {
	auto commit = [](int val) -> pybind11::object { return pybind11::cast(val); };
	auto callback = [](const pybind11::object& obj) { return obj.cast<int>(); };

	PyManager& manager = getContext().manager;
	PyManager::InvokeHandler reflect =
		manager.loadPythonModule("tests.test_modules.identity", "invoke", 8, 2);

	const int kThreads = 4;
	const int kPerThread = 250;
	std::vector<std::vector<std::future<int>>> per_thread(kThreads);
	std::vector<std::thread> producers;

	for(int t = 0; t < kThreads; t++) {
		producers.emplace_back([&, t] {
			for(int i = 0; i < kPerThread; i++) {
				per_thread[t].push_back(
					reflect.queue_invoke(commit, callback, t * kPerThread + i));
			}
		});
	}
	for(auto& th : producers) th.join();

	int seen = 0;
	for(int t = 0; t < kThreads; t++) {
		for(int i = 0; i < kPerThread; i++) {
			REQUIRE(per_thread[t][i].get() == t * kPerThread + i);
			seen++;
		}
	}
	REQUIRE(seen == kThreads * kPerThread);
	REQUIRE(reflect.get_queue_stats().total_enqueued ==
			static_cast<int64_t>(kThreads * kPerThread));
}

TEST_CASE("Two plugin DSOs share one PyManager global state", "[shared-state][plugin]") {
	auto base_arc = PyManager::debug_arc_count();

	auto load_plugin = [](const char* path) -> void* {
		void* handle = dlopen(path, RTLD_NOW | RTLD_LOCAL);
		if(handle == nullptr) {
			FAIL(std::string("dlopen failed for ") + path + ": " + dlerror());
		}
		return handle;
	};

	auto load_symbol = [](void* handle, const char* name) -> void* {
		dlerror();
		void* sym = dlsym(handle, name);
		const char* err = dlerror();
		if(err != nullptr) {
			FAIL(std::string("dlsym failed for ") + name + ": " + err);
		}
		return sym;
	};

	void* plugin_a = load_plugin(PYSCHEDULER_TEST_PLUGIN_A_PATH);
	void* plugin_b = load_plugin(PYSCHEDULER_TEST_PLUGIN_B_PATH);

	auto start_a = reinterpret_cast<void (*)()>(load_symbol(plugin_a, "plugin_start"));
	auto stop_a = reinterpret_cast<void (*)()>(load_symbol(plugin_a, "plugin_stop"));
	auto arc_a = reinterpret_cast<uint64_t (*)()>(load_symbol(plugin_a, "plugin_arc_count"));
	auto addr_a =
		reinterpret_cast<uintptr_t (*)()>(load_symbol(plugin_a, "plugin_shared_state_address"));

	auto start_b = reinterpret_cast<void (*)()>(load_symbol(plugin_b, "plugin_start"));
	auto stop_b = reinterpret_cast<void (*)()>(load_symbol(plugin_b, "plugin_stop"));
	auto arc_b = reinterpret_cast<uint64_t (*)()>(load_symbol(plugin_b, "plugin_arc_count"));
	auto addr_b =
		reinterpret_cast<uintptr_t (*)()>(load_symbol(plugin_b, "plugin_shared_state_address"));

	REQUIRE(addr_a() == PyManager::debug_shared_state_address());
	REQUIRE(addr_b() == PyManager::debug_shared_state_address());

	start_a();
	REQUIRE(arc_a() == base_arc + 1);

	start_b();
	REQUIRE(arc_a() == base_arc + 2);
	REQUIRE(arc_b() == base_arc + 2);

	stop_b();
	REQUIRE(arc_a() == base_arc + 1);

	stop_a();
	REQUIRE(PyManager::debug_arc_count() == base_arc);

	dlclose(plugin_b);
	dlclose(plugin_a);
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
