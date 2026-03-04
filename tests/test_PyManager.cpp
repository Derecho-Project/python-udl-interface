#include "pyscheduler/pyscheduler.hpp"
#include <catch2/catch_all.hpp>

using namespace pyscheduler;

struct Context {
	PyManager manager;
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
