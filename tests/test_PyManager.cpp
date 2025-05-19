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
	auto to_double = [](const pybind11::object& obj) { return obj.cast<double>(); };

	PyManager& manager = getContext().manager;
	PyManager::InvokeHandler reflect = manager.loadPythonModule("tests.test_modules.identity");

	std::vector<std::future<double>> futures;
	for(size_t i = 0; i < 5; i++) {
		futures.push_back(reflect.queue_invoke(to_double, 3.1415926535));
	}

	for(auto& future : futures) {
		REQUIRE(future.get() == 3.1415926535);
	}
}

TEST_CASE("Asynchronous invoke perfect forwarding", "[basic]") {

	PyManager& manager = getContext().manager;
	PyManager::InvokeHandler reflect = manager.loadPythonModule("tests.test_modules.identity");

	std::vector<std::future<int>> futures;
	for(size_t i = 0; i < 5; i++) {
		std::unique_ptr<int> ptr = std::make_unique<int>(1);
		auto to_int_and_inc = [ptr = std::move(ptr)](const pybind11::object& obj) {
			return obj.cast<int>() + *ptr;
		};
		futures.push_back(reflect.queue_invoke(std::move(to_int_and_inc), 1));
	}

	for(auto& future : futures) {
		REQUIRE(future.get() == 2);
	}
}