#include "pyscheduler/pyscheduler.hpp"
#include <catch2/catch_all.hpp>

using namespace pyscheduler;

static PyManager manager;

TEST_CASE("Load Standard Module", "[basic]") {
	PyManager::InvokeHandler reflect = manager.getPythonModule("tests.test_modules.reflect");
}

TEST_CASE("Reflect Test", "[basic]") {
	PyManager::InvokeHandler reflect = manager.getPythonModule("tests.test_modules.reflect");

	REQUIRE(reflect.invoke<std::string>("hello") == "hello");
	REQUIRE(reflect.invoke<double>(3.1415926535) == 3.1415926535);
}