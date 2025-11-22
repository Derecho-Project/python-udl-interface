#include <chrono>
#include <pyscheduler/pyscheduler.hpp>

using namespace std::chrono;
using namespace std;

constexpr int const NUM_ITERATIONS = 600;
constexpr int const DIM = 100;

int main() {
	auto module_load_start = chrono::high_resolution_clock::now();
	pyscheduler::PyManager manager;
	pyscheduler::PyManager::InvokeHandler generator = manager.loadPythonModule("examples.tensor_share.python_modules.tensor_juggler", "generate_tensor");
	pyscheduler::PyManager::InvokeHandler fma = manager.loadPythonModule("examples.tensor_share.python_modules.tensor_juggler", "multiply_sum_tensors");
	auto module_load_end = chrono::high_resolution_clock::now();

	vector<std::pair<pybind11::object, pybind11::object>> cache;

	for (int i = 0; i < NUM_ITERATIONS; i++) {
		cache.emplace_back(
			std::make_pair(
				generator.invoke([&](pybind11::object x){return x; }, DIM, DIM),
				generator.invoke([&](pybind11::object x){return x; }, DIM, DIM)
			)
		);
	}

	// solve problems
	std::vector<std::future<int>> promises;
	promises.reserve(NUM_ITERATIONS);


	auto module_solve_start = chrono::high_resolution_clock::now();
	for (int i = 0; i < NUM_ITERATIONS; i++) {
		auto [a, b] = std::move(cache[i]);
		promises.emplace_back(
			fma.queue_invoke(
				[](pybind11::object x){return x.cast<int>();}, 
				std::move(a),
				std::move(b)
			)
		);
	}
	for(auto &o : promises) {
		o.wait();
	}
	auto module_solve_end = chrono::high_resolution_clock::now();
	std::cout << duration_cast<microseconds>(module_solve_end - module_solve_start).count() << std::endl;

	return 0;
}