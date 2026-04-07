#include <pyscheduler/pyscheduler.hpp>

#include <chrono>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <thread>
#include <vector>

using namespace std;
using namespace pyscheduler;

int main() {
	constexpr int NUM_REQUESTS = 50;
	constexpr double SLEEP_SECONDS = 0.01;
	constexpr int BATCH_SIZE = 8;
	constexpr int PREFETCH_DEPTH = 2;

	PyManager manager;

	auto module_a = manager.loadPythonModule(
		"examples.multithreaded.python_modules.module_a", "invoke", BATCH_SIZE, PREFETCH_DEPTH);
	auto module_b = manager.loadPythonModule(
		"examples.multithreaded.python_modules.module_b", "invoke", BATCH_SIZE, PREFETCH_DEPTH);

	auto commit = [](double seconds) -> pybind11::object { return pybind11::cast(seconds); };
	auto callback = [](const pybind11::object& obj) { return obj.cast<double>(); };

	vector<future<double>> futures_a;
	vector<future<double>> futures_b;

	auto start = chrono::steady_clock::now();

	// Spawn two threads, each queueing work to a different handler
	thread t1([&] {
		for(int i = 0; i < NUM_REQUESTS; i++)
			futures_a.push_back(module_a.queue_invoke(commit, callback, SLEEP_SECONDS));
	});

	thread t2([&] {
		for(int i = 0; i < NUM_REQUESTS; i++)
			futures_b.push_back(module_b.queue_invoke(commit, callback, SLEEP_SECONDS));
	});

	t1.join();
	t2.join();

	// Collect results
	vector<double> results_a;
	vector<double> results_b;

	for(auto& f : futures_a)
		results_a.push_back(f.get());
	for(auto& f : futures_b)
		results_b.push_back(f.get());

	auto end = chrono::steady_clock::now();
	auto ms = chrono::duration_cast<chrono::milliseconds>(end - start).count();

	double expected_sequential = NUM_REQUESTS * 2 * SLEEP_SECONDS * 1000;
	cout << "Expected sequential: " << fixed << setprecision(0) << expected_sequential
		 << " ms\n";
	cout << "Actual:              " << ms << " ms\n";
	cout << "Speedup:             " << fixed << setprecision(2)
		 << (expected_sequential / ms) << "x\n";

	// Write results to file for diffing (outside timed region)
	ofstream out("/tmp/multithreaded_cpp.txt");
	for(size_t i = 0; i < results_a.size(); i++)
		out << "a[" << i << "] = " << setprecision(10) << results_a[i] << "\n";
	for(size_t i = 0; i < results_b.size(); i++)
		out << "b[" << i << "] = " << setprecision(10) << results_b[i] << "\n";

	cout << "Results written to /tmp/multithreaded_cpp.txt\n";

	return 0;
}
