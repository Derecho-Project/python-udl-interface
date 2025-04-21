#define NPY_NO_DEPRECATED_API NPY_1_7_API_VERSION
#include "lib/python_manager.hpp"
#include <fstream>
#include <nvml.h>
#include <pybind11/stl.h>
#include <random>

using namespace std::chrono;
namespace pys = pyscheduler;

typedef std::vector<system_clock::time_point> time_vector;
const size_t NUM_WORKERS = pys::PyManager::NUM_WORKERS;
const size_t BATCH_SIZE = 128;
const size_t START_REQ_PER_SECOND = 10;
const size_t END_REQ_PER_SECOND = 100;
const size_t INCREMENT_REQ_PER_SECOND = 2;

unsigned long long get_total_cpu_time() {
	std::ifstream file("/proc/stat");
	std::string cpu;
	unsigned long long user, nice, system, idle, iowait, irq, softirq;
	file >> cpu >> user >> nice >> system >> idle >> iowait >> irq >> softirq;
	return user + nice + system + idle + iowait + irq + softirq;
}

unsigned long long get_proc_cpu_time(pid_t pid) {
	std::ifstream file("/proc/" + std::to_string(pid) + "/stat");
	std::string token;
	for(int i = 1; i <= 13; ++i)
		file >> token;
	unsigned long long utime, stime;
	file >> utime >> stime;
	return utime + stime;
}

double get_cpu_usage(pid_t pid) {
	auto total1 = get_total_cpu_time();
	auto proc1 = get_proc_cpu_time(pid);
	sleep(1);
	auto total2 = get_total_cpu_time();
	auto proc2 = get_proc_cpu_time(pid);

	double total_diff = total2 - total1;
	double proc_diff = proc2 - proc1;

	return (proc_diff / total_diff) * 100.0;
}

void latency_measurement(std::atomic<bool>& stop_flag,
						 std::vector<std::tuple<uint64_t, double, int>>& metrics) {
	pys::PyManager manager;
	pys::PyManager::InvokeHandler add = manager.getPythonModule("python_models.add");

	std::random_device rd;
	std::mt19937 gen(rd());
	std::uniform_int_distribution<> uniform_distribution(-1234567, 123456);

	auto callback = [](pybind11::object object) { return high_resolution_clock::now(); };

	time_vector start_times;
	std::vector<std::future<system_clock::time_point>> futures;
	std::vector<double> cpu_utilization;
	std::vector<double> gpu_utilization;

	const pid_t pid = getpid();

	nvmlDevice_t device;
	nvmlDeviceGetHandleByIndex(0, &device);

	while(!stop_flag) {
		int random1 = uniform_distribution(gen);
		int random2 = uniform_distribution(gen);

		// enqueue start time
		// and add query into the thread pool
		start_times.emplace_back(high_resolution_clock::now());
		futures.emplace_back(add.queue_invoke(callback, random1, random2));

		auto total1 = get_total_cpu_time();
		auto proc1 = get_proc_cpu_time(pid);
		std::this_thread::sleep_for(milliseconds(1000));
		auto total2 = get_total_cpu_time();
		auto proc2 = get_proc_cpu_time(pid);

		double total_diff = total2 - total1;
		double proc_diff = proc2 - proc1;
		double cpu_util = (proc_diff / total_diff) * 100.0;

		cpu_utilization.push_back(cpu_util);

		nvmlUtilization_t utilization;
		nvmlDeviceGetUtilizationRates(device, &utilization);
		gpu_utilization.push_back(utilization.gpu);

		std::cout << cpu_util << " " << utilization.gpu << " " << utilization.memory << std::endl;
	}

	for(size_t i = 0; i < start_times.size(); i++) {
		futures[i].wait();
		auto value = futures[i].get();

		uint64_t time_elapsed = duration_cast<microseconds>(value - start_times[i]).count();
		// latency_us.push_back();
	}
}

void benchmark_encode(size_t number_of_queries,
					  size_t queries_per_second,
					  std::atomic<bool>& stop_flag) {
	pys::PyManager manager;
	pys::PyManager::InvokeHandler encoder = manager.getPythonModule("python_models.encoder");

	std::vector<std::string> batch(BATCH_SIZE, "hello world");

	auto callback = [](pybind11::object object) { return 0; };
	std::vector<std::future<int>> futures;
	for(size_t i = 0; i < number_of_queries; i++) {
		futures.emplace_back(encoder.queue_invoke(callback, batch));
		std::this_thread::sleep_for(
			milliseconds(static_cast<long>((1. / (double)queries_per_second) * 1000)));
	}
	stop_flag.store(true);

	for(size_t i = 0; i < number_of_queries; i++) {
		futures[i].wait();
	}
}

inline int64_t currentTimeUs() {
	return duration_cast<microseconds>(high_resolution_clock::now().time_since_epoch()).count();
}

std::vector<std::string_view> get_views(const std::vector<std::string>& input) {
	std::vector<std::string_view> views;
	for(const auto& str : input) {
		views.emplace_back(str);
	}
	return views; // Views refer to original strings
}

int main() {
	nvmlInit();
	pys::PyManager manager;

	pys::PyManager::InvokeHandler add = manager.getPythonModule("python_models.add");
	pys::PyManager::InvokeHandler encoder = manager.getPythonModule("python_models.encoder");
	std::atomic<bool> stop_flag = false;

	std::thread sampler([&add, &stop_flag]() {
		auto time_elapsed = [](pybind11::object object) {
			int64_t time_start = object.cast<int64_t>();
			int64_t time_now = currentTimeUs();
			return time_now - time_start;
		};

		std::queue<std::future<int64_t>> futures;
		while(!stop_flag) {
			futures.emplace(add.queue_invoke(time_elapsed, currentTimeUs(), 0));
			std::this_thread::sleep_for(milliseconds(200));

			while(!futures.empty() && futures.front().valid()) {
				std::cout << "time elapsed: " << futures.front().get() << "\n";
				futures.pop();
			}
		}

		while(!futures.empty()) {
			futures.front().wait();
			std::cout << "time elapsed: " << futures.front().get() << "\n";
			futures.pop();
		}
	});

	auto black_box = [](pybind11::object object) { return 0; };
	std::vector<std::string> batch(
		BATCH_SIZE,
		"this is just a test, but I want to emulate a piece of text that has a good length");
	std::vector<std::string_view> views = get_views(batch);

	for(int i = 0; i < 10 * 50; i++) {
		encoder.queue_invoke(black_box, views);
		std::this_thread::sleep_for(milliseconds(30));
	}

	std::this_thread::sleep_for(seconds(5));
	stop_flag = true;
	sampler.join();
	std::cout << "joined" << std::endl;
	std::this_thread::sleep_for(seconds(5));

	// std::ofstream fout("csv/cpp_benchmark_" + std::to_string(NUM_WORKERS) + "_workers_" +
	// 				   std::to_string(BATCH_SIZE) + "_batch.csv");
	// fout << "Num Workers, Batch Size, Queries Per Second, Latency (us)\n";
	// for(const int req_per_second : requests_per_second) {

	// 	std::atomic<bool> stop_flag = false;
	// 	std::vector<std::tuple<uint64_t, double, int>> metrics;

	// 	std::thread latency_measurement_thread(
	// 		latency_measurement, std::ref(stop_flag), std::ref(metrics));
	// 	benchmark_encode(req_per_second * 10, req_per_second, stop_flag);
	// 	latency_measurement_thread.join();

	// 	for(auto metric : metrics) {
	// 		fout << NUM_WORKERS << ", " << BATCH_SIZE << ", " << req_per_second << ", "
	// 			 << std::get<0>(metric) << '\n';
	// 	}
	// 	fout.flush();
	// }
	// fout.close();
	// nvmlShutdown();
	return 0;
}
