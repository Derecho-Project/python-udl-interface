#include "lib/python_manager.hpp"
#include <cuda_runtime.h>
#include <dlpack/dlpack.h>
#include <pybind11/pybind11.h>
#include <vector>

using namespace std::chrono;
namespace pys = pyscheduler;

// Simple example of how to receive a DLPack object from python

std::vector<size_t> history;

auto dlStatusPrinter = [](pybind11::object object) {
	// Note that the capsule object destructor requires the GIL to be held
	// and may result in some awkwardness like the following
	//
	// pybind11:scoped_gil_acquire gil;
	// {
	//		pybind11::capsule capsule = handler.invoke(1, 2, 3);
	//		process capsule here
	// }
	//
	// c++ destructors are run in reverse order of construction,
	// so unless the thread already holds the gil before
	// calling invoke (object is created inside the invoke method, so returning it outside
	// extends the lifetime beyond the scope of the function), the above awkwardness is required.

	// however, this callback function is being run inside a gil-acquired environment
	// so we don't need to do anything here!
	pybind11::capsule capsule = object.cast<pybind11::capsule>();
	DLManagedTensor* dlmt = reinterpret_cast<DLManagedTensor*>(capsule.get_pointer());

	if(!dlmt) {
		throw std::runtime_error("Unable to retrieve DLPack tensor from capsule.");
	}

	std::cout << "Tensor data pointer: " << dlmt->dl_tensor.data << std::endl;
	std::cout << "Tensor device:" << dlmt->dl_tensor.device.device_type << std::endl;
	std::cout << "Tensor dimensions: " << dlmt->dl_tensor.ndim << std::endl;
	size_t num_items = 1;
	for(int i = 0; i < dlmt->dl_tensor.ndim; i++) {
		std::cout << "Shape[" << i << "] = " << dlmt->dl_tensor.shape[i] << std::endl;
		num_items *= dlmt->dl_tensor.shape[i];
	}

	// can mutate global state
	history.emplace_back(num_items);
	cudaFree(capsule.get_pointer());

	return static_cast<int>(num_items);
};

int main() {
	pys::PyManager manager;
	pys::PyManager::InvokeHandler tensor =
		manager.getPythonModule("python_models.tensor_gen", "invoke");

	tensor.invoke(dlStatusPrinter, 128, 128);
	tensor.invoke(dlStatusPrinter, 3, 3);
	tensor.invoke(dlStatusPrinter, 24, 36);

	std::vector<std::future<int>> promises;
	for(int i = 0; i < 8; i++) {
		promises.emplace_back(tensor.queue_invoke(dlStatusPrinter, 1024, 1024));
	}

	for(auto& p : promises) {
		p.wait();
		std::cout << p.get() << std::endl;
		;
	}

	return 0;
}