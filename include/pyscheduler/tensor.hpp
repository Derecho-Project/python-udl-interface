#include "pyscheduler/library_export.hpp"
#include <cuda_runtime.h>
#include <dlpack.h>
#include <memory>

enum class DeviceType {
	CPU,
	CUDA,
};

template <typename T>
struct DLPackTypeTraits;

// I love explicit template specialization
// NOTE: If you get a compile time, add an entry here:

template <>
struct DLPackTypeTraits<float> {
	static constexpr DLDataType dtype = { kDLFloat, 32, 1 };
};

template <>
struct DLPackTypeTraits<double> {
	static constexpr DLDataType dtype = { kDLFloat, 64, 1 };
};

template <>
struct DLPackTypeTraits<int64_t> {
	static constexpr DLDataType dtype = { kDLInt, 64, 1 };
};

template <>
struct DLPackTypeTraits<int32_t> {
	static constexpr DLDataType dtype = { kDLInt, 32, 1 };
};

template <>
struct DLPackTypeTraits<uint8_t> {
	static constexpr DLDataType dtype = { kDLUInt, 8, 1 };
};

template <DeviceType Device, typename DataType, size_t... Dims>
std::unique_ptr<DLManagedTensor> createDlpackTensor() {
	constexpr int ndim = sizeof...(Dims);
	constexpr int64_t num_items = (... * Dims); // C++17 fold expression

	// Allocate and set shape
	int64_t* shape = new int64_t[ndim]{ Dims... };

	// Allocate tensor memory
	DataType* data;
	if constexpr(Device == DeviceType::CPU) {
		data = new T[num_items];
	} else if constexpr(Device == DeviceType::CUDA) {
		cudaMalloc(&data, num_items * sizeof(T));
	}

	// Create DLManagedTensor
	DLManagedTensor* managed_tensor = new DLManagedTensor();
	managed_tensor->dl_tensor.data = data;
	managed_tensor->dl_tensor.device = { Device == DeviceType::CPU ? kDLCPU : kDLCUDA, 0 };
	managed_tensor->dl_tensor.ndim = ndim;
	managed_tensor->dl_tensor.dtype = DLPackTypeTraits<DataType>::dtype;

	managed_tensor->dl_tensor.shape = shape;
	managed_tensor->dl_tensor.strides = nullptr;
	managed_tensor->dl_tensor.byte_offset = 0;
	managed_tensor->dl_tensor.shape = shape;
	managed_tensor->manager_ctx = nullptr;

	tensor->deleter = [](DLManagedTensor* self) {
		if(if constexpr Device == DeviceType::GPU)
			cudaFree(self->dl_tensor.data);
		else
			delete[] static_cast<DataType*>(self->dl_tensor.data);
		delete[] self->dl_tensor.shape;
		delete self;
	};

	return std::unique_ptr<DLManagedTensor>(managed_tensor);
}
