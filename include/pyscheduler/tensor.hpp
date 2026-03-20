#pragma once

#include "pyscheduler/library_export.hpp"
#include <cuda_runtime.h>
#include <dlpack/dlpack.h>

#include <cstdint>
#include <memory>
#include <stdexcept>
#include <vector>

namespace pyscheduler {

template <typename T>
struct DLPackTypeTraits;

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

template <typename T>
inline DLManagedTensor*
createCudaMatrixDlpack(const std::vector<T>& host_data, int64_t rows, int64_t cols) {
	if(host_data.size() != static_cast<size_t>(rows * cols)) {
		throw std::invalid_argument("host_data size does not match rows*cols");
	}

	auto* managed = new DLManagedTensor();
	auto* shape = new int64_t[2]{ rows, cols };

	T* device_data = nullptr;
	cudaError_t alloc_err =
		cudaMalloc(reinterpret_cast<void**>(&device_data), host_data.size() * sizeof(T));
	if(alloc_err != cudaSuccess) {
		delete[] shape;
		delete managed;
		throw std::runtime_error("cudaMalloc failed");
	}

	cudaError_t copy_err = cudaMemcpy(
		device_data, host_data.data(), host_data.size() * sizeof(T), cudaMemcpyHostToDevice);
	if(copy_err != cudaSuccess) {
		cudaFree(device_data);
		delete[] shape;
		delete managed;
		throw std::runtime_error("cudaMemcpy failed");
	}

	managed->dl_tensor.data = device_data;
	managed->dl_tensor.device = DLDevice{ kDLCUDA, 0 };
	managed->dl_tensor.ndim = 2;
	managed->dl_tensor.dtype = DLPackTypeTraits<T>::dtype;
	managed->dl_tensor.shape = shape;
	managed->dl_tensor.strides = nullptr;
	managed->dl_tensor.byte_offset = 0;
	managed->manager_ctx = nullptr;
	managed->deleter = [](DLManagedTensor* self) {
		cudaFree(self->dl_tensor.data);
		delete[] self->dl_tensor.shape;
		delete self;
	};

	return managed;
}

} // namespace pyscheduler
