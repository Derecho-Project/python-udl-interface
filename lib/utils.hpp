#pragma once

#include "lib/python_manager.hpp"
#include <Python.h>

namespace pyscheduler {
namespace utils {

class ScopedPyGIL {
private:
	PyGILState_STATE _gil;

public:
	ScopedPyGIL()
		: _gil(PyGILState_Ensure()) { }
	~ScopedPyGIL() {
		PyGILState_Release(_gil);
	}
};

template <typename Container>
PyObjectShared asPyList(const Container& container) {
	ScopedPyGIL gil;

	PyObject* py_list = PyList_New(container.size());

	if(!py_list) return nullptr;

	size_t index = 0;
	for(const auto& item : container) {
		PyObject* py_item = nullptr;

		if constexpr(std::is_same_v<typename Container::value_type, long long>) {
			py_item = PyLong_FromLongLong(item);
		} else if constexpr(std::is_integral_v<typename Container::value_type>) {
			py_item = PyLong_FromLongLong(static_cast<long>(item));
		} else if constexpr(std::is_floating_point_v<typename Container::value_type>) {
			py_item = PyLong_FromDouble(static_cast<double>(item));
		} else if constexpr(std::is_same_v<typename Container::value_type, std::string>) {
			py_item = PyUnicode_FromString(item.c_str());
		} else {
			Py_DECREF(py_list);
			PyErr_SetString(PyExc_TypeError, "Unsupported container element type.");
			return nullptr;
		}

		if(!py_item) {
			Py_DECREF(py_list);
			return nullptr;
		}

		PyList_SET_ITEM(py_list, index++, py_item); // Steals reference to py_item
	}
	return make_pyobject(py_list);
}

template <typename T>
PyObjectShared asPyNumeric(const T& value) {
	ScopedPyGIL gil;

	static_assert(std::is_integral_v<T> || std::is_floating_point_v<T>,
				  "Value must be either integral or floating type");

	PyObject* py_item = nullptr;

	if constexpr(std::is_same_v<T, long long>) {
		py_item = PyLong_FromLongLong(value);
	} else if constexpr(std::is_integral_v<T>) {
		py_item = PyLong_FromLong(static_cast<long>(value));
	} else {
		py_item = PyLong_FromDouble(static_cast<double>(value));
	}

	return make_pyobject(py_item);
}

template <typename... Args>
PyObjectShared asPyArgs(Args&&... args) {
	static_assert((std::is_same_v<std::decay_t<Args>, PyObjectShared> && ...),
				  "All arguments must be of type PyObjectShared");

	ScopedPyGIL gil;
	return make_pyobject(PyTuple_Pack(sizeof...(Args), args.get()...));
}

} // namespace utils
} // namespace pyscheduler