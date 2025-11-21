#pragma once
#include "pyscheduler/library_export.hpp"
#include "pyscheduler/move_only.hpp"

#include <atomic>
#include <chrono>
#include <filesystem>
#include <future>
#include <iostream>
#include <map>
#include <memory>
#include <blockingconcurrentqueue.h>
#include <pybind11/embed.h>
#include <pybind11/pybind11.h>
#include <queue>
#include <set>
#include <shared_mutex>
#include <string>

using namespace std::chrono;

namespace pyscheduler {
///////////////////////////////////////////////////////////////////////////////
// Python Interpreter Management
//
// Only a single python interpreter is allowed per program due. As a workaround,
// we provide methods to "schedule" function invocations.
///////////////////////////////////////////////////////////////////////////////

// InvokeHandler and PyManager are tightly bound but that's ok since we
// want to tie the lifetime of InvokeHandlers to the lifetime of the PyManager
// to avoid UB

/// @brief Manages the Python interpreter and module/function state.
///
/// Since only one Python Interpreter is allowed per program, PyManager provides a
/// mechanism to load Python modules and enqueue function calls. It also manages
/// a thread pool to asynchronously invoke Python functions; optimizing GIL usage.
class PYSCHEDULER_LIBRARY_EXPORT PyManager {
public:
	/// @brief Handles the invocation of a predefined python function from a loaded module
	///
	/// The InvokeHandler class is tightly coupled with PyManager so that tht lifetime of
	/// InvokeHandlers is tied to that of the PyManager, preventing undefined behavior.
	class PYSCHEDULER_LIBRARY_EXPORT InvokeHandler {
		friend PyManager;

	public:
		/// @brief Synchronously invokes the Python function with given arguments.
		///
		///	This method acquires the GIL and calls the Python function, then casts the result into the
		/// specified return type.
		///
		/// @tparam ReturnType The expected return type after casting the Python result.
		/// @tparam Args Types of the arguments to be forwarded to the Python function.
		/// @param args Arguments to forward to the Python function.
		/// @return The result of the Python function call, cast to ReturnType.
		/// @return Result of the Python function call
		template <typename ReturnType, typename... Args>
		ReturnType invoke(Args&&... args);

		/// @brief Synchronously invokes the Python function and processes its result with a callback.
		///
		/// This method acquires the GIL, calls the Python function with the provided arguments,
		/// and then passes the result to a user-specified callback function.
		///
		///  @tparam Callback Callable type that accepts a pybind11::object.
		///  @tparam Args Types of the arguments to be forwarded to the Python function.
		///  @param callback Function to process the Python function's return value.
		///  @param args Arguments to forward to the Python function.
		///  @return The result produced by the callback function.
		template <typename Callback, typename... Args>
		auto invoke(Callback&& callback, Args&&... args)
			-> std::invoke_result_t<Callback, pybind11::object>;

		/// @brief Asynchronously invokes the Python function and processes its result with a callback.
		///
		/// The function is scheduled to run on a separate worker thread. THe callback is executed once
		/// the Python function returns. A std::future is returned so that the caller can wait for
		/// or retrieve the result of the callback.
		///
		/// @tparam Callback Callable type that accepts a pybind11::object and returns a value convertible to the final result.
		/// @tparam Args Types of the arguments to be forwarded to the Python function.
		/// @param callback Function to process the Python function's return value.
		/// @param args Arguments to forward to the Python function.
		/// @return A std::future that will hold the result of the callback once the Python function completes.
		///
		/// @warning The callback function provided might be used in contexts beyond the lifetime of the
		///	capture objects. As such, the user needs to take extreme care to ensure the captured objects
		/// do not go out of scope until the callback function is executed. A surefire way to do this
		/// is to `wait()` on every promise before returning or exiting from the current scope.
		///
		/// @note Compile with the "-Wno-attributes" flag to disable warnings about capture lifetimes
		template <typename Callback, typename... Args>
		auto queue_invoke(Callback&& callback, Args&&... args)
			-> std::future<std::invoke_result_t<Callback, pybind11::object>>;

	private:
		/// @brief Retrieves the Python module and function associated with this handler.
		/// @return A shared ptr to a pair, where the first element is the Python
		/// module and the second element is the Python function object.
		///
		/// @note This function does not require the GIL because it does not increment the reference
		/// count of the underlying Python objects. If the Pybind11 object copy constructor is called,
		/// the current thread must hold the GIL.
		///
		/// @note Should not be public because we don't want dangling references
		const std::shared_ptr<std::pair<pybind11::module_, pybind11::object>>& getModuleAndFunc();

	private:
		/// @brief
		/// @param id
		/// @param manager
		InvokeHandler(size_t id, std::unique_ptr<PyManager> manager);

		size_t _id;

		// this pointer prevents the destructor for PyManager from
		// finalizing the python interpreter and releasing all imported modules
		// until all InvokeHandlers go out of scope
		std::unique_ptr<PyManager> _manager;
	};

public:
	PyManager();
	PyManager(const PyManager& udl_manager) = delete;
	PyManager(PyManager&& udl_manager) = delete;

	~PyManager();
	PyManager operator=(const PyManager& udl_manager) = delete;
	PyManager operator=(PyManager&& udl_manager) = delete;

public:
	static const size_t NUM_WORKERS = 16;

	/// @brief Loads a Python module and its entry point, or retrieves an existing one.
	/// @param module_name Name of the Python module to load.
	/// @param entry_point Function name to retrieve from the module.
	/// @return An InvokeHandler for calling the specified function
	InvokeHandler loadPythonModule(const std::string& module_name,
								   const std::string& entry_point = "invoke");

private:
	struct PYSCHEDULER_LIBRARY_LOCAL SharedState {
		std::atomic<uint64_t> arc = 0;

		std::shared_mutex py_mutex;
		std::map<std::string, size_t> py_invoke_handler_map;
		std::vector<std::shared_ptr<std::pair<pybind11::module_, pybind11::object>>> py_modules;

		/// @brief extremely fast queue that has several microseconds performance
		/// @note thread safe queue but is not linearizable nor sequentially consistent.
		/// this is ok because all methods are processed independently.
		moodycamel::BlockingConcurrentQueue<MoveOnlyFunction<void()>> task_queue;

		std::atomic<bool> threads_active = true;
		std::thread main_worker;

		std::atomic<bool> interpreter_initialized = false;
	};

	static SharedState _instance;
	inline static SharedState& shared() {
		return _instance;
	}

	void mainLoop();
};

} // namespace pyscheduler

#include "pyscheduler/details/pyscheduler_impl.hpp"