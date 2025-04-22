#pragma once
#include "pyscheduler/library_export.hpp"

#include <atomic>
#include <chrono>
#include <moodycamel/blockingconcurrentqueue.h>
#include <filesystem>
#include <future>
#include <iostream>
#include <map>
#include <memory>
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
		/// @brief Retrieves the Python module and function associated with this handler.
		/// @return A shared ptr to a pair, where the first element is the Python
		/// module and the second element is the Python function object.
		///
		/// @note This function does not require the GIL because it does not increment the reference
		/// count of the underlying Python objects. If the Pybind11 object copy constructor is called,
		/// the current thread must hold the GIL.
		const std::shared_ptr<std::pair<pybind11::module_, pybind11::object>> getModuleAndFunc();

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
		ReturnType invoke(Args&&... args) {
			auto mod_and_func = getModuleAndFunc();
			pybind11::gil_scoped_acquire gil;
			pybind11::object result = mod_and_func->second(std::forward<Args>(args)...);
			return result.cast<ReturnType>();
		}

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
		auto invoke(Callback callback,
					Args&&... args) -> std::invoke_result_t<Callback, pybind11::object> {
			auto mod_and_func = getModuleAndFunc();
			pybind11::gil_scoped_acquire gil;
			pybind11::object result = mod_and_func->second(std::forward<Args>(args)...);
			return callback(result);
		}

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
		auto queue_invoke(Callback callback, Args&&... args)
			-> std::future<std::invoke_result_t<Callback, pybind11::object>> {

			// Need to wrap a promise inside a shared_ptr because Promises are not
			// copy constructable (requirement enforced by appending to task queue)
			//
			// solution was to wrap a promise inside a shared pointer, which is
			// copy constructable
			using ReturnType = std::invoke_result_t<Callback, pybind11::object>;
			using PromisePtr = std::shared_ptr<std::promise<ReturnType>>;

			auto args_tuple = std::make_tuple(std::forward<Args>(args)...);
			PromisePtr promise_ptr = std::make_shared<std::promise<ReturnType>>();
			std::future<ReturnType> future = promise_ptr->get_future();

			// Dear reader, I'm sorry
			// this section creates a closure that executes a python method with
			// the provided arguments
			//
			// the return result from the python function is processed using the
			// callback function, and the value from that is stored into the
			// promise.
			auto method =
				[this, callback, args_tuple = std::move(args_tuple), promise_ptr]() mutable {
					auto mod_and_func = getModuleAndFunc();
					pybind11::object result = std::apply(
						[&mod_and_func](auto&&... unpackedArgs) {
							pybind11::gil_scoped_acquire gil;
							return mod_and_func->second(
								std::forward<decltype(unpackedArgs)>(unpackedArgs)...);
						},
						args_tuple);

					promise_ptr->set_value(callback(result));
				};
			PyManager::shared().task_queue.enqueue(std::move(method));
			return future;
		}

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
	InvokeHandler getPythonModule(const std::string& module_name,
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
		moodycamel::BlockingConcurrentQueue<std::function<void()>> task_queue;

		std::atomic<bool> threads_active = true;
		std::thread main_worker;

		std::atomic<bool> interpreter_initialized = false;
	};

	inline static SharedState& shared() {
		static SharedState instance;
		return instance;
	}

	void mainLoop();
};

} // namespace pyscheduler