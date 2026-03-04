#pragma once
#include "pyscheduler/library_export.hpp"
#include "pyscheduler/move_only.hpp"

#include <atomic>
#include <blockingconcurrentqueue.h>
#include <chrono>
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
	/// The InvokeHandler class is tightly coupled with PyManager so that the lifetime of
	/// InvokeHandlers is tied to that of the PyManager, preventing undefined behavior.
	///
	/// Each InvokeHandler owns a dedicated worker thread that manages GIL acquisition,
	/// batching, and prefetching of pybind11 objects.
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

		/// @brief Asynchronously enqueues a Python function call with batching support.
		///
		/// The commit function converts C++ arguments into a pybind11::object (which may
		/// involve GPU memory initialization or Python object construction). Items are
		/// batched according to batch_size: the Python function receives a list of
		/// committed objects and must return a list of results. Each result is dispatched
		/// to its corresponding callback (fan-in/fan-out).
		///
		/// @tparam CommitFn Callable: (Args...) -> pybind11::object
		/// @tparam Callback Callable: (pybind11::object) -> ReturnType
		/// @tparam Args Types of the arguments captured and later passed to commit.
		/// @param commit Function that converts C++ args into a pybind11::object.
		/// @param callback Function to process each individual result from the batch.
		/// @param args Arguments to forward to the commit function.
		/// @return A std::future holding the result of the callback for this item.
		template <typename CommitFn, typename Callback, typename... Args>
		auto queue_invoke(CommitFn&& commit, Callback&& callback, Args&&... args)
			-> std::future<std::invoke_result_t<Callback, pybind11::object>>;

		~InvokeHandler();
		InvokeHandler(InvokeHandler&& other) noexcept;
		InvokeHandler& operator=(InvokeHandler&& other) noexcept;
		InvokeHandler(const InvokeHandler&) = delete;
		InvokeHandler& operator=(const InvokeHandler&) = delete;

	private:
		struct QueueEntry {
			MoveOnlyFunction<pybind11::object()> commit;
			MoveOnlyFunction<void(pybind11::object)> on_result;
			MoveOnlyFunction<void(std::exception_ptr)> on_error;
		};

		struct CommittedEntry {
			pybind11::object committed_obj;
			MoveOnlyFunction<void(pybind11::object)> on_result;
			MoveOnlyFunction<void(std::exception_ptr)> on_error;
		};

		struct WorkerState {
			std::shared_ptr<pybind11::object> resource;
			size_t batch_size;
			size_t prefetch_depth;
			moodycamel::BlockingConcurrentQueue<QueueEntry> queue;
			std::atomic<bool> active{ true };

			WorkerState(std::shared_ptr<pybind11::object> res, size_t bs, size_t pd)
				: resource(std::move(res))
				, batch_size(bs)
				, prefetch_depth(pd) { }
		};

		InvokeHandler(size_t id,
					  std::shared_ptr<pybind11::object> resource,
					  std::unique_ptr<PyManager> manager,
					  size_t batch_size,
					  size_t prefetch_depth);

		static void workerLoop(std::shared_ptr<WorkerState> state);

		size_t _id;

		// prevents PyManager destructor from finalizing the interpreter
		// until all InvokeHandlers go out of scope
		std::unique_ptr<PyManager> _manager;

		std::shared_ptr<WorkerState> _state;
		std::thread _worker;
	};

public:
	PyManager();
	PyManager(const PyManager& udl_manager) = delete;
	PyManager(PyManager&& udl_manager) = delete;

	~PyManager();
	PyManager operator=(const PyManager& udl_manager) = delete;
	PyManager operator=(PyManager&& udl_manager) = delete;

public:
	/// @brief Loads a Python module and its entry point, or retrieves an existing one.
	/// @param module_name Name of the Python module to load.
	/// @param entry_point Function name to retrieve from the module.
	/// @param batch_size Number of items per batched Python call.
	/// @param prefetch_depth Number of batches to pre-commit as pybind11 objects.
	/// @return An InvokeHandler for calling the specified function
	InvokeHandler loadPythonModule(const std::string& module_name,
								   const std::string& entry_point = "invoke",
								   size_t batch_size = 1,
								   size_t prefetch_depth = 1);

private:
	struct PYSCHEDULER_LIBRARY_LOCAL PyInvokeHandlerEntry {
		pybind11::module_ module_;
		std::unordered_map<std::string, size_t> handler_map;
	};

	struct PYSCHEDULER_LIBRARY_LOCAL SharedState {
		std::atomic<uint64_t> arc = 0;

		/// @brief stores all loaded modules and objects
		std::unordered_map<std::string, PyInvokeHandlerEntry> py_invoke_handler_map;

		std::vector<std::shared_ptr<pybind11::object>> py_objects;

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