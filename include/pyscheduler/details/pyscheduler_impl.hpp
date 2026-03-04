#ifdef __INTELLISENSE__
#	include "pyscheduler/pyscheduler.hpp"
#endif

#include "pyscheduler/move_only.hpp"
#include <cassert>
#include <chrono>
#include <deque>

namespace pyscheduler {

///////////////////////////////////////////////////////////////////////////////
// Impl InvokeHandler
///////////////////////////////////////////////////////////////////////////////

PyManager::InvokeHandler::InvokeHandler(size_t id,
										std::shared_ptr<pybind11::object> resource,
										std::unique_ptr<PyManager> manager,
										size_t batch_size,
										size_t prefetch_depth)
	: _id(id)
	, _manager(std::move(manager))
	, _state(std::make_shared<WorkerState>(std::move(resource), batch_size, prefetch_depth))
	, _worker(&InvokeHandler::workerLoop, _state) { }

PyManager::InvokeHandler::~InvokeHandler() {
	if(_state) _state->active.store(false);
	if(_worker.joinable()) _worker.join();
}

PyManager::InvokeHandler::InvokeHandler(InvokeHandler&& other) noexcept
	: _id(other._id)
	, _manager(std::move(other._manager))
	, _state(std::move(other._state))
	, _worker(std::move(other._worker)) { }

PyManager::InvokeHandler& PyManager::InvokeHandler::operator=(InvokeHandler&& other) noexcept {
	if(this != &other) {
		if(_state) _state->active.store(false);
		if(_worker.joinable()) _worker.join();

		_id = other._id;
		_manager = std::move(other._manager);
		_state = std::move(other._state);
		_worker = std::move(other._worker);
	}
	return *this;
}

template <typename ReturnType, typename... Args>
ReturnType PyManager::InvokeHandler::invoke(Args&&... args) {
	pybind11::gil_scoped_acquire gil;
	pybind11::object result = (*_state->resource.get())(std::forward<Args>(args)...);
	return result.cast<ReturnType>();
}

template <typename Callback, typename... Args>
auto PyManager::InvokeHandler::invoke(Callback&& callback, Args&&... args)
	-> std::invoke_result_t<Callback, pybind11::object> {
	pybind11::gil_scoped_acquire gil;
	pybind11::object result = (*_state->resource.get())(std::forward<Args>(args)...);
	return callback(std::move(result));
}

template <typename CommitFn, typename Callback, typename... Args>
auto PyManager::InvokeHandler::queue_invoke(CommitFn&& commit_fn,
											Callback&& callback,
											Args&&... args)
	-> std::future<std::invoke_result_t<Callback, pybind11::object>> {
	using ReturnType = std::invoke_result_t<Callback, pybind11::object>;

	static_assert(
		!std::is_same_v<ReturnType, pybind11::object>,
		"ReturnType must not be pybind11::object; convert to a pure C++ type in the callback.");

	auto args_tuple = std::make_tuple(std::forward<Args>(args)...);

	auto promise = std::make_shared<std::promise<ReturnType>>();
	auto future = promise->get_future();

	// Type-erase commit: captures commit_fn + args, returns pybind11::object
	auto commit = [commit_fn = std::forward<CommitFn>(commit_fn),
				   args = std::move(args_tuple)]() mutable -> pybind11::object {
		return std::apply(
			[&commit_fn](auto&&... unpacked) -> pybind11::object {
				return commit_fn(std::forward<decltype(unpacked)>(unpacked)...);
			},
			std::move(args));
	};

	// Type-erase callback: captures callback + promise, processes one result
	auto on_result = [cb = std::forward<Callback>(callback),
					  promise](pybind11::object result) mutable {
		try {
			ReturnType value = std::invoke(cb, std::move(result));
			promise->set_value(std::move(value));
		} catch(...) {
			promise->set_exception(std::current_exception());
		}
	};

	// Error path: propagates exception to the future
	auto on_error = [promise](std::exception_ptr eptr) { promise->set_exception(eptr); };

	_state->queue.enqueue(
		QueueEntry{ std::move(commit), std::move(on_result), std::move(on_error) });

	return future;
}

///////////////////////////////////////////////////////////////////////////////
// InvokeHandler Worker Loop
///////////////////////////////////////////////////////////////////////////////

void PyManager::InvokeHandler::workerLoop(std::shared_ptr<WorkerState> state) {
	std::deque<CommittedEntry> prefetch_buffer;

	while(state->active || state->queue.size_approx() > 0 || !prefetch_buffer.empty()) {

		// Block-wait only when prefetch buffer is empty and queue is empty
		if(prefetch_buffer.empty() && state->queue.size_approx() == 0) {
			QueueEntry entry;
			bool got = state->queue.wait_dequeue_timed(entry, std::chrono::milliseconds(100));
			if(!got) continue;
			// Re-enqueue so the main loop below picks it up uniformly
			state->queue.enqueue(std::move(entry));
		}

		{ // GIL scope
			pybind11::gil_scoped_acquire gil;

			// Phase 1: Refill prefetch buffer up to batch_size * prefetch_depth
			while(prefetch_buffer.size() < state->batch_size * state->prefetch_depth) {
				QueueEntry entry;
				if(!state->queue.try_dequeue(entry)) break;

				try {
					pybind11::object committed = entry.commit();
					prefetch_buffer.push_back(CommittedEntry{ std::move(committed),
															  std::move(entry.on_result),
															  std::move(entry.on_error) });
				} catch(...) {
					entry.on_error(std::current_exception());
				}
			}

			// Phase 2: Execute batch — consume up to batch_size items (opportunistic)
			size_t batch_target = std::min(state->batch_size, prefetch_buffer.size());

			if(batch_target > 0) {
				pybind11::list batch;
				std::vector<MoveOnlyFunction<void(pybind11::object)>> result_callbacks;
				std::vector<MoveOnlyFunction<void(std::exception_ptr)>> error_callbacks;
				result_callbacks.reserve(batch_target);
				error_callbacks.reserve(batch_target);

				for(size_t i = 0; i < batch_target; i++) {
					batch.append(std::move(prefetch_buffer.front().committed_obj));
					result_callbacks.push_back(std::move(prefetch_buffer.front().on_result));
					error_callbacks.push_back(std::move(prefetch_buffer.front().on_error));
					prefetch_buffer.pop_front();
				}

				try {
					pybind11::object results = (*state->resource)(batch);

					// Phase 3: Fan-out — dispatch each result to its callback
					for(size_t i = 0; i < result_callbacks.size(); i++) {
						result_callbacks[i](results[pybind11::int_(i)]);
					}
				} catch(...) {
					auto eptr = std::current_exception();
					for(size_t i = 0; i < error_callbacks.size(); i++) {
						error_callbacks[i](eptr);
					}
				}
			}
		} // GIL released
	}

	// Clean up any remaining pybind11 objects with GIL held
	if(!prefetch_buffer.empty()) {
		pybind11::gil_scoped_acquire gil;
		prefetch_buffer.clear();
	}
}

///////////////////////////////////////////////////////////////////////////////
// Impl PyManager
///////////////////////////////////////////////////////////////////////////////

PyManager::SharedState PyManager::_instance;

PyManager::PyManager() {
	if(shared().arc.fetch_add(1) == 0) {
		shared().destructor_thread = std::thread(&PyManager::mainLoop, this);
	}
	// small cost paid to block until interpreter is initialized
	while(!shared().interpreter_initialized)
		continue;
}

PyManager::~PyManager() {
	shared().arc--;
	if(shared().arc == 0) {
		shared().threads_active = false;

		// main worker handles interpreter cleanup
		// https://docs.python.org/3/c-api/init.html#c.Py_FinalizeEx
		//   Py_FinalizeEx should be called in the same thread as Py_InitializeEx
		{
			std::lock_guard<std::mutex> lock(shared().destructor_mutex);
			shared().threads_active = false;
		}
		shared().destructor_cv.notify_all();
		shared().destructor_thread.join();
	}
}

PyManager::InvokeHandler PyManager::loadPythonModule(const std::string& module_name,
													 const std::string& entry_point,
													 size_t batch_size,
													 size_t prefetch_depth) {

	if(!shared().interpreter_initialized) {
		throw std::runtime_error("Python interpreter not initialized");
	}

	SharedState& state = shared();

	pybind11::gil_scoped_acquire gil;

	auto module_it = state.py_invoke_handler_map.find(module_name);
	if(module_it == state.py_invoke_handler_map.end()) {
		pybind11::module_ mod;
		try {
			mod = pybind11::module_::import(module_name.c_str());
		} catch(pybind11::error_already_set& e) {
			throw std::invalid_argument("Could not import module: " + module_name);
		}

		auto [inserted_it, successful] =
			state.py_invoke_handler_map.emplace(module_name, PyInvokeHandlerEntry{ mod, {} });
		module_it = inserted_it;

		if(!successful) {
			throw std::runtime_error("Could not insert module: " + module_name);
		}
	}

	auto object_it = module_it->second.handler_map.find(entry_point);
	if(object_it == module_it->second.handler_map.end()) {
		pybind11::object obj;
		try {
			obj = module_it->second.module_.attr(entry_point.c_str());
		} catch(pybind11::error_already_set& e) {
			throw std::invalid_argument("Could not find the '" + entry_point +
										"' method in module " + module_name);
		}

		size_t id = state.py_objects.size();
		auto obj_ptr = state.py_objects.emplace_back(std::make_shared<pybind11::object>(obj));
		module_it->second.handler_map[module_name] = id;

		return PyManager::InvokeHandler(
			id, obj_ptr, std::make_unique<PyManager>(), batch_size, prefetch_depth);
	}

	size_t id = object_it->second;
	return PyManager::InvokeHandler(
		id, state.py_objects[id], std::make_unique<PyManager>(), batch_size, prefetch_depth);
}

void PyManager::mainLoop() {
	if(shared().interpreter_initialized) {
		throw std::runtime_error(
			"Cannot reinitialize Python interpreter once it has been shut down.");
	}

	// Do not register python signal handlers
	// https://docs.python.org/3/c-api/init.html#c.Py_InitializeEx
	pybind11::initialize_interpreter(false);

	pybind11::module_ sys = pybind11::module_::import("sys");
	pybind11::module_ atexit = pybind11::module_::import("atexit");

	pybind11::list(sys.attr("path")).append(".");
	int major = pybind11::list(sys.attr("version_info"))[0].cast<int>();
	int minor = pybind11::list(sys.attr("version_info"))[1].cast<int>();

	{
		// Release GIL so InvokeHandler worker threads can acquire it
		pybind11::gil_scoped_release gil;

		shared().interpreter_initialized.store(true);

		std::unique_lock<std::mutex> lock(shared().destructor_mutex);
		shared().destructor_cv.wait(lock, [] { return !shared().threads_active; });
	} // GIL reacquired

	shared().py_invoke_handler_map.clear();
	shared().py_objects.clear();

	pybind11::module_ gc = pybind11::module_::import("gc");
	gc.attr("collect")();
	gc.attr("collect")();
	gc.attr("collect")();
	gc.attr("collect")();
	gc.attr("collect")();

	atexit = {};
	sys = {};
	gc = {};

	pybind11::finalize_interpreter();
}
} // namespace pyscheduler
