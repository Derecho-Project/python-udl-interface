#ifdef __INTELLISENSE__
#	include "pyscheduler/pyscheduler.hpp"
#endif

#include "pyscheduler/move_only.hpp"
#include <chrono>
#include <cstdlib>
#include <deque>
#include <iostream>
#include <stdexcept>
#include <string_view>
#include <thread>

namespace pyscheduler {

///////////////////////////////////////////////////////////////////////////////
// Impl InvokeHandler
///////////////////////////////////////////////////////////////////////////////

PyManager::InvokeHandler::InvokeHandler(size_t /*id*/,
										std::shared_ptr<pybind11::object> resource,
										std::unique_ptr<PyManager> manager,
										size_t batch_size,
										size_t prefetch_depth)
	: _manager(std::move(manager))
	, _resource(std::move(resource))
	, _batch_size(batch_size)
	, _prefetch_depth(prefetch_depth)
	, _active(std::make_shared<std::atomic<bool>>(true))
	, _state(std::make_shared<WorkerState>())
	, _worker(
		  &InvokeHandler::workerLoop, _state, _resource, _batch_size, _prefetch_depth, _active) { }

PyManager::InvokeHandler::~InvokeHandler() {
	if(_active) {
		_active->store(false);
	}
	if(_worker.joinable()) _worker.join();
	if(_state) {
		pybind11::gil_scoped_acquire gil;
		_state.reset();
	}
}

PyManager::InvokeHandler::InvokeHandler(InvokeHandler&& other) noexcept
	: _manager(std::move(other._manager))
	, _resource(std::move(other._resource))
	, _batch_size(other._batch_size)
	, _prefetch_depth(other._prefetch_depth)
	, _active(std::move(other._active))
	, _state(std::move(other._state))
	, _worker(std::move(other._worker)) {
	if(!_resource) {
		std::cerr << "InvokeHandler has no bound Python callable." << std::endl;
		std::abort();
	}
}

PyManager::InvokeHandler& PyManager::InvokeHandler::operator=(InvokeHandler&& other) noexcept {
	if(this != &other) {
		if(_active) {
			_active->store(false);
		}
		if(_worker.joinable()) _worker.join();
		if(_state) {
			pybind11::gil_scoped_acquire gil;
			_state.reset();
		}

		_manager = std::move(other._manager);
		_resource = std::move(other._resource);
		_batch_size = other._batch_size;
		_prefetch_depth = other._prefetch_depth;
		_active = std::move(other._active);
		_state = std::move(other._state);
		_worker = std::move(other._worker);
	}
	return *this;
}

template <typename ReturnType, typename... Args>
ReturnType PyManager::InvokeHandler::invoke(Args&&... args) {
	pybind11::gil_scoped_acquire gil;
	pybind11::object result = (*_resource.get())(std::forward<Args>(args)...);
	return result.cast<ReturnType>();
}

template <typename Callback, typename... Args>
auto PyManager::InvokeHandler::invoke(Callback&& callback, Args&&... args)
	-> std::invoke_result_t<Callback, pybind11::object> {
	pybind11::gil_scoped_acquire gil;
	pybind11::object result = (*_resource.get())(std::forward<Args>(args)...);
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
			if constexpr(std::is_void_v<ReturnType>) {
				std::invoke(cb, std::move(result));
				promise->set_value();
			} else {
				ReturnType value = std::invoke(cb, std::move(result));
				promise->set_value(std::move(value));
			}
		} catch(...) {
			promise->set_exception(std::current_exception());
		}
	};

	// Error path: propagates exception to the future
	auto on_error = [promise](std::exception_ptr eptr) { promise->set_exception(eptr); };

	_state->commit_queue.enqueue(
		QueueEntry{ std::move(commit), std::move(on_result), std::move(on_error) });
	_state->total_enqueued.fetch_add(1, std::memory_order_relaxed);

	return future;
}

inline PyManager::InvokeHandler::QueueStats
PyManager::InvokeHandler::get_queue_stats() const {
	QueueStats stats;
	stats.commit_queue_size = _state->commit_queue.size_approx();
	stats.execute_queue_size = _state->execute_queue_size.load(std::memory_order_relaxed);
	stats.total_enqueued = _state->total_enqueued.load(std::memory_order_relaxed);
	{
		std::lock_guard<std::mutex> lock(_state->stats_mutex);
		stats.commit_batch_size_ema = _state->commit_batch_size_ema;
		stats.execute_batch_size_ema = _state->execute_batch_size_ema;
		stats.commit_ns_per_batch_ema = _state->commit_ns_per_batch_ema;
		stats.execute_ns_per_batch_ema = _state->execute_ns_per_batch_ema;
	}
	return stats;
}

///////////////////////////////////////////////////////////////////////////////
// InvokeHandler Worker Loop
///////////////////////////////////////////////////////////////////////////////

inline void PyManager::InvokeHandler::workerLoop(std::shared_ptr<WorkerState> state,
												 std::shared_ptr<pybind11::object> resource,
												 size_t batch_size,
												 size_t prefetch_depth,
												 std::shared_ptr<std::atomic<bool>> active) {
	std::deque<CommittedEntry> prefetch_buffer;
	const size_t buffer_capacity = batch_size * prefetch_depth;
	constexpr double kEmaAlpha = 0.1;

	auto blend = [](double ema, double sample, bool first) {
		return first ? sample : kEmaAlpha * sample + (1.0 - kEmaAlpha) * ema;
	};

	while(active->load() || state->commit_queue.size_approx() > 0 || !prefetch_buffer.empty()) {

		// Block-wait only when prefetch buffer is empty and queue is empty
		if(prefetch_buffer.empty() && state->commit_queue.size_approx() == 0) {
			std::this_thread::sleep_for(std::chrono::milliseconds(5));
			continue;
		}

		{ // GIL scope
			pybind11::gil_scoped_acquire gil;

			// Phase 1: Refill prefetch buffer up to batch_size * prefetch_depth
			size_t commit_count = 0;
			auto commit_start = std::chrono::steady_clock::now();
			while(prefetch_buffer.size() < buffer_capacity) {
				QueueEntry entry;
				if(!state->commit_queue.try_dequeue(entry)) break;

				try {
					pybind11::object committed = entry.commit();
					prefetch_buffer.push_back(CommittedEntry{ std::move(committed),
														  std::move(entry.on_result),
														  std::move(entry.on_error) });
					commit_count++;
				} catch(...) {
					try {
						entry.on_error(std::current_exception());
					} catch(...) {
					}
				}
			}
			auto commit_end = std::chrono::steady_clock::now();

			if(commit_count > 0) {
				double commit_ns =
					static_cast<double>(std::chrono::duration_cast<std::chrono::nanoseconds>(
											commit_end - commit_start)
											.count());
				std::lock_guard<std::mutex> lock(state->stats_mutex);
				const bool first = !state->has_commit_sample;
				state->commit_batch_size_ema = blend(
					state->commit_batch_size_ema, static_cast<double>(commit_count), first);
				state->commit_ns_per_batch_ema =
					blend(state->commit_ns_per_batch_ema, commit_ns, first);
				state->has_commit_sample = true;
			}
			state->execute_queue_size.store(prefetch_buffer.size(), std::memory_order_relaxed);

			// Phase 2: Execute batch — consume up to batch_size items (opportunistic)
			size_t batch_target = std::min(batch_size, prefetch_buffer.size());

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
				state->execute_queue_size.store(prefetch_buffer.size(),
												std::memory_order_relaxed);

				auto execute_start = std::chrono::steady_clock::now();
				try {
					pybind11::object results = (*resource)(batch);

					// Phase 3: Fan-out — dispatch each result to its callback
					for(size_t i = 0; i < result_callbacks.size(); i++) {
						try {
							result_callbacks[i](results[pybind11::int_(i)]);
						} catch(...) {
						}
					}
				} catch(...) {
					auto eptr = std::current_exception();
					for(size_t i = 0; i < error_callbacks.size(); i++) {
						try {
							error_callbacks[i](eptr);
						} catch(...) {
						}
					}
				}
				auto execute_end = std::chrono::steady_clock::now();

				double execute_ns =
					static_cast<double>(std::chrono::duration_cast<std::chrono::nanoseconds>(
											execute_end - execute_start)
											.count());
				std::lock_guard<std::mutex> lock(state->stats_mutex);
				const bool first = !state->has_execute_sample;
				state->execute_batch_size_ema = blend(
					state->execute_batch_size_ema, static_cast<double>(batch_target), first);
				state->execute_ns_per_batch_ema =
					blend(state->execute_ns_per_batch_ema, execute_ns, first);
				state->has_execute_sample = true;
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

PyManager::PyManager() {
	shared().arc.fetch_add(1, std::memory_order_acq_rel);

	// The Python interpreter is initialized exactly once per process and the
	// owning thread is parked forever. This keeps a valid PyThreadState alive
	// for the lifetime of the process so that static destructors in libraries
	// such as libtorch_python (which decref Python objects at exit) can safely
	// acquire the GIL.
	std::call_once(shared().init_flag, [] {
		std::thread([] { PyManager::mainLoop(); }).detach();
		while(!shared().interpreter_initialized.load(std::memory_order_acquire))
			continue;

		// Release any cached pybind11 handles before static destructors run,
		// so dec_ref happens with the GIL held.
		std::atexit([] {
			if(!shared().interpreter_initialized.load(std::memory_order_acquire)) return;
			pybind11::gil_scoped_acquire gil;
			shared().py_invoke_handler_map.clear();
		});
	});
}

PyManager::~PyManager() {
	shared().arc.fetch_sub(1, std::memory_order_acq_rel);
	// Interpreter is intentionally not finalized; see PyManager() for rationale.
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
			throw std::invalid_argument("Could not import module '" + module_name +
										"': " + std::string(e.what()));
		}

		auto [inserted_it, successful] =
			state.py_invoke_handler_map.emplace(module_name, PyInvokeHandlerEntry{ mod, { } });
		module_it = inserted_it;
	}

	auto object_it = module_it->second.handler_map.find(entry_point);
	if(object_it == module_it->second.handler_map.end()) {
		pybind11::object obj;
		try {
			obj = module_it->second.module_.attr(entry_point.c_str());
		} catch(pybind11::error_already_set& e) {
			throw std::invalid_argument("Could not find the '" + entry_point +
										"' method in module " + module_name +
										": " + std::string(e.what()));
		}

		auto obj_ptr = std::make_shared<pybind11::object>(obj);
		auto [handler_it, handler_inserted] =
			module_it->second.handler_map.emplace(entry_point, obj_ptr);
		if(!handler_inserted) {
			obj_ptr = handler_it->second;
		}
		size_t id = module_it->second.handler_map.size() - 1;

		return PyManager::InvokeHandler(
			id, obj_ptr, std::make_unique<PyManager>(), batch_size, prefetch_depth);
	}

	size_t id = 0;
	return PyManager::InvokeHandler(
		id, object_it->second, std::make_unique<PyManager>(), batch_size, prefetch_depth);
}

void PyManager::add_path(const std::string& directory) {
	if(directory.empty()) {
		throw std::invalid_argument("Path cannot be empty");
	}

	if(!shared().interpreter_initialized) {
		throw std::runtime_error("Python interpreter not initialized");
	}

	SharedState& state = shared();

	pybind11::gil_scoped_acquire gil;

	constexpr const char* kSysModule = "sys";
	constexpr const char* kPathObjectKey = "__sys_path__";

	auto module_it = state.py_invoke_handler_map.find(kSysModule);
	if(module_it == state.py_invoke_handler_map.end()) {
		pybind11::module_ mod;
		try {
			mod = pybind11::module_::import(kSysModule);
		} catch(pybind11::error_already_set& e) {
			throw std::runtime_error("Could not import module: sys");
		}

		auto [inserted_it, successful] =
			state.py_invoke_handler_map.emplace(kSysModule, PyInvokeHandlerEntry{ mod, { } });
		module_it = inserted_it;
	}

	auto path_it = module_it->second.handler_map.find(kPathObjectKey);
	if(path_it == module_it->second.handler_map.end()) {
		pybind11::object path_obj;
		try {
			path_obj = module_it->second.module_.attr("path");
		} catch(pybind11::error_already_set& e) {
			throw std::runtime_error("Could not access sys.path");
		}

		auto path_ptr = std::make_shared<pybind11::object>(std::move(path_obj));
		module_it->second.handler_map[kPathObjectKey] = path_ptr;
		path_it = module_it->second.handler_map.find(kPathObjectKey);
	}

	pybind11::list sys_path = path_it->second->cast<pybind11::list>();

	for(auto item : sys_path) {
		if(pybind11::str(item).cast<std::string>() == directory) {
			return;
		}
	}

	sys_path.append(pybind11::str(directory));
}

uintptr_t PyManager::debug_shared_state_address() {
	return reinterpret_cast<uintptr_t>(&shared());
}

uint64_t PyManager::debug_arc_count() {
	return shared().arc.load();
}

void PyManager::mainLoop() {
	// Do not register python signal handlers
	// https://docs.python.org/3/c-api/init.html#c.Py_InitializeEx
	pybind11::initialize_interpreter(false);

	pybind11::module_ sys = pybind11::module_::import("sys");
	(void)pybind11::module_::import("atexit");

	pybind11::list(sys.attr("path")).append(".");

	// Release the GIL so InvokeHandler worker threads (and any other thread
	// that calls into Python via pybind11::gil_scoped_acquire) can acquire it.
	// The PyThreadState owned by this thread remains valid for the lifetime of
	// the process; we never let this function return.
	(void)PyEval_SaveThread();

	shared().interpreter_initialized.store(true, std::memory_order_release);

	// Park forever. The OS reaps this thread at process exit. Keeping it alive
	// preserves a valid Python thread state for static destructors that decref
	// Python objects (e.g. libtorch_python).
	std::promise<void> never;
	never.get_future().wait();
}
} // namespace pyscheduler
