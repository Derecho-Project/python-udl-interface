#ifdef __INTELLISENSE__
#	include "pyscheduler/pyscheduler.hpp"
#endif

#include "pyscheduler/move_only.hpp"
#include <chrono>
#include <cstdlib>
#include <deque>
#include <stdexcept>
#include <string_view>
#include <thread>

namespace pyscheduler {

void PyManager::InvokeHandler::WorkerState::enqueue(QueueEntry entry) {
	std::lock_guard<std::mutex> lock(queue_mutex);
	if(queued_request_ids.find(entry.request_id) != queued_request_ids.end()) {
		throw std::invalid_argument("Duplicate request_id already exists in queue.");
	}
	queued_request_ids.insert(entry.request_id);
	queued_entries.push_back(std::move(entry));
}

bool PyManager::InvokeHandler::WorkerState::commit(QueueEntry& out) {
	std::lock_guard<std::mutex> lock(queue_mutex);
	if(queued_entries.empty()) return false;
	out = std::move(queued_entries.front());
	queued_request_ids.erase(out.request_id);
	queued_entries.pop_front();
	return true;
}

size_t PyManager::InvokeHandler::WorkerState::commit(size_t max_items,
													 std::vector<QueueEntry>& out) {
	if(max_items == 0) return 0;
	std::lock_guard<std::mutex> lock(queue_mutex);
	const size_t n = std::min(max_items, queued_entries.size());
	out.reserve(out.size() + n);
	for(size_t i = 0; i < n; i++) {
		out.push_back(std::move(queued_entries.front()));
		queued_request_ids.erase(out.back().request_id);
		queued_entries.pop_front();
	}
	return n;
}

size_t PyManager::InvokeHandler::WorkerState::queued_size() const {
	std::lock_guard<std::mutex> lock(queue_mutex);
	return queued_entries.size();
}

bool PyManager::InvokeHandler::WorkerState::swap(RequestId request_a, RequestId request_b) {
	if(request_a == request_b) return true;

	std::lock_guard<std::mutex> lock(queue_mutex);
	if(queued_request_ids.find(request_a) == queued_request_ids.end() ||
	   queued_request_ids.find(request_b) == queued_request_ids.end()) {
		return false;
	}

	auto it_a = queued_entries.end();
	auto it_b = queued_entries.end();
	for(auto it = queued_entries.begin(); it != queued_entries.end(); ++it) {
		if(it->request_id == request_a) it_a = it;
		if(it->request_id == request_b) it_b = it;
		if(it_a != queued_entries.end() && it_b != queued_entries.end()) break;
	}

	if(it_a == queued_entries.end() || it_b == queued_entries.end()) return false;
	std::iter_swap(it_a, it_b);
	return true;
}

std::vector<PyManager::InvokeHandler::QueueEntry>
PyManager::InvokeHandler::WorkerState::drop_count(size_t count) {
	std::vector<QueueEntry> dropped;
	std::lock_guard<std::mutex> lock(queue_mutex);
	const size_t n = std::min(count, queued_entries.size());
	dropped.reserve(n);
	for(size_t i = 0; i < n; i++) {
		dropped.push_back(std::move(queued_entries.front()));
		queued_request_ids.erase(dropped.back().request_id);
		queued_entries.pop_front();
	}
	return dropped;
}

std::optional<PyManager::InvokeHandler::QueueEntry>
PyManager::InvokeHandler::WorkerState::drop_req(RequestId request_id) {
	std::lock_guard<std::mutex> lock(queue_mutex);
	for(auto it = queued_entries.begin(); it != queued_entries.end(); ++it) {
		if(it->request_id == request_id) {
			QueueEntry dropped = std::move(*it);
			queued_entries.erase(it);
			queued_request_ids.erase(request_id);
			return dropped;
		}
	}
	return std::nullopt;
}

std::vector<PyManager::InvokeHandler::QueueEntry>
PyManager::InvokeHandler::WorkerState::drop_batch(const std::vector<RequestId>& request_ids) {
	std::vector<QueueEntry> dropped;
	dropped.reserve(request_ids.size());
	for(RequestId request_id : request_ids) {
		auto maybe = drop_req(request_id);
		if(maybe.has_value()) dropped.push_back(std::move(*maybe));
	}
	return dropped;
}

bool PyManager::InvokeHandler::WorkerState::empty() const {
	std::lock_guard<std::mutex> lock(queue_mutex);
	return queued_entries.empty();
}

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
	, _resource(std::move(resource))
	, _batch_size(batch_size)
	, _prefetch_depth(prefetch_depth)
	, _state(std::make_shared<WorkerState>())
	, _worker(
		  &InvokeHandler::workerLoop, _state, _resource, _batch_size, _prefetch_depth, &_active) { }

PyManager::InvokeHandler::~InvokeHandler() {
	_active.store(false);
	if(_worker.joinable()) _worker.join();
	if(_state) {
		pybind11::gil_scoped_acquire gil;
		_state.reset();
	}
}

PyManager::InvokeHandler::InvokeHandler(InvokeHandler&& other) noexcept
	: _id(other._id)
	, _manager(std::move(other._manager))
	, _state(std::move(other._state))
	, _worker(std::move(other._worker)) { }

PyManager::InvokeHandler& PyManager::InvokeHandler::operator=(InvokeHandler&& other) noexcept {
	if(this != &other) {
		_active.store(false);
		if(_worker.joinable()) _worker.join();
		if(_state) {
			pybind11::gil_scoped_acquire gil;
			_state.reset();
		}

		_id = other._id;
		_manager = std::move(other._manager);
		_resource = std::move(other._resource);
		_batch_size = other._batch_size;
		_prefetch_depth = other._prefetch_depth;
		_next_request_id.store(other._next_request_id.load());
		_active.store(other._active.load());
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
	RequestId request_id = _next_request_id.fetch_add(1);
	return queue_invoke_with_id(request_id,
								std::forward<CommitFn>(commit_fn),
								std::forward<Callback>(callback),
								std::forward<Args>(args)...);
}

template <typename CommitFn, typename Callback, typename... Args>
auto PyManager::InvokeHandler::queue_invoke_with_id(RequestId request_id,
													CommitFn&& commit_fn,
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

	_state->enqueue(
		QueueEntry{ request_id, std::move(commit), std::move(on_result), std::move(on_error) });

	return future;
}

bool PyManager::InvokeHandler::swap_requests(RequestId request_a, RequestId request_b) {
	return _state->swap(request_a, request_b);
}

std::optional<PyManager::InvokeHandler::QueueEntry>
PyManager::InvokeHandler::take_request(RequestId request_id) {
	return _state->drop_req(request_id);
}

std::vector<PyManager::InvokeHandler::QueueEntry>
PyManager::InvokeHandler::take_requests(const std::vector<RequestId>& request_ids) {
	std::vector<QueueEntry> taken_entries;
	taken_entries.reserve(request_ids.size());

	for(RequestId request_id : request_ids) {
		auto taken = _state->drop_req(request_id);
		if(!taken.has_value()) continue;
		taken_entries.push_back(std::move(*taken));
	}

	return taken_entries;
}

///////////////////////////////////////////////////////////////////////////////
// InvokeHandler Worker Loop
///////////////////////////////////////////////////////////////////////////////

void PyManager::InvokeHandler::workerLoop(std::shared_ptr<WorkerState> state,
										  std::shared_ptr<pybind11::object> resource,
										  size_t batch_size,
										  size_t prefetch_depth,
										  std::atomic<bool>* active) {
	std::deque<CommittedEntry> prefetch_buffer;

	while(active->load() || !state->empty() || !prefetch_buffer.empty()) {

		// Block-wait only when prefetch buffer is empty and queue is empty
		if(prefetch_buffer.empty() && state->empty()) {
			std::this_thread::sleep_for(std::chrono::milliseconds(5));
			continue;
		}

		{ // GIL scope
			pybind11::gil_scoped_acquire gil;

			// Phase 1: Refill prefetch buffer up to batch_size * prefetch_depth
			while(prefetch_buffer.size() < batch_size * prefetch_depth) {
				std::vector<QueueEntry> popped_entries;
				size_t needed = batch_size * prefetch_depth - prefetch_buffer.size();
				size_t popped = state->commit(needed, popped_entries);
				if(popped == 0) break;

				for(auto& entry : popped_entries) {
					try {
						pybind11::object committed = entry.commit();
						prefetch_buffer.push_back(CommittedEntry{ entry.request_id,
																  std::move(committed),
																  std::move(entry.on_result),
																  std::move(entry.on_error) });
					} catch(...) {
						entry.on_error(std::current_exception());
					}
				}
			}

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
			state.py_invoke_handler_map.emplace(module_name, PyInvokeHandlerEntry{ mod, { } });
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

		auto obj_ptr = std::make_shared<pybind11::object>(obj);
		module_it->second.handler_map[entry_point] = obj_ptr;
		size_t id = module_it->second.handler_map.size() - 1;

		return PyManager::InvokeHandler(
			id, obj_ptr, std::make_unique<PyManager>(), batch_size, prefetch_depth);
	}

	size_t id = 0;
	return PyManager::InvokeHandler(
		id, object_it->second, std::make_unique<PyManager>(), batch_size, prefetch_depth);
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

	{
		// Release GIL so InvokeHandler worker threads can acquire it
		pybind11::gil_scoped_release gil;

		shared().interpreter_initialized.store(true);

		std::unique_lock<std::mutex> lock(shared().destructor_mutex);
		shared().destructor_cv.wait(lock, [] { return !shared().threads_active; });
	} // GIL reacquired

	shared().py_invoke_handler_map.clear();
	shared().interpreter_initialized.store(false);

	atexit = { };
	sys = { };

	// In embedded multi-threaded use, Py_FinalizeEx can crash during GC traversal
	// of extension-managed objects. Default to process-exit cleanup, and allow
	// explicit opt-in finalization for controlled environments.
	const char* finalize_env = std::getenv("PYSCHEDULER_FINALIZE_INTERPRETER");
	if(finalize_env != nullptr && std::string_view(finalize_env) == "1") {
		pybind11::finalize_interpreter();
	}
}
} // namespace pyscheduler
