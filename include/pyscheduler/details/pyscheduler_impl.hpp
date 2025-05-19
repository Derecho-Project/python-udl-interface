#ifdef __INTELLISENSE__
#	include "pyscheduler/pyscheduler.hpp"
#endif

#include "pyscheduler/move_only.hpp"
#include <cassert>
#include <chrono>

namespace pyscheduler {

///////////////////////////////////////////////////////////////////////////////
// Impl Invoke Handler
///////////////////////////////////////////////////////////////////////////////
PyManager::InvokeHandler::InvokeHandler(size_t id, std::unique_ptr<PyManager> manager)
	: _id(id)
	, _manager(std::move(manager)) { }

const std::shared_ptr<std::pair<pybind11::module_, pybind11::object>>&
PyManager::InvokeHandler::getModuleAndFunc() {
	// need to lock py_mutex because we don't want a vector resize
	// to happen during lookup

	// no need to acquire gil because not incrementing python reference count

	// should allow multiple reads concurrently which don't mutate state
	PyManager::SharedState& state = _manager->shared();
	std::shared_lock lock(state.py_mutex);
	return _manager->shared().py_modules.at(_id);
}

template <typename ReturnType, typename... Args>
ReturnType PyManager::InvokeHandler::invoke(Args&&... args) {
	auto mod_and_func = getModuleAndFunc();
	pybind11::gil_scoped_acquire gil;
	pybind11::object result = mod_and_func->second(std::forward<Args>(args)...);
	return result.cast<ReturnType>();
}

template <typename Callback, typename... Args>
auto PyManager::InvokeHandler::invoke(Callback&& callback, Args&&... args)
	-> std::invoke_result_t<Callback, pybind11::object> {
	auto mod_and_func = getModuleAndFunc();
	pybind11::gil_scoped_acquire gil;
	pybind11::object result = mod_and_func->second(std::forward<Args>(args)...);
	return callback(result);
}
template <typename Callback, typename... Args>
auto PyManager::InvokeHandler::queue_invoke(Callback&& callback, Args&&... args)
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

	auto mod_and_func = getModuleAndFunc();
	auto method = [this,
				   callback = std::move(callback),
				   mod_and_func = std::move(mod_and_func),
				   args_tuple = std::move(args_tuple),
				   promise_ptr]() mutable {
		pybind11::gil_scoped_acquire gil;
		pybind11::object result = std::apply(
			[&mod_and_func](auto&&... unpackedArgs) {
				return mod_and_func->second(std::forward<decltype(unpackedArgs)>(unpackedArgs)...);
			},
			args_tuple);

		promise_ptr->set_value(callback(result));
	};
	PyManager::shared().task_queue.enqueue(std::move(method));
	return future;
}

///////////////////////////////////////////////////////////////////////////////
// Impl PyManager
///////////////////////////////////////////////////////////////////////////////

PyManager::SharedState PyManager::_instance;

PyManager::PyManager() {
	// this lock should be dropped at return so that postcondition (Python Interpreter Initialized)
	// is guaranteed
	std::unique_lock lock(shared().py_mutex);
	if(shared().arc.fetch_add(1) == 0) {
		shared().main_worker = std::thread(&PyManager::mainLoop, this);
	}
	// small cost paid to block until interpreter is initalized
	while(!shared().interpreter_initialized)
		continue;
}

PyManager::~PyManager() {
	shared().arc--;
	if(shared().arc == 0) {
		shared().threads_active = false;

		// main worker handles interpreter cleanup
		// 1. https://docs.python.org/3/c-api/init.html#c.Py_FinalizeEx
		//		Py_FinalizeEx should be called in the same thread as Py_InitializeEx
		//
		shared().main_worker.join();
	}
}

PyManager::InvokeHandler PyManager::loadPythonModule(const std::string& module_name,
													 const std::string& entry_point) {

	SharedState& state = shared();

	// require write lock
	std::unique_lock lock(state.py_mutex);

	auto it = state.py_invoke_handler_map.find(module_name);
	if(it == state.py_invoke_handler_map.end()) {
		// - Load python module if it has not been loaded before.
		// - The idea of passing a PyManager unique pointer is so that
		// the cleanup code for PyManager e.g. interpreter tear down
		// does not happen until all instances of InvokeHandlers that were
		// handed out go out of scope.

		pybind11::gil_scoped_acquire gil;

		pybind11::module_ mod;
		try {
			mod = pybind11::module_::import(module_name.c_str());
		} catch(pybind11::error_already_set& e) {
			throw std::invalid_argument("Could not import module: " + module_name);
		}

		pybind11::object func;
		try {
			func = mod.attr(entry_point.c_str());
		} catch(pybind11::error_already_set& e) {
			throw std::invalid_argument("Could not find the '" + entry_point +
										"' method in module " + module_name);
		}

		size_t id = state.py_modules.size();

		state.py_modules.emplace_back(
			std::make_shared<std::pair<pybind11::module_, pybind11::object>>(
				std::make_pair(mod, func)));
		state.py_invoke_handler_map[module_name] = id;
		lock.unlock();
		return PyManager::InvokeHandler(id, std::make_unique<PyManager>());
	}

	lock.unlock();
	return PyManager::InvokeHandler(it->second, std::make_unique<PyManager>());
}

void PyManager::mainLoop() {
	// ok to spin here; could be a case where PyManager session
	// is being cleaned up WHILE a new PyManager session is being
	// created (small cost that is almost never paid)
	if(shared().interpreter_initialized) {
		throw std::runtime_error(
			"Cannot reinitialize Python interpreter once it has been shut down.");
	}

	{
		// Do not register python signal handlers
		// https://docs.python.org/3/c-api/init.html#c.Py_InitializeEx

		// passing false as an argument gives a RAII version of
		// Py_InitializeEx(0);

		// the python interpreter must be destroyed by the same thread that created it
		pybind11::scoped_interpreter interpreter(false);

		shared().interpreter_initialized.store(true);

		pybind11::module_ sys = pybind11::module_::import("sys");
		pybind11::list sys_path = sys.attr("path");
		sys_path.append(".");

		{
			// do not acquire GIL in this chunk because workers need to acquire
			// GIL to finish their workload
			pybind11::gil_scoped_release gil;

			std::vector<std::thread> sub_workers;
			for(size_t i = 0; i < NUM_WORKERS; i++) {
				sub_workers.emplace_back(std::thread([i]() {
					// worker should only end if stop signal is set and queue is empty
					while(shared().threads_active || shared().task_queue.size_approx() > 0) {
						MoveOnlyFunction<void()> task;

						// have a small timeout so threads can wake up and check if they
						// need to exit.

						// should not cause program to crash, but should not wake up frequently
						// to take up precious CPU cycles.
						bool success =
							shared().task_queue.wait_dequeue_timed(task, milliseconds(100));

						if(!success) {
							continue;
						}

						task();
					}
				}));
			}

			for(auto& worker : sub_workers) {
				worker.join();
			}
		} // end gil_scoped_release

		// need to reacquire GIL since we're destroying the interpreter
		// here, we do not need to match an ensure with a release
		// because Py_Finalize terminates execution

		shared().py_invoke_handler_map.clear();
		shared().py_modules.clear();

		// closures still might hold references to python objects
		// we clear and drop all items in queue to safely free
		// memory.
		while(shared().task_queue.size_approx() > 0) {
			MoveOnlyFunction<void()> black_box;
			shared().task_queue.try_dequeue(black_box);
		}
	} // end python interpreter
}
} // namespace pyscheduler
