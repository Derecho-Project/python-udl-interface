#include "lib/python_manager.hpp"
#include <chrono>

namespace pyscheduler {

///////////////////////////////////////////////////////////////////////////////
// Impl Invoke Handler
///////////////////////////////////////////////////////////////////////////////
PyManager::InvokeHandler::InvokeHandler(size_t id, std::unique_ptr<PyManager> manager)
	: _id(id)
	, _manager(std::move(manager)) { }

const std::shared_ptr<std::pair<pybind11::module_, pybind11::object>>
PyManager::InvokeHandler::getModuleAndFunc() {
	// need to lock py_mutex because we don't want a vector resize
	// to happen during lookup

	// should allow multiple reads concurrently which don't mutate state
	PyManager::SharedState& state = _manager->shared();
	std::shared_lock lock(state.py_mutex);
	return _manager->shared().py_modules.at(_id);
}

///////////////////////////////////////////////////////////////////////////////
// Impl PyManager
///////////////////////////////////////////////////////////////////////////////

PyManager::PyManager() {
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

PyManager::InvokeHandler PyManager::getPythonModule(const std::string& module_name,
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

		pybind11::module_ mod = pybind11::module_::import(module_name.c_str());

		if(!mod) {
			PyErr_Print();
			throw std::invalid_argument("Could not import module: " + module_name);
		}

		pybind11::object func = mod.attr(entry_point.c_str());

		if(!func) {
			PyErr_Print();
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
					while(shared().threads_active) {
						std::function<void()> task;

						// have a small timeout so threads can wake up and check if they
						// need to exit.

						// should not cause program to crash, but should not wake up frequently
						// to take up precious CPU cycles.
						bool success =
							shared().task_queue.wait_dequeue_timed(task, milliseconds(100));

						if(!success) {
							continue;
						}

						// std::cout << shared().task_queue.size_approx() << std::endl;
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
			std::function<void()> black_box;
			shared().task_queue.try_dequeue(black_box);
		}

	} // end python interpreter
}
} // namespace pyscheduler
