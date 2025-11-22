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
PyManager::InvokeHandler::InvokeHandler(size_t id, 
			std::shared_ptr<pybind11::object> resource,
			std::unique_ptr<PyManager> manager
		)
	: _id(id)
	, _manager(std::move(manager))
	, _resource(resource)
	{ }

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
	return callback(result);
}
template <typename Callback, typename... Args>
auto PyManager::InvokeHandler::queue_invoke(Callback&& callback, Args&&... args)
    -> std::future<std::invoke_result_t<Callback, pybind11::object>>
{
    using ReturnType = std::invoke_result_t<Callback, pybind11::object>;
    using PromisePtr = std::shared_ptr<std::promise<ReturnType>>;

    // Safety: we don't want to ever store pybind11::object in the future,
    // because its destruction might happen without the GIL.
    static_assert(!std::is_same_v<ReturnType, pybind11::object>,
                  "ReturnType must not be pybind11::object; convert to a pure C++ type in the callback.");

    // Capture the call arguments into a tuple. If any of Args are pybind11::object,
    // you should only call this function while holding the GIL.
    auto args_tuple = std::make_tuple(std::forward<Args>(args)...);

    auto promise_ptr = std::make_shared<std::promise<ReturnType>>();
    auto future      = promise_ptr->get_future();

    // Everything captured by value; the tuple is moved in, so the only live
    // pybind11::object instances will be inside this lambda.
    auto method =
        [this,
         cb         = std::forward<Callback>(callback),
         args       = std::move(args_tuple),
         promise_ptr]() mutable
        {
            pybind11::gil_scoped_acquire gil;

            pybind11::object py_result = std::apply(
                [this](auto&&... unpackedArgs) -> pybind11::object {
                    // Forward the tuple elements into the Python callable
                    return (*this->_resource.get())(
                        std::forward<decltype(unpackedArgs)>(unpackedArgs)...);
                },
                std::move(args));

            try {
                ReturnType value = std::invoke(cb, py_result);
                promise_ptr->set_value(std::move(value));
            } catch (...) {
                promise_ptr->set_exception(std::current_exception());
            }

			args = {};
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
	pybind11::gil_scoped_acquire gil;

	auto module_it = state.py_invoke_handler_map.find(module_name);
	if(module_it == state.py_invoke_handler_map.end()) {
		// - Load python module if it has not been loaded before.
		// - The idea of passing a PyManager unique pointer is so that
		// the cleanup code for PyManager e.g. interpreter tear down
		// does not happen until all instances of InvokeHandlers that were
		// handed out go out of scope.


		pybind11::module_ mod;
		try {
			mod = pybind11::module_::import(module_name.c_str());
		} catch(pybind11::error_already_set& e) {
			throw std::invalid_argument("Could not import module: " + module_name);
		}

		auto [inserted_it, successful] = state.py_invoke_handler_map.emplace(module_name, PyInvokeHandlerEntry{mod, {}});
		module_it = inserted_it;

		if (!successful) {
			throw std::runtime_error("Could not insert module: " + module_name);
		}
	}


	auto object_it = module_it->second.handler_map.find(entry_point);
	if (object_it == module_it->second.handler_map.end()) {
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

		return PyManager::InvokeHandler(id, obj_ptr, std::make_unique<PyManager>());
	}

	size_t id = object_it->second;
	return PyManager::InvokeHandler(id, state.py_objects[id], std::make_unique<PyManager>());
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
		shared().py_objects.clear();

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
