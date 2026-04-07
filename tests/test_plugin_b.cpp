#include <pyscheduler/pyscheduler.hpp>

#include <memory>

using namespace pyscheduler;

namespace {
std::unique_ptr<PyManager> g_manager;
}

extern "C" void plugin_start() {
	if(!g_manager) {
		g_manager = std::make_unique<PyManager>();
	}
}

extern "C" void plugin_stop() {
	g_manager.reset();
}

extern "C" uint64_t plugin_arc_count() {
	return PyManager::debug_arc_count();
}

extern "C" uintptr_t plugin_shared_state_address() {
	return PyManager::debug_shared_state_address();
}
