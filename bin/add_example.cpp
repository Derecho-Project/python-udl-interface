#include "lib/python_manager.hpp"
#include <vector>
// #include "lib/utils.hpp"

using namespace std::chrono;
namespace pys = pyscheduler;

// this is a simple example of how to synchronously execute a python method using
// PyManager

// Note that it doesn't matter if the current thread holds the GIL when executing
// "invoke" because invoke tries to obtain the GIL before making the python call.

int main() {
	pys::PyManager manager;
	pys::PyManager::InvokeHandler add = manager.getPythonModule("python_models.add", "invoke");

	std::cout << add.invoke<int>(12, 13) << std::endl;
	return 0;
}