#include <pyscheduler/pyscheduler.hpp>

using namespace std;

int main() {
	pyscheduler::PyManager manager;
	pyscheduler::PyManager::InvokeHandler add =
		manager.loadPythonModule("examples.add.python_modules.add", "invoke");
	for (int i = 0; i < 200; i++) {
		std::cout << add.invoke<int64_t>(3000, -1234) << "\n";
	}
	std::cout.flush();
	return 0;
}