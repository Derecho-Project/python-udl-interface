#include <pyscheduler/pyscheduler.hpp>

using namespace std;

int main() {
	pyscheduler::PyManager manager;
	pyscheduler::PyManager::InvokeHandler add =
		manager.getPythonModule("examples.add.python_modules.add", "invoke");
	std::cout << add.invoke<int64_t>(3000, -1234) << std::endl;
	return 0;
}