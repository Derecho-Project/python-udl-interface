set(_example_targets
	add
)

foreach(_target IN LISTS _example_targets)
	add_executable(${_target} ${_target}/main.cpp)
	target_link_libraries(${_target} PRIVATE pyscheduler::pyscheduler)
endforeach()