file(GLOB TEST_SOURCES CONFIGURE_DEPENDS test_*.cpp)
file(GLOB BENCH_SOURCES CONFIGURE_DEPENDS bench_*.cpp)

list(REMOVE_ITEM TEST_SOURCES
	"${CMAKE_CURRENT_SOURCE_DIR}/test_plugin_a.cpp"
	"${CMAKE_CURRENT_SOURCE_DIR}/test_plugin_b.cpp"
)

add_library(pyscheduler_test_plugin_a SHARED test_plugin_a.cpp)
target_link_libraries(pyscheduler_test_plugin_a PRIVATE pyscheduler::pyscheduler)

add_library(pyscheduler_test_plugin_b SHARED test_plugin_b.cpp)
target_link_libraries(pyscheduler_test_plugin_b PRIVATE pyscheduler::pyscheduler)

if(TEST_SOURCES)
	add_executable(pyscheduler_tests ${TEST_SOURCES})
	target_compile_definitions(pyscheduler_tests PRIVATE
		PYSCHEDULER_SOURCE_DIR="${CMAKE_SOURCE_DIR}"
		PYSCHEDULER_TEST_PLUGIN_A_PATH="$<TARGET_FILE:pyscheduler_test_plugin_a>"
		PYSCHEDULER_TEST_PLUGIN_B_PATH="$<TARGET_FILE:pyscheduler_test_plugin_b>"
	)
	find_package(BLAS REQUIRED)
	target_link_libraries(pyscheduler_tests PRIVATE
		pyscheduler::pyscheduler
		Catch2::Catch2WithMain
		BLAS::BLAS
		dl
	)

	find_package(CUDAToolkit REQUIRED)
	target_link_libraries(pyscheduler_tests PRIVATE CUDA::cudart)
	target_compile_definitions(pyscheduler_tests PRIVATE PYSCHEDULER_TEST_HAS_CUDA=1)

	# Symlink LSAN suppressions file into test binary directory
	set(_lsan_src "${CMAKE_SOURCE_DIR}/lsan_supressions.txt")
	set(_lsan_dst "$<TARGET_FILE_DIR:pyscheduler_tests>/lsan_suppressions.txt")
	if(EXISTS "${_lsan_src}")
		add_custom_command(TARGET pyscheduler_tests POST_BUILD
			COMMAND ${CMAKE_COMMAND} -E create_symlink "${_lsan_src}" "${_lsan_dst}"
			COMMENT "Symlinking lsan_suppressions.txt into test directory"
		)
	endif()

	include(CTest)
	include(Catch)
	catch_discover_tests(pyscheduler_tests)
endif()

if(BENCH_SOURCES)
	add_executable(pyscheduler_bench ${BENCH_SOURCES})
	target_compile_definitions(pyscheduler_bench PRIVATE
		PYSCHEDULER_SOURCE_DIR="${CMAKE_SOURCE_DIR}"
	)
	target_link_libraries(pyscheduler_bench PRIVATE
		pyscheduler::pyscheduler
		benchmark::benchmark_main
	)

	find_package(CUDAToolkit REQUIRED)
	target_link_libraries(pyscheduler_bench PRIVATE CUDA::cudart)
	target_compile_definitions(pyscheduler_bench PRIVATE PYSCHEDULER_TEST_HAS_CUDA=1)
endif()
