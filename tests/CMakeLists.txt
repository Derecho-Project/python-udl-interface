file(GLOB TEST_SOURCES CONFIGURE_DEPENDS test_*.cpp)
file(GLOB BENCH_SOURCES CONFIGURE_DEPENDS bench_*.cpp)

if(TEST_SOURCES)
	add_executable(pyscheduler_tests ${TEST_SOURCES})
	target_link_libraries(pyscheduler_tests PRIVATE
		pyscheduler::pyscheduler
		Catch2::Catch2WithMain
	)

	include(CTest)
	include(Catch)
	catch_discover_tests(pyscheduler_tests)
endif()

if(BENCH_SOURCES)
	add_executable(pyscheduler_bench ${BENCH_SOURCES})
	target_link_libraries(pyscheduler_bench PRIVATE
		pyscheduler::pyscheduler
		benchmark::benchmark
	)
endif()
