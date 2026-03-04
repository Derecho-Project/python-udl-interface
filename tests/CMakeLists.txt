file(GLOB TEST_SOURCES CONFIGURE_DEPENDS test_*.cpp)
file(GLOB BENCH_SOURCES CONFIGURE_DEPENDS bench_*.cpp)

if(TEST_SOURCES)
	add_executable(pyscheduler_tests ${TEST_SOURCES})
	target_link_libraries(pyscheduler_tests PRIVATE
		pyscheduler::pyscheduler
		Catch2::Catch2WithMain
	)

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
	target_link_libraries(pyscheduler_bench PRIVATE
		pyscheduler::pyscheduler
		benchmark::benchmark
	)
endif()
