file(GLOB TEST_SOURCES CONFIGURE_DEPENDS *.cpp)

get_target_property(_inc Catch2::Catch2WithMain INTERFACE_INCLUDE_DIRECTORIES)
message(STATUS "Catch2::Catch2WithMain include dirs: ${_inc}")


add_executable(pyscheduler_tests ${TEST_SOURCES})
target_link_libraries(pyscheduler_tests PRIVATE pyscheduler Catch2::Catch2WithMain)

include(CTest)
include(Catch)
catch_discover_tests(pyscheduler_tests)
