cmake_minimum_required(VERSION 3.27)
cmake_policy(SET CMP0148 NEW)

project(pyscheduler VERSION 0.1.0 LANGUAGES CXX)


set(CMAKE_CXX_STANDARD 20)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

include(GNUInstallDirs)
include(CMakePackageConfigHelpers)

set(_pyscheduler_install_cmakedir "${CMAKE_INSTALL_LIBDIR}/cmake/${PROJECT_NAME}")

option(BUILD_EXAMPLES "Build example programs" OFF)
option(BUILD_TESTS "Build test cases" OFF)
option(ENABLE_FP "Enable frame pointer for flamegraph generation" OFF)
option(ENABLE_ASAN "Enable AddressSanitizer" OFF)
option(ENABLE_TSAN "Enable ThreadSanitizer" OFF)

if(ENABLE_ASAN AND ENABLE_TSAN)
    message(FATAL_ERROR "ASan and TSan cannot be enabled simultaneously.")
endif()

add_subdirectory(extern/concurrentqueue)

find_package(pybind11 CONFIG REQUIRED)
find_package(Threads REQUIRED)

add_library(${PROJECT_NAME} INTERFACE)
add_library(${PROJECT_NAME}::${PROJECT_NAME} ALIAS ${PROJECT_NAME})

target_compile_features(${PROJECT_NAME} INTERFACE cxx_std_17)
target_include_directories(${PROJECT_NAME} INTERFACE
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/extern/dlpack/include>
    $<INSTALL_INTERFACE:include>
)
target_link_libraries(${PROJECT_NAME} INTERFACE
    pybind11::embed
    concurrentqueue
    Threads::Threads
)

if(CMAKE_CXX_COMPILER_ID MATCHES "GNU|Clang")
    if(ENABLE_ASAN)
        message(STATUS "Using AddressSanitizer")
    endif()
    if(ENABLE_TSAN)
        message(STATUS "Using ThreadSanitizer")
    endif()

    target_compile_options(${PROJECT_NAME} INTERFACE
        $<$<BOOL:${ENABLE_FP}>:-fno-omit-frame-pointer>
        $<$<AND:$<BOOL:${ENABLE_FP}>,$<CONFIG:Debug,RelWithDebInfo>>:-g>
        $<$<BOOL:${ENABLE_ASAN}>:-fsanitize=address;-fno-omit-frame-pointer>
        $<$<BOOL:${ENABLE_TSAN}>:-fsanitize=thread;-fno-omit-frame-pointer>
    )
    target_link_options(${PROJECT_NAME} INTERFACE
        $<$<BOOL:${ENABLE_ASAN}>:-fsanitize=address>
        $<$<BOOL:${ENABLE_TSAN}>:-fsanitize=thread>
    )
endif()

if(BUILD_EXAMPLES)
    add_subdirectory(examples)
endif()

if(BUILD_TESTS)
    add_subdirectory(extern/Catch2)
    set(BENCHMARK_ENABLE_TESTING OFF CACHE BOOL "Disable benchmark internal tests" FORCE)
    set(BENCHMARK_ENABLE_GTEST_TESTS OFF CACHE BOOL "Disable benchmark gtest-based tests" FORCE)
    add_subdirectory(extern/benchmark)
    enable_testing()
    add_subdirectory(tests)
endif()

install(
    TARGETS ${PROJECT_NAME}
    EXPORT ${PROJECT_NAME}Targets
    INCLUDES DESTINATION ${CMAKE_INSTALL_INCLUDEDIR}
)

install(
    DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}/include/
    DESTINATION ${CMAKE_INSTALL_INCLUDEDIR}
    FILES_MATCHING PATTERN "*.hpp" PATTERN "*.h"
)

install(
    EXPORT ${PROJECT_NAME}Targets
    FILE ${PROJECT_NAME}Targets.cmake
    NAMESPACE ${PROJECT_NAME}::
    DESTINATION ${_pyscheduler_install_cmakedir}
)

write_basic_package_version_file(
    "${CMAKE_CURRENT_BINARY_DIR}/${PROJECT_NAME}ConfigVersion.cmake"
    VERSION ${PROJECT_VERSION}
    COMPATIBILITY AnyNewerVersion
    ARCH_INDEPENDENT
)

configure_package_config_file(
    "${CMAKE_CURRENT_LIST_DIR}/cmake/Config.cmake.in"
    "${CMAKE_CURRENT_BINARY_DIR}/${PROJECT_NAME}Config.cmake"
    INSTALL_DESTINATION ${_pyscheduler_install_cmakedir}
)

install(
    FILES
        "${CMAKE_CURRENT_BINARY_DIR}/${PROJECT_NAME}Config.cmake"
        "${CMAKE_CURRENT_BINARY_DIR}/${PROJECT_NAME}ConfigVersion.cmake"
    DESTINATION ${_pyscheduler_install_cmakedir}
)
