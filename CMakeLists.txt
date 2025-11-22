cmake_minimum_required(VERSION 3.27)
cmake_policy(SET CMP0148 NEW)
project(pyscheduler VERSION 0.1.0 LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

# fetch concurrent queue
include(FetchContent)
FetchContent_Declare(
    concurrentqueue
    GIT_REPOSITORY https://github.com/cameron314/concurrentqueue.git
    GIT_TAG master
)
FetchContent_MakeAvailable(concurrentqueue)

find_package(pybind11 CONFIG REQUIRED)
find_package(Threads REQUIRED)

# create library target
add_library(${PROJECT_NAME} INTERFACE)
target_link_libraries(${PROJECT_NAME} INTERFACE
    pybind11::embed
    concurrentqueue
    Threads::Threads
)
target_include_directories(${PROJECT_NAME} INTERFACE
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
    $<INSTALL_INTERFACE:include>
)

# Diagnostic flags
option(ENABLE_GPROF "Enable gprof profiling flags (-pg)" OFF)
option(ENABLE_FP "Enable frame pointer for flamegraph generation" OFF)
option(ENABLE_ASAN  "Enable AddressSanitizer" OFF)
option(ENABLE_TSAN  "Enable ThreadSanitizer" OFF)
if (ENABLE_ASAN AND ENABLE_TSAN)
    message(FATAL_ERROR "ASan and TSan cannot be enabled simultaneously.")
endif()

if (CMAKE_CXX_COMPILER_ID MATCHES "GNU|Clang")
    if (ENABLE_GPROF)
        message(STATUS "Enabling gprof (-pg)")
        add_compile_options(-pg)
        add_link_options(-pg)
    endif()

    if (ENABLE_FP)
        message(STATUS "Enabling frame pointer (-fno-omit-frame-pointer)")
        add_compile_options(-fno-omit-frame-pointer -g)
        if (CMAKE_BUILD_TYPE MATCHES "Debug|RelWithDebInfo")
            add_compile_options(-g)
        endif()
    endif()

    if (ENABLE_ASAN)
        message(STATUS "Using AddressSanitizer")
        add_compile_options(-fsanitize=address -fno-omit-frame-pointer)
        add_link_options(-fsanitize=address)
    endif()

    if (ENABLE_TSAN)
        message(STATUS "Using ThreadSanitizer")
        add_compile_options(-fsanitize=thread -fno-omit-frame-pointer)
        add_link_options(-fsanitize=thread)
    endif()
endif()

# Build example programs
option(BUILD_EXAMPLES "Build example programs" OFF)
if(BUILD_EXAMPLES)
    add_subdirectory(examples)
endif()

# Build test cases
option(BUILD_TESTS "Build test cases" OFF)
if(BUILD_TESTS)
    add_subdirectory(extern/Catch2)
    enable_testing()
    add_subdirectory(tests)
endif()

# Install
include(GNUInstallDirs)
include(CMakePackageConfigHelpers)

install(
    TARGETS ${PROJECT_NAME}
    EXPORT ${PROJECT_NAME}Targets
    LIBRARY DESTINATION ${CMAKE_INSTALL_LIBDIR} # .so / .dylib
    ARCHIVE DESTINATION ${CMAKE_INSTALL_LIBDIR} # .a
    RUNTIME DESTINATION ${CMAKE_INSTALL_BINDIR} # .exe on Windows
    INCLUDES DESTINATION ${CMAKE_INSTALL_INCLUDEDIR} # for modern CMake

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
    DESTINATION ${CMAKE_INSTALL_LIBDIR}/cmake/${PROJECT_NAME}
)

write_basic_package_version_file(
    "${CMAKE_CURRENT_BINARY_DIR}/${PROJECT_NAME}ConfigVersion.cmake"
    VERSION ${PROJECT_VERSION}
    COMPATIBILITY AnyNewerVersion
)

configure_package_config_file(
    "${CMAKE_CURRENT_LIST_DIR}/cmake/Config.cmake.in"
    "${CMAKE_CURRENT_BINARY_DIR}/${PROJECT_NAME}Config.cmake"
    INSTALL_DESTINATION ${CMAKE_INSTALL_LIBDIR}/cmake/${PROJECT_NAME}
)

install(
    FILES
    "${CMAKE_CURRENT_BINARY_DIR}/${PROJECT_NAME}Config.cmake"
    "${CMAKE_CURRENT_BINARY_DIR}/${PROJECT_NAME}ConfigVersion.cmake"
    DESTINATION ${CMAKE_INSTALL_LIBDIR}/cmake/${PROJECT_NAME}
)
