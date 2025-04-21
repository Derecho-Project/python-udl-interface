cmake_minimum_required(VERSION 3.14)
project(pyscheduler VERSION 0.1.0 LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)

add_subdirectory(lib/dlpack)
add_subdirectory(lib/pybind11)
add_subdirectory(lib/concurrentqueue)

add_library(${PROJECT_NAME} SHARED "include/pyscheduler/pyscheduler.cpp")
target_link_libraries(pyscheduler PUBLIC
    dlpack
    concurrentqueue
    pybind11::embed
)
target_include_directories(${PROJECT_NAME} PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/include)
target_include_directories(${PROJECT_NAME} PUBLIC include)

# Build example programs
option(BUILD_EXAMPLES "Build example programs" OFF)
if(BUILD_EXAMPLES)
    add_subdirectory(examples)
endif()
