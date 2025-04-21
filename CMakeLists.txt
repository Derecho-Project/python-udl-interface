cmake_minimum_required(VERSION 3.14)
project(pyscheduler VERSION 0.1.0 LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)

add_subdirectory(lib/dlpack)
add_subdirectory(lib/pybind11)
add_subdirectory(lib/concurrentqueue)

add_library(${PROJECT_NAME} SHARED "include/pyscheduler/python_manager.cpp")
target_link_libraries(pyscheduler PRIVATE
    dlpack
    concurrentqueue
    pybind11::embed
)
target_include_directories(${PROJECT_NAME} PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/include)
target_include_directories(${PROJECT_NAME} PUBLIC include)

# find_package(pybind11 REQUIRED)
# find_package(Python3 REQUIRED Interpreter Development)
# find_package(CUDAToolkit REQUIRED)

# # Python Manager static Library
# add_library(python_manager STATIC
#     lib/python_manager.cpp
# )
# target_include_directories(python_manager PUBLIC
#     ${CMAKE_SOURCE_DIR}
#     ${Python3_INCLUDE_DIRS}
#     ${pybind11_INCLUDE_DIRS}
# )
# target_link_libraries(python_manager PRIVATE
#     ${Python3_LIBRARIES}
#     pybind11::embed
# )

# # Add Example 
# add_executable(add_example
#     bin/add_example.cpp
# )
# target_include_directories(add_example PRIVATE
#     ${Python3_INCLUDE_DIRS}
#     ${CMAKE_SOURCE_DIR}
# )
# target_link_libraries(
#     add_example
#     PRIVATE
#     ${Python3_LIBRARIES}
#     pybind11::embed
#     python_manager
# )

# # Cupy Example 
# add_executable(dlpack_recv_example
#     bin/dlpack_recv_example.cpp
# )
# # add_compile_options(dlpack_recv_example -Wno-attributes)
# target_include_directories(dlpack_recv_example PRIVATE
#     ${Python3_INCLUDE_DIRS}
#     ${CMAKE_SOURCE_DIR}
#     ${CUDA_INCLUDE}
#     ${CMAKE_CURRENT_SOURCE_DIR}/dlpack/include
# )
# target_link_libraries(
#     dlpack_recv_example
#     PRIVATE
#     ${Python3_LIBRARIES}
#     pybind11::embed
#     python_manager
#     CUDA::cudart
# )

# # Stress test benchmark Example 
# add_executable(encoder_stress_example
#     bin/encoder_stress_example.cpp
# )
# # add_compile_options(encoder_stress_example PRIVATE -Wno-attributes)
# target_include_directories(encoder_stress_example PRIVATE
#     ${Python3_INCLUDE_DIRS}
#     ${CMAKE_SOURCE_DIR}
#     # ${CUDA_INCLUDE}
#     # ${CMAKE_CURRENT_SOURCE_DIR}/dlpack/include
# )
# target_link_libraries(
#     encoder_stress_example
#     PRIVATE
#     ${Python3_LIBRARIES}
#     pybind11::embed
#     python_manager
#     # CUDA::cudart
# )

# # Stress test benchmark Example 
# add_executable(benchmark
#     bin/benchmark.cpp
# )
# # add_compile_options(encoder_stress_example PRIVATE -Wno-attributes)
# target_include_directories(benchmark PRIVATE
#     ${Python3_INCLUDE_DIRS}
#     ${CMAKE_SOURCE_DIR}
#     # ${CUDA_INCLUDE}
#     # ${CMAKE_CURRENT_SOURCE_DIR}/dlpack/include
# )
# target_link_libraries(
#     benchmark
#     PRIVATE
#     ${Python3_LIBRARIES}
#     pybind11::embed
#     python_manager
#     nvidia-ml
#     # CUDA::cudart
# )
