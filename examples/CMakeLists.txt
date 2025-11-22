add_executable(add add/main.cpp)
target_link_libraries(add PRIVATE pyscheduler)

add_executable(tensor_share tensor_share/main.cpp)
target_link_libraries(tensor_share PRIVATE pyscheduler)