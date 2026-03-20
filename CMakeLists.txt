cmake_minimum_required(VERSION 3.20)
project(fluid LANGUAGES CXX)
set(CMAKE_CXX_STANDARD 17)
find_package(glad CONFIG REQUIRED)
find_package(glfw3 CONFIG REQUIRED)

add_executable(fluid main.cpp)
target_link_libraries(fluid PRIVATE glad::glad glfw)