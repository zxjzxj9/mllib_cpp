cmake_minimum_required(VERSION 3.17)
project(mllib_cpp)

set(CMAKE_CXX_STANDARD 14)

add_subdirectory(src)
find_package(armadillo)

add_executable(mllib_cpp main.cpp)