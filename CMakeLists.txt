cmake_minimum_required(VERSION 3.17)
project(mllib_cpp)

find_package(Armadillo REQUIRED)
include_directories(${ARMADILLO_INCLUDE_DIRS})

set(CMAKE_CXX_STANDARD 14)
add_subdirectory(src)