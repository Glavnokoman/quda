cmake_minimum_required(VERSION 3.8)
project(Quda)

option(${PROJECT_NAME}_build_tests "build the tests" ON)
option(${PROJECT_NAME}_stub_cuda "debugging option to simulate running cuda kernels on cpu" OFF)

set(CMAKE_CXX_STANDARD 11) # nvcc 9.0 does not support c++14
set(CMAKE_CXX_STANDARD_REQUIRED true)
list(APPEND CMAKE_CXX_FLAGS "-std=c++11 -Wall")

set(CMAKE_MODULE_PATH ${CMAKE_MODULE_PATH} "${CMAKE_SOURCE_DIR}/cmake" )

add_subdirectory(src)
if(${PROJECT_NAME}_build_tests)
	include(CTest)
	add_subdirectory(test)
endif()
