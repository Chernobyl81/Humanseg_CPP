cmake_minimum_required(VERSION 3.22.1)
project(PP_HumanSeg_CPP VERSION 0.1.0)

set(CMAKE_BUILD_TYPE Debug)
set(CMAKE_CXX_STANDARD 17)
set(VERSION_STRING 0.1.0)
set(SOVERSION_STRING 0.1.0)

include(CTest)
enable_testing()

add_executable(PP_HumanSeg_CPP main.cpp)

set(CPACK_PROJECT_NAME ${PROJECT_NAME})
set(CPACK_PROJECT_VERSION ${PROJECT_VERSION})
include(CPack)
