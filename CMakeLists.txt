cmake_minimum_required(VERSION 3.22.1)
project(PP_HumanSeg_CPP VERSION 0.1.0)

set(CMAKE_BUILD_TYPE Release)
set(CMAKE_CXX_FLAGS "-O3")
set(CMAKE_CXX_STANDARD 17)
set(VERSION_STRING 0.1.0)
set(SOVERSION_STRING 0.1.0)

message(STATUS "BUILD FLAGS: ${CMAKE_CXX_FLAGS}")

include(CTest)
enable_testing()

find_package(OpenCV REQUIRED)
#set(TOOLKIT_ROOT_DIR ${CMAKE_SOURCE_DIR})
include_directories(${CMAKE_CURRENT_SOURCE_DIR}/onnxruntime)

add_executable(PP_HumanSeg_CPP main.cpp)
target_link_libraries(${PROJECT_NAME} onnxruntime ${OpenCV_LIBS})

set(CPACK_PROJECT_NAME ${PROJECT_NAME})
set(CPACK_PROJECT_VERSION ${PROJECT_VERSION})
include(CPack)
