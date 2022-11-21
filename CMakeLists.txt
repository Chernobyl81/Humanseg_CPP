cmake_minimum_required(VERSION 3.24.3)
project(PP_HumanSeg_CPP VERSION 0.1.0)

set(CMAKE_BUILD_TYPE Debug)
set(CMAKE_CXX_FLAGS "-O3")
set(CMAKE_CXX_STANDARD 17)
set(VERSION_STRING 0.1.0)
set(SOVERSION_STRING 0.1.0)
set(CMAKE_CXX_FLAGS "-Wno-deprecated ${CMAKE_CXX_FLAGS} -L/home/david/Projects/client/build/install/lib")

message(STATUS "BUILD FLAGS: ${CMAKE_CXX_FLAGS}")

include(CTest)
enable_testing()

find_package(OpenCV REQUIRED)

#set(TOOLKIT_ROOT_DIR ${CMAKE_SOURCE_DIR})
include_directories(
    ${CMAKE_CURRENT_SOURCE_DIR}/onnxruntime 
    ${CMAKE_CURRENT_SOURCE_DIR}/opencv2
)

option(USE_CUDA "use CUDA" OFF)
option(USE_TR "use TensorRT" OFF)
configure_file(config.h.in config.h @ONLY)

add_executable(${PROJECT_NAME} main.cc onnx/cv/human_segmentation.cc)

target_include_directories(
    ${PROJECT_NAME} PRIVATE 
    /home/david/Projects/client/build/install/include
    )

target_link_libraries(
    ${PROJECT_NAME} PRIVATE 
    onnxruntime 
    ${OpenCV_LIBS} 
    pthread 
    grpcclient
    protobuf
    )

set(CPACK_PROJECT_NAME ${PROJECT_NAME})
set(CPACK_PROJECT_VERSION ${PROJECT_VERSION})
include(CPack)
