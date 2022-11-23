#!/bin/bash

rm -rf build
cmake -B build
cmake --build build

# build/PP_HumanSeg_CPP \
# /home/david/Projects/Humanseg_CPP/model_repository/ppseg_onnx/1/model.onnx \
# /home/david/Desktop/R-C.jpg \
# images/background.jpg \
# 2 \
# CUDA

build/PP_HumanSeg_CPP \
localhost:8001 \
not_verbose \
/home/david/Desktop/R-C.jpg \
images/background.jpg \
1

