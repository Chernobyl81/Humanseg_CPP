#!/bin/bash

rm -rf build
cmake -B build
cmake --build build

build/PP_HumanSeg_CPP \
models/ppseg_lite_portrait_398x224_with_softmax.onnx \
/home/david/Desktop/R-C.jpg  \
images/background.jpg \
2