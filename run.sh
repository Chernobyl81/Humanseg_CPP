#!/bin/bash

rm -rf build
cmake -B build
cmake --build build
build/PP_HumanSeg_CPP