#!/bin/sh
g++ -std=c++17 -fPIC -shared -O3 -march=native cpp_metrics.cpp -o cpp_metrics.so