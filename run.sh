#!/bin/sh
cd ./core/ctree/
CC="clang --std=c++11" sh make.sh
cd ../../
python3 main.py