#!/bin/bash

export CC=gcc
export CXX=g++
 
pushd ../../NumbaWithOpenmp
find . -name "*.so" -exec rm {} \;
python setup.py clean --all
python setup.py build_ext --inplace
python setup.py develop
popd
