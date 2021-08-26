#!/bin/bash

cd numba
#find . -name "*.so" -exec rm {} \;
python setup.py clean --all
python setup.py build_ext --inplace
python setup.py install
cd ..
