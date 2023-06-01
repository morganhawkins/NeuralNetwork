#!/usr/bin/env bash

#setting directory to library folder
echo
echo

cd /Users/morganhawkins/Desktop/first_python_lib

echo set working directory to library folder
echo
echo



echo
echo

python setup.py pytest

echo tests passed!
echo
echo



echo
echo

python setup.py bdist_wheel

echo built wheel file
echo
echo


echo
echo

pip3 uninstall /Users/morganhawkins/Desktop/first_python_lib/dist/nnlearn-1.0-py3-none-any.whl

echo
echo

echo uninstalled old library

echo
echo

pip3 install /Users/morganhawkins/Desktop/first_python_lib/dist/nnlearn-1.0-py3-none-any.whl

echo installed library to python
echo
echo



