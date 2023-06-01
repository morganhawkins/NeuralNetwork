#!/usr/bin/env bash

#setting directory to library folder
echo
echo

cd /Users/morganhawkins/Desktop/first_python_lib

echo 
echo 


python setup.py pytest

echo 
echo 

python setup.py bdist_wheel

echo 
echo 

pip3 uninstall /Users/morganhawkins/Desktop/first_python_lib/dist/nnlearn-1.0-py3-none-any.whl

echo 
echo 

pip3 install /Users/morganhawkins/Desktop/first_python_lib/dist/nnlearn-1.0-py3-none-any.whl




