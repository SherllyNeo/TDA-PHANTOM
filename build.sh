#!/bin/sh

rm -rf ./build

python3 setup.py sdist bdist_wheel

twine check dist/*

twine upload --verbose dist/*
