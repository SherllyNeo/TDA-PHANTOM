#!/bin/sh

rm -rf dist/ build/ *.egg-info

python3 setup.py sdist bdist_wheel

twine check dist/*

twine upload --verbose dist/*
