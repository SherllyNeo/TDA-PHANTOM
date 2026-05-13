#!/bin/sh

export PYTHONPATH=.
pytest tests/unit_tests/test_init.py -v

pytest tests/integration_tests/null_hypothesis_test.py -v
