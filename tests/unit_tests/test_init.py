"""
Unit tests for tdaphantom.Phantom

Tests cover:
  * Valid construction
  * k validation
  * dgm coercion
  * Shape validation
  * Value validation
  * Properties

Run with:
    pytest tests/unit_tests/test_init.py -v
"""

import warnings
import numpy as np
import pytest
from tdaphantom.tdaphantom import Phantom
