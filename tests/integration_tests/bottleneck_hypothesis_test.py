"""
Integration tests for Phantom.hypothesis_test

Tests that hypothesis_test produces sensible results for three
geometrically distinct point clouds with known topology:
    - Filled disk   (beta_1 = 0): no significant H_1 bars expected
    - Circle        (beta_1 = 1): one significant H_1 bar expected
    - Torus         (beta_1 = 2): two significant H_1 bars expected

Run with:
    pytest tests/integration_tests/bottleneck_hypothesis_test.py -v
"""

import warnings
import numpy as np
import pytest
from scipy.spatial.distance import cdist
import os

from tdaphantom.tdaphantom import Phantom

PRECOMPUTED_DIR = "../../precomputed/"


def _make_disk(n=2000, seed=0):
    rng = np.random.default_rng(seed)
    pts = []
    while len(pts) < n:
        b = rng.uniform(-1, 1, (n * 4, 2))
        pts.append(b[np.linalg.norm(b, axis=1) <= 1.0])
    return np.vstack(pts)[:n]


def _make_circle(n=2000, noise=0.03, seed=1):
    rng = np.random.default_rng(seed)
    theta = rng.uniform(0, 2 * np.pi, n)
    pts = np.stack([np.cos(theta), np.sin(theta)], axis=1)
    return pts + rng.normal(0, noise, pts.shape)


def _make_torus(n=3000, R=5.0, r=3.0, seed=2):
    rng = np.random.default_rng(seed)
    theta = rng.uniform(0, 2 * np.pi, n)
    phi_list = []
    while len(phi_list) < n:
        phi_c = rng.uniform(0, 2 * np.pi, n * 2)
        w = (R + r * np.cos(phi_c)) / (R + r)
        phi_list.append(phi_c[rng.uniform(0, 1, len(phi_c)) < w])
    phi = np.concatenate(phi_list)[:n]
    x = (R + r * np.cos(phi)) * np.cos(theta)
    y = (R + r * np.cos(phi)) * np.sin(theta)
    z = r * np.sin(phi)
    return np.stack([x, y, z], axis=1)
