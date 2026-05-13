"""
Integration tests for Phantom.hypothesis_test

Tests that hypothesis_test produces sensible results for three
geometrically distinct point clouds with known topology:
    - Filled disk   (beta_1 = 0): no significant H₁ bars expected
    - Circle        (beta_1 = 1): one significant H₁ bar expected
    - Torus         (beta_1 = 2): two significant H₁ bars expected

Run with:
    pytest tests/integration_tests/null_hypothesis_test.py -v
"""

import warnings
import numpy as np
import pytest
from scipy.spatial.distance import cdist
from ripser import ripser

from tdaphantom.tdaphantom import Phantom



def _make_disk(n=2000, seed=0):
    rng = np.random.default_rng(seed)
    pts = []
    while len(pts) < n:
        b = rng.uniform(-1, 1, (n * 4, 2))
        pts.append(b[np.linalg.norm(b, axis=1) <= 1.0])
    return np.vstack(pts)[:n]


def _make_circle(n=2000, noise=0.03, seed=1):
    rng   = np.random.default_rng(seed)
    theta = rng.uniform(0, 2 * np.pi, n)
    pts   = np.stack([np.cos(theta), np.sin(theta)], axis=1)
    return pts + rng.normal(0, noise, pts.shape)


def _make_torus(n=3000, R=5.0, r=3.0, seed=2):
    rng   = np.random.default_rng(seed)
    theta = rng.uniform(0, 2 * np.pi, n)
    phi_list = []
    while len(phi_list) < n:
        phi_c = rng.uniform(0, 2 * np.pi, n * 2)
        w     = (R + r * np.cos(phi_c)) / (R + r)
        phi_list.append(phi_c[rng.uniform(0, 1, len(phi_c)) < w])
    phi = np.concatenate(phi_list)[:n]
    x   = (R + r * np.cos(phi)) * np.cos(theta)
    y   = (R + r * np.cos(phi)) * np.sin(theta)
    z   =  r * np.sin(phi)
    return np.stack([x, y, z], axis=1)


def _ripser_h1(pts):
    D   = cdist(pts, pts)
    dgm = ripser(D, distance_matrix=True, maxdim=1)["dgms"][1]
    # ripser may return bars with birth == death — strip them
    return dgm[dgm[:, 1] > dgm[:, 0]]



@pytest.fixture(scope="module")
def phantom_disk():
    dgm = _ripser_h1(_make_disk())
    return Phantom(dgm, k=1)


@pytest.fixture(scope="module")
def phantom_circle():
    dgm = _ripser_h1(_make_circle())
    return Phantom(dgm, k=1)


@pytest.fixture(scope="module")
def phantom_torus():
    dgm = _ripser_h1(_make_torus())
    return Phantom(dgm, k=1)


class TestHypothesisTestReturnStructure:

    def test_returns_dict(self, phantom_circle):
        result = phantom_circle.hypothesis_test()
        assert isinstance(result, dict)

    def test_universal_null_key_present(self, phantom_circle):
        result = phantom_circle.hypothesis_test(methods=["universal_null"])
        assert "universal_null" in result

    def test_result_is_ndarray(self, phantom_circle):
        result = phantom_circle.hypothesis_test()
        assert isinstance(result["universal_null"], np.ndarray)

    def test_result_has_five_columns(self, phantom_circle):
        result = phantom_circle.hypothesis_test()
        arr    = result["universal_null"]
        assert arr.shape[1] == 5

    def test_result_row_count_matches_diagram(self, phantom_circle):
        result = phantom_circle.hypothesis_test()
        arr    = result["universal_null"]
        assert arr.shape[0] == len(phantom_circle)

    def test_significant_column_is_binary(self, phantom_circle):
        result = phantom_circle.hypothesis_test()
        sig    = result["universal_null"][:, 4]
        assert set(np.unique(sig)).issubset({0.0, 1.0})

    def test_p_values_between_zero_and_one(self, phantom_circle):
        result   = phantom_circle.hypothesis_test()
        p_values = result["universal_null"][:, 3]
        finite   = p_values[np.isfinite(p_values)]
        assert np.all(finite >= 0.0)
        assert np.all(finite <= 1.0)

    def test_births_match_diagram(self, phantom_circle):
        result = phantom_circle.hypothesis_test()
        arr    = result["universal_null"]
        np.testing.assert_array_equal(arr[:, 0], phantom_circle.dgm[:, 0])

    def test_deaths_match_diagram(self, phantom_circle):
        result = phantom_circle.hypothesis_test()
        arr    = result["universal_null"]
        np.testing.assert_array_equal(arr[:, 1], phantom_circle.dgm[:, 1])


class TestHypothesisTestAlphaValidation:

    def test_alpha_not_float_raises(self, phantom_circle):
        with pytest.raises(TypeError):
            phantom_circle.hypothesis_test(alpha="1")

    def test_alpha_above_one_raises(self, phantom_circle):
        with pytest.raises(ValueError):
            phantom_circle.hypothesis_test(alpha=1.5)

    def test_alpha_below_zero_raises(self, phantom_circle):
        with pytest.raises(ValueError):
            phantom_circle.hypothesis_test(alpha=-0.1)

    def test_alpha_zero_raises(self, phantom_circle):
        with pytest.raises(ValueError):
            phantom_circle.hypothesis_test(alpha=0.0)

    def test_alpha_one_raises(self, phantom_circle):
        with pytest.raises(ValueError):
            phantom_circle.hypothesis_test(alpha=1.0)


class TestHypothesisTestGeometry:

    def test_disk_no_significant_bars(self, phantom_disk):
        """Filled disk is contractible — no genuine H_1 loops."""
        result = phantom_disk.hypothesis_test(alpha=0.01)
        sig    = result["universal_null"][:, 4]
        assert sig.sum() == 0

    def test_circle_one_significant_bar(self, phantom_circle):
        """Circle has beta_1 = 1 — exactly one significant H_1 bar."""
        result = phantom_circle.hypothesis_test(alpha=0.01)
        sig    = result["universal_null"][:, 4]
        assert sig.sum() == 1

    def test_circle_significant_bar_has_largest_persistence(self, phantom_circle):
        """The one significant bar should be the longest-lived."""
        result = phantom_circle.hypothesis_test(alpha=0.01)
        arr    = result["universal_null"]
        pers   = arr[:, 1] - arr[:, 0]
        sig_idx = np.where(arr[:, 4] == 1.0)[0]
        assert sig_idx[0] == np.argmax(pers)

    def test_torus_detects_both_loops_relaxed_alpha(self, phantom_torus):
        """Both torus loops detected at a less conservative alpha."""
        result = phantom_torus.hypothesis_test(alpha=0.20)
        sig    = result["universal_null"][:, 4]
        assert sig.sum() >= 2

    def test_torus_significant_bars_are_among_longest(self, phantom_torus):
        """Significant bars should be among the top bars by persistence."""
        result   = phantom_torus.hypothesis_test(alpha=0.01)
        arr      = result["universal_null"]
        pers     = arr[:, 1] - arr[:, 0]
        sig_idx  = set(np.where(arr[:, 4] == 1.0)[0])
        top5_idx = set(np.argsort(pers)[-5:])
        assert sig_idx.issubset(top5_idx)

    def test_significant_bars_have_smaller_p_values(self, phantom_circle):
        """Significant bars must have strictly smaller p-values than noise bars."""
        result = phantom_circle.hypothesis_test(alpha=0.01)
        arr = result["universal_null"]
        p_values = arr[:, 3]
        sig = arr[:, 4] == 1.0
        if sig.any() and (~sig).any():
            assert p_values[sig].max() < p_values[~sig].min()

    def test_stricter_alpha_does_not_increase_rejections(self, phantom_circle):
        """Fewer or equal bars should be significant at a stricter alpha."""
        result_loose  = phantom_circle.hypothesis_test(alpha=0.10)
        result_strict = phantom_circle.hypothesis_test(alpha=0.01)
        n_loose  = result_loose["universal_null"][:, 4].sum()
        n_strict = result_strict["universal_null"][:, 4].sum()
        assert n_strict <= n_loose
