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


def _make_dgm(*rows):
    return np.array(rows, dtype=float)


VALID_H1 = _make_dgm(
    (0.1, 0.8),
    (0.2, 0.4),
    (0.05, 0.3),
)

VALID_H1_WITH_INF = _make_dgm(
    (0.1, 0.8),
    (0.2, np.inf),
)

VALID_H0 = _make_dgm(
    (0.0, 0.5),
    (0.0, np.inf),
)


class TestValidConstruction:

    def test_basic_h1(self):
        p = Phantom(VALID_H1, k=1)
        assert p.k == 1
        assert len(p) == 3

    def test_h0_with_single_infinite(self):
        with warnings.catch_warnings():
            warnings.simplefilter("error")
            p = Phantom(VALID_H0, k=0)
        assert len(p) == 2

    def test_h2(self):
        dgm = _make_dgm((0.3, 0.9), (0.1, 0.6))
        p   = Phantom(dgm, k=2)
        assert p.k == 2

    def test_single_bar(self):
        dgm = _make_dgm((0.0, 1.0))
        p   = Phantom(dgm, k=1)
        assert len(p) == 1

    def test_birth_zero_is_valid(self):
        dgm = _make_dgm((0.0, 0.5))
        p   = Phantom(dgm, k=1)
        assert p.dgm[0, 0] == 0.0

    def test_np_integer_k(self):
        dgm = _make_dgm((0.1, 0.5))
        p   = Phantom(dgm, k=np.int64(1))
        assert p.k == 1

    def test_list_input_coerced(self):
        dgm = [[0.1, 0.8], [0.2, 0.4]]
        p   = Phantom(dgm, k=1)
        assert isinstance(p.dgm, np.ndarray)

    def test_integer_array_coerced_to_float(self):
        dgm = np.array([[0, 1], [0, 2]], dtype=int)
        p   = Phantom(dgm, k=1)
        assert p.dgm.dtype == float

    def test_infinite_death_accepted(self):
        dgm = _make_dgm((0.1, np.inf))
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", UserWarning)
            p = Phantom(dgm, k=1)
        assert not np.isfinite(p.dgm[0, 1])

    def test_empty_dgm_warns(self):
        dgm = np.empty((0, 2), dtype=float)
        with pytest.warns(UserWarning):
            Phantom(dgm, k=1)


class TestKValidation:

    def test_k_float_raises(self):
        with pytest.raises(TypeError):
            Phantom(VALID_H1, k=1.0)

    def test_k_string_raises(self):
        with pytest.raises(TypeError):
            Phantom(VALID_H1, k="1")

    def test_k_none_raises(self):
        with pytest.raises(TypeError):
            Phantom(VALID_H1, k=None)

    def test_k_negative_raises(self):
        with pytest.raises(ValueError):
            Phantom(VALID_H1, k=-1)

    def test_k_zero_valid(self):
        p = Phantom(VALID_H0, k=0)
        assert p.k == 0


class TestDgmCoercion:

    def test_nested_list_accepted(self):
        dgm = [[0.0, 1.0], [0.1, 0.5]]
        p   = Phantom(dgm, k=1)
        assert p.dgm.shape == (2, 2)

    def test_tuple_of_tuples_accepted(self):
        dgm = ((0.0, 1.0), (0.2, 0.9))
        p   = Phantom(dgm, k=1)
        assert p.dgm.shape == (2, 2)

    def test_non_numeric_raises(self):
        with pytest.raises(TypeError):
            Phantom([["a", "b"], ["c", "d"]], k=1)

    def test_none_raises(self):
        with pytest.raises((TypeError, ValueError)):
            Phantom(None, k=1)


class TestShapeValidation:

    def test_1d_raises(self):
        with pytest.raises(ValueError):
            Phantom(np.array([0.1, 0.8]), k=1)

    def test_3d_raises(self):
        with pytest.raises(ValueError):
            Phantom(np.ones((2, 2, 2)), k=1)

    def test_wrong_columns_raises(self):
        with pytest.raises(ValueError):
            Phantom(np.ones((3, 3)), k=1)

    def test_single_column_raises(self):
        with pytest.raises(ValueError):
            Phantom(np.ones((3, 1)), k=1)

    def test_correct_shape_passes(self):
        dgm = np.ones((5, 2))
        dgm[:, 1] += 1.0
        p = Phantom(dgm, k=1)
        assert p.dgm.shape == (5, 2)


class TestValueValidation:

    def test_nan_in_births_raises(self):
        dgm = _make_dgm((np.nan, 0.5), (0.1, 0.8))
        with pytest.raises(ValueError):
            Phantom(dgm, k=1)

    def test_nan_in_deaths_raises(self):
        dgm = _make_dgm((0.1, np.nan), (0.2, 0.8))
        with pytest.raises(ValueError):
            Phantom(dgm, k=1)

    def test_inf_birth_raises(self):
        dgm = _make_dgm((np.inf, 1.0))
        with pytest.raises(ValueError):
            Phantom(dgm, k=1)

    def test_negative_birth_raises(self):
        dgm = _make_dgm((-0.1, 0.5))
        with pytest.raises(ValueError):
            Phantom(dgm, k=1)

    def test_death_equal_birth_raises(self):
        dgm = _make_dgm((0.5, 0.5))
        with pytest.raises(ValueError):
            Phantom(dgm, k=1)

    def test_death_less_than_birth_raises(self):
        dgm = _make_dgm((0.8, 0.1))
        with pytest.raises(ValueError):
            Phantom(dgm, k=1)

    def test_mixed_valid_invalid_raises(self):
        dgm = _make_dgm((0.1, 0.8), (0.9, 0.2), (0.0, 0.5))
        with pytest.raises(ValueError):
            Phantom(dgm, k=1)

    def test_h0_wrong_inf_count_warns(self):
        dgm = _make_dgm((0.0, np.inf), (0.0, np.inf))
        with pytest.warns(UserWarning):
            Phantom(dgm, k=0)

    def test_h0_no_inf_warns(self):
        dgm = _make_dgm((0.0, 0.5), (0.1, 0.8))
        with pytest.warns(UserWarning):
            Phantom(dgm, k=0)

    def test_multiple_nan_raises(self):
        dgm = _make_dgm((np.nan, np.nan))
        with pytest.raises(ValueError):
            Phantom(dgm, k=1)


class TestProperties:

    def setup_method(self):
        self.dgm = _make_dgm(
            (0.1, 0.8),
            (0.2, 0.4),
            (0.05, np.inf),
        )
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", UserWarning)
            self.p = Phantom(self.dgm, k=1)

    def test_finite_shape(self):
        assert self.p.finite.shape == (2, 2)

    def test_finite_excludes_inf(self):
        assert np.all(np.isfinite(self.p.finite[:, 1]))

    def test_finite_values(self):
        np.testing.assert_array_equal(
            self.p.finite,
            _make_dgm((0.1, 0.8), (0.2, 0.4)),
        )

    def test_infinite_shape(self):
        assert self.p.infinite.shape == (1, 2)

    def test_infinite_values(self):
        assert self.p.infinite[0, 0] == pytest.approx(0.05)
        assert not np.isfinite(self.p.infinite[0, 1])

    def test_persistences_length(self):
        assert len(self.p.persistences) == 3

    def test_persistences_finite_values(self):
        finite_pers = self.p.persistences[np.isfinite(self.p.persistences)]
        np.testing.assert_allclose(np.sort(finite_pers), [0.2, 0.7])

    def test_persistences_contains_inf(self):
        assert np.any(~np.isfinite(self.p.persistences))

    def test_persistences_finite_all_positive(self):
        finite_pers = self.p.persistences[np.isfinite(self.p.persistences)]
        assert np.all(finite_pers > 0)

    def test_len_total(self):
        assert len(self.p) == 3

    def test_len_empty(self):
        dgm = np.empty((0, 2), dtype=float)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", UserWarning)
            p = Phantom(dgm, k=1)
        assert len(p) == 0

    def test_repr_contains_dimension(self):
        assert "H_1" in repr(self.p)

    def test_repr_contains_bar_count(self):
        assert "3 bars" in repr(self.p)

    def test_repr_contains_finite_count(self):
        assert "2 finite" in repr(self.p)

    def test_repr_contains_infinite_count(self):
        assert "1 infinite" in repr(self.p)

    def test_repr_format_h0(self):
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", UserWarning)
            p = Phantom(VALID_H0, k=0)
        assert "H_0" in repr(p)

    def test_all_finite_infinite_property_empty(self):
        p = Phantom(VALID_H1, k=1)
        assert len(p.infinite) == 0

    def test_all_finite_finite_property_full(self):
        p = Phantom(VALID_H1, k=1)
        assert len(p.finite) == len(p)
