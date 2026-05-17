import gudhi
import numpy as np
import random
from scipy.spatial.distance import cdist


def w_infinity(dgm_1: np.ndarray, dgm_2: np.ndarray) -> float:
    w_inf_approx = gudhi.bottleneck_distance(
        dgm_1.tolist(), dgm_2.tolist(), e=0.01)
    return w_inf_approx


def hausdorff_directed(A: np.ndarray, B: np.ndarray) -> float:
    """
    Directed Hausdorff distance from Algorithm 2 in "An Efficient Algorithm for Calculating the Exact Hausdorff Distance"
    from Taha & Hanbury
    """
    rng = np.random.default_rng()
    A = rng.permutation(A)
    B = rng.permutation(B)

    c_max = 0.0
    for a in A:
        c_min = np.inf
        for b in B:
            d = float(np.linalg.norm(a - b))
            if d < c_max:
                c_min = d  # paper omits this but it seems required
                break
            if d < c_min:
                c_min = d
        if c_min > c_max:
            c_max = c_min

    return c_max


def hausdorff(A: np.ndarray, B: np.ndarray) -> float:
    """
    Symmetric Hausdorff distance
    """
    return max(hausdorff_directed(A, B), hausdorff_directed(B, A))


def hausdorff_directed_dist_matrix(A: np.ndarray, B: np.ndarray) -> float:
    """
    Taken from "an Efficient Algorithm for Calculating the Exact Hausdorff Distance"
    by Abdel Aziz Taha and Allan Hanbury.

    This does not need to compute euclidean distance as this is done for us
    """
    rng = np.random.default_rng()
    A_idx = rng.permutation(A_idx)
    B_idx = rng.permutation(B_idx)

    c_max = 0.0
    for i in A_idx:
        c_min = np.inf
        for j in B_idx:
            d = float(A[i, j])
            if d < c_max:
                c_min = d
                break
            if d < c_min:
                c_min = d
        if c_min > c_max:
            c_max = c_min

    return c_max


def hausdorff_dist_matrix(A_idx: np.ndarray, B_idx: np.ndarray) -> float:
    """
    Symmetric Hausdorff distance H(A,B) = max(h(A,B), h(B,A))
    """
    return max(
        hausdorff_directed_dist_matrix(A_idx, B_idx),
        hausdorff_directed_dist_matrix(B_idx, A_idx),
    )
