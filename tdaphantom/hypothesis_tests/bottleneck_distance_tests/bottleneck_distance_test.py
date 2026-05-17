import numpy as np
from typing import List, Optional
import math
import random
import gudhi
from scipy.spatial.distance import cdist


class BNTest:
    def __init__(
        self,
        point_cloud:         np.ndarray,
        is_distance_matrix:  bool = False,
        dgm:                 np.ndarray = None,
        k:                   int = 1,
        alpha:               float = 0.05,
        complex:             str = "VR",
        max_depth:           int = 100,  # low and slow
        method: str = "bottleneck:subsample"
    ):
        """
        The bottleneck hypothesis testing from
        'confidence sets for persistence diagrams'
        by Fasy et al
        """
        self.method = method
        self.method_calls = {"bottleneck:subsample": self.subsample}
        self.pc = point_cloud
        self.dgm = dgm
        self.is_distance_matrix = is_distance_matrix
        self.k = k
        self.complex = complex  # currently only VR is supported
        self.max_depth = max_depth
        self.alpha = alpha

    def w_infinity(self, dgm_1: np.ndarray, dgm_2: np.ndarray) -> float:
        w_inf_approx = gudhi.bottleneck_distance(
            dgm_1.tolist(), dgm_2.tolist(), e=0.01)
        return w_inf_approx

    def hausdorff_directed(self, A: np.ndarray, B: np.ndarray) -> float:
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

    def hausdorff(self, A: np.ndarray, B: np.ndarray) -> float:
        """
        Symmetric Hausdorff distance
        """
        return max(self.hausdorff_directed(A, B), self.hausdorff_directed(B, A))

    def hausdorff_directed_dist_matrix(self, A: np.ndarray, B: np.ndarray) -> float:
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
                d = float(self.distance_matrix[i, j])
                if d < c_max:
                    c_min = d
                    break
                if d < c_min:
                    c_min = d
            if c_min > c_max:
                c_max = c_min

        return c_max

    def hausdorff_dist_matrix(self, A_idx: np.ndarray, B_idx: np.ndarray) -> float:
        """
        Symmetric Hausdorff distance H(A,B) = max(h(A,B), h(B,A))
        """
        return max(
            self.hausdorff_directed_dist_matrix(A_idx, B_idx),
            self.hausdorff_directed_dist_matrix(B_idx, A_idx),
        )

    def _subsampling_method_via_persistence(self, subsample_percentage: float = 0.3) -> float:
        """
        Fasy et al. 4.2 subsampling
        b   = subsample size = O(n / log(n))
        N   = number of subsamples (theory uses n choose b, but we will use a subset)
        The paper uses the Hausdorff distance on the original point clouds, we do not have access to this
        so we will use the bottleneck distance between subsamples of the persistence diagrams.

        This should be justified at some point.

        A bar with persistence > C_b is significant at level alpha.
        By the bottleneck stability theorem, W_inf(PH(S_n), PH(P)) <= C_b
        with probability >= 1 - alpha.
        """

        n = len(self.dgm)
        b = int(0.8*n)
        try:
            N = min(int(subsample_percentage * math.comb(n, b)), self.max_depth)
        except OverflowError:
            N = self.max_depth

        T_j_array = np.zeros(N)
        for i in range(N):
            idx = np.random.choice(n, size=b, replace=False)
            subsample = self.dgm[idx]
            T_j_array[i] = self.w_infinity(subsample, self.dgm)

    def _subsampling_method(self, subsample_percentage: float = 0.8) -> np.ndarray:
        """
        Fasy et al. 4.2 subsampling
        b   = subsample size = O(n / log(n))
        N   = number of subsamples (theory uses n choose b, but we will use a subset)
        A bar with persistence > C_b is significant at level alpha.
        By the bottleneck stability theorem, W_inf(PH(S_n), PH(P)) <= C_b
        with probability >= 1 - alpha.
        """
        n = len(self.pc)
        b = int(0.8*n)
        try:
            N = min(int(subsample_percentage * math.comb(n, b)), self.max_depth)
        except OverflowError:
            N = self.max_depth
        all_idx = np.arange(n)

        T_j_array = np.zeros(N)
        for i in range(N):
            idx = np.random.choice(n, size=b, replace=False)
            if self.is_distance_matrix:
                T_j_array[i] = self.hausdorff_dist_matrix(idx, all_idx)
            else:
                T_j_array[i] = self.hausdorff(self.pc[idx], self.pc)

        return T_j_array

    def subsample(self):
        """
        Calls subsampling method to calculate c_n
        """
        T_j_array = self._subsampling_method()
        c_n = float(np.quantile(T_j_array, 1.0 - self.alpha))
        return c_n, T_j_array

    def results(self) -> dict:
        """
        Returns a structured array with one row per bar.
        Cols: birth, death, pers, p_value, significant
        """
        c_n = -np.inf
        if self.method in self.method_calls.keys():
            # ugly fix, make elegant later
            c_n, T_j_array = self.method_calls[self.method]()

        births = self.dgm[:, 0]
        deaths = self.dgm[:, 1]
        pers = deaths - births

        # p_i = fraction of null distances >= pers_i / 2
        # (distance from bar i to the diagonal under L_inf)
        p_values = np.array([
            float(np.mean(T_j_array >= p / 2)) for p in pers
        ])

        rejected = pers > 2*c_n

        return {
            "results_array": np.column_stack([
                births,
                deaths,
                pers,
                p_values,
                rejected.astype(float),
            ]),
            "threshold": 2*c_n  # used for diagram
        }
