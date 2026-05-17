import numpy as np
from typing import List, Optional
import math
import random
import gudhi
from scipy.spatial.distance import cdist
from tdaphantom.metrics.metrics import w_infinity, hausdorff_dist_matrix, hausdorff


class BNTest:
    def __init__(
        self,
        point_cloud:         np.ndarray,
        is_distance_matrix:  bool = False,
        dgm:                 np.ndarray = None,
        k:                   int = 1,
        alpha:               float = 0.05,
        complex:             str = "VR",
        max_depth:           int = 50,  # low and slow
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

    def _subsampling_method_via_persistence(self, subsample_percentage: float = 0.3) -> float:
        """
        DEPRECIATED - DO NOT USE
        E[W_infnity(hat(P),P)] != E[W_infnity(hat(P),subsample_hat(P)]
        """

        n = len(self.dgm)
        b = int(0.4*n)
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
        b = int(0.4*n)
        try:
            N = min(int(subsample_percentage * math.comb(n, b)), self.max_depth)
        except OverflowError:
            N = self.max_depth
        all_idx = np.arange(n)

        if not self.is_distance_matrix:
            D = cdist(self.pc, self.pc)
        else:
            D = self.pc

        T_j_array = np.zeros(N)
        for i in range(N):
            idx = np.random.choice(n, size=b, replace=False)
            # h(S_n, S_b*) = max_{i in S_n} min_{j in S_b*} D[i,j]
            T_j_array[i] = float(D[:, idx].min(axis=1).max())

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
