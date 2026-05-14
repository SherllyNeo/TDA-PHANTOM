import numpy as np
from typing import List
import math
import random
import gudhi

EULER_MASCHERONI = 0.57721566490153


class BNTest:
    def __init__(
        self,
        dgm:                 np.ndarray,
        k:                   int,
        alpha:               float = 0.05,
        complex:             str   = "VR",
        correction_strategy: str   = "BH",
        max_depth:           int   = 1000,
    ):
        """ 
        Inspired by the bottleneck hypothesis testing from
        'confidence sets for persistence diagrams'
        by Fasy et al
        """
        self.dgm                 = dgm
        self.k                   = k
        self.complex             = complex  # currently only VR is supported
        self.max_depth           = max_depth
        self.alpha               = alpha

    def w_infinity(self,dgm_1: np.ndarray, dgm_2: np.ndarray) -> float:
        w_inf_approx = gudhi.bottleneck_distance(dgm_1.tolist(), dgm_2.tolist(), e=0.01)
        return w_inf_approx

    def _subsampling_method(self,subsample_percentage: float = 0.6) -> float:
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


        n   = len(self.dgm)
        b = max(int(n / np.log(n)), 10)
        N = min(int(subsample_percentage * math.comb(n, b)),self.max_depth)

        T_j_array = np.zeros(N)
        for i in range(N):
            idx = np.random.choice(n, size=b, replace=False)
            subsample = self.dgm[idx]
            T_j_array[i] = self.w_infinity(subsample, self.dgm)

            T_j = self.w_infinity(subsamples[i],self.dgm)
            T_j_array[i] = T_j

        return T_j_array

    def results(self) -> np.ndarray:
        """
        Returns a structured array with one row per bar.
        Cols: birth, death, pers, p_value, significant
        """
        T_j_array = self._subsampling_method(return_null=True)
        c_n       = 2.0 * float(np.quantile(T_j_array, 1.0 - self.alpha))

        births    = self.dgm[:, 0]
        deaths    = self.dgm[:, 1]
        pers      = deaths - births


        # p_i = fraction of null distances >= pers_i / 2
        # (distance from bar i to the diagonal under L_inf)
        p_values  = np.array([
            float(np.mean(T_j_array >= p / 2)) for p in pers
        ])

        rejected  = pers > c_n

        return np.column_stack([
            births,
            deaths,
            pers,
            p_values,
            rejected.astype(float),
        ])
