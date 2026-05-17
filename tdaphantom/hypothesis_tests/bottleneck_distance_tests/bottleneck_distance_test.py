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
        self.method_calls = {
            "bottleneck:subsample": self.subsample,
            "bottleneck:concentration": self.concentration,
            "bottleneck:shells": self.shells,
            "bottleneck:density": self.density
        }
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
        Fasy et al. 4.1 subsampling
        b   = subsample size = O(n / log(n))
        N   = number of subsamples (theory uses n choose b, but we will use a subset)
        A bar with persistence > C_b is significant at level alpha.
        By the bottleneck stability theorem, W_inf(PH(S_n), PH(P)) <= C_b
        with probability >= 1 - alpha.

        From the paper:
        P(H(S_n, M) > C_n) <= alpha + O((b/n)^(1/4))

        The bias term O((b/n)^(1/4)) -> 0 as b/n -> 0, so theory requires b << n.
        The paper uses b = O(n / log(n)) for the theoretical guarantee.
        In practice, larger b gives smaller c_n and more power but looser theory guarantees.
        """
        n = len(self.pc)
        b = min(int(3.5*(n / np.log(n))), int(0.8*n))
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

        # bias_order = (b/n)**(0.25)

        return T_j_array

    def subsample(self):
        """
        Calls subsampling method to calculate c_n and p_values
        """
        births = self.dgm[:, 0]
        deaths = self.dgm[:, 1]
        pers = deaths - births
        T_j_array = self._subsampling_method()
        c_n = float(np.quantile(T_j_array, 1.0 - self.alpha))
        p_values = np.array([
            float(np.mean(T_j_array >= p / 2)) for p in pers
        ])
        return c_n, p_values

    def concentration_of_measure_method(self):
        """
        Fasy et al. 4.2 concentration of measure

        From the paper:
        P(H(S_n, M) > \hat(t_n)) <= alpha + O((log(n)/n)^(1/(2+d)))
        """

        ...

    def concentration(self):
        """
        Calls concentration method to calculate c_n
        """
        c_n = self.concentration_of_measure_method()
        return c_n, None

    def shells_method(self):
        """
        Fasy et al. 4.3 method of shells

        From the paper:
        P(H(S_{2,n}, M) > \hat(t_{1,n}) <= alpha + O(r_n)
        """
        ...

    def shells(self):
        """
        Calls shells method to calculate c_n
        """
        c_n = self.shells_method()
        return c_n, None

    def denisty_method(self):
        """
        Fasy et al. 4.4 Density estimation

        From the paper:
        P(||\hat{p}_h - p_h||_infinity > Z_alpha / sqrt(nh^D) ) <= alpha + O(log(n)/nh^D)^((4+D)/(4+2D))
        """
        ...

    def denisty(self):
        """
        Calls shells method to calculate c_n
        """
        c_n = self.density_method()
        return c_n, None

    def results(self) -> dict:
        """
        Returns a structured array with one row per bar.
        Cols: birth, death, pers, p_value, significant

        and the threshold c_n
        """
        c_n = -np.inf
        p_values = None
        if self.method in self.method_calls.keys():
            c_n, p_values = self.method_calls[self.method]()

        births = self.dgm[:, 0]
        deaths = self.dgm[:, 1]
        pers = deaths - births

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
