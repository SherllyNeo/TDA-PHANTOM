import numpy as np
from typing import List

EULER_MASCHERONI = 0.57721566490153


class UNTest:
    def __init__(
        self,
        dgm:                 np.ndarray,
        k:                   int,
        alpha:               float = 0.05,
        complex:             str = "VR",
        correction_strategy: str = "BH",
        max_depth:           int = 1000,
        max_threshold=None,
        method: str = "universal_null:median"
    ):
        """
        Implimentation of the universal null hypothesis test from
        'A universal null‑distribution for topological data analysis'
        by Omer Bobrowski & Primoz Skraba
        """
        self.dgm = dgm
        self.k = k
        self.complex = complex  # currently only VR is supported
        self.max_depth = max_depth
        self.alpha = alpha
        self.correction_strategy = correction_strategy

        default_max = 10.0  # max epsilon for ripser for example
        if max_threshold is not None:
            self.max_threshold = max_threshold
        else:
            finite_deaths = self.dgm[np.isfinite(self.dgm[:, 1]), 1]
            self.max_threshold = float(np.max(finite_deaths)) if len(
                finite_deaths) > 0 else default_max
        self.A = None
        if self.complex == "VR":
            self.A = 1  # for Vietoris-Rips complex
        elif self.complex == "C":
            self.A = 0.5
        else:
            raise ValueError(
                f"Unknown complex type {self.complex}. Use 'VR' or 'C'.")

        births = self.dgm[:, 0]
        deaths = np.where(np.isfinite(
            self.dgm[:, 1]), self.dgm[:, 1], self.max_threshold)
        pi_values = deaths / births
        self.L_hat = None

        if self.method == "universal_null:median":
            self.L_hat = float(
                np.median(np.log(np.log(pi_values[(births > 0) & (pi_values > 1.0)]))))
        elif self.method == "universal_null:mean":
            self.L_hat = float(
                np.mean(np.log(np.log(pi_values[(births > 0) & (pi_values > 1.0)]))))
        else:
            raise ValueError(
                f"Unknown L_hat_strategy "
                f"{self.method}. Use method 'universal_null:mean' or 'universal_null:median'."
            )

    def _correct_alpha(self) -> float:
        if self.correction_strategy == "Bonferroni":
            return self.alpha / len(self.dgm)
        elif self.correction_strategy == "BH":
            return self.alpha
        else:
            raise ValueError(
                f"Unknown multiple testing correction strategy "
                f"{self.correction_strategy}. Use 'Bonferroni' or 'BH'."
            )

    def _pi_min(self, x: float) -> float:
        """
        minimum death/birth ratio such that the p-value for this diagram is under x
        """
        if x <= 0.0:
            return np.inf
        if x >= 1.0:
            return 1.0

        B = -EULER_MASCHERONI - self.A * self.L_hat
        l_thresh = np.log(-np.log(x))
        return float(np.exp(np.exp((l_thresh - B) / self.A)))

    def _find_threshold_for_infinite_cycles(self, t_0: float) -> float:
        """
        This algorithm gives us the threshold we need to use when calculating p_values
        for infinite cycles
        We choose the earliest-born infinite cycle (min(I)),
        while we could have chosen the latest-born (max(I)), or any intermediate value.
        This choice represents  trade-off between the number of iterations needed and the overestimation of τ.
        Choosing the earliest born cycle results in the smallest threshold, but with potentially more iterations, while choosing the last cycle will
        have fewer iterations with a possible overestimation of the threshold.
        """
        tau = t_0
        i = 0

        while i < self.max_depth:
            i += 1
            D = self.dgm[self.dgm[:, 0] <= tau]
            if len(D) == 0:
                break

            threshold = self._pi_min(self.alpha / len(D))
            inf_births = D[D[:, 1] >= tau, 0]
            inf_births = inf_births[inf_births > 0]

            if len(inf_births) == 0:
                break

            I_births = inf_births[tau / inf_births < threshold]

            if len(I_births) == 0:
                break

            tau = float(np.min(I_births) * threshold)

        return tau

    def _calculate_l(self) -> np.ndarray:
        """
        Normalises the persistence diagram values so the noise values follow
        the Lgumbel(0,1) distribution

        l_value is defined piecewise
        for finite death it is defined using a log(log()) transform and normalised
        for infinite death values we let death = max_threshold
        """

        births = self.dgm[:, 0]
        deaths = np.where(np.isfinite(
            self.dgm[:, 1]), self.dgm[:, 1], self.max_threshold)
        pi_values = deaths / births

        tau = self._find_threshold_for_infinite_cycles(self.max_threshold)
        inf_mask = ~np.isfinite(self.dgm[:, 1])
        pi_values[inf_mask] = tau / births[inf_mask]

        log_log_pi = np.full(len(self.dgm), np.nan)
        valid = (births > 0) & (pi_values > 1.0)
        log_log_pi[valid] = np.log(np.log(pi_values[valid]))

        l_values = self.A * log_log_pi - EULER_MASCHERONI - self.A * self.L_hat
        return l_values

    def _calculate_p_values_for_persistence_diagram(self) -> np.ndarray:
        """
        Finds the p value for each cycle.
        l_values for infinite cycles are handeled by _calculate_l
        """
        l_values = self._calculate_l()
        p_values = np.exp(-np.exp(l_values))
        self.p_values = p_values
        return p_values

    def calculate_significance_for_persistence_diagram(self) -> np.ndarray:
        """
        This actually applies your corrected alpha to check which p are significant
        """
        p_values = self._calculate_p_values_for_persistence_diagram()

        if self.correction_strategy == "Bonferroni":
            rejected = p_values < self._correct_alpha()

        elif self.correction_strategy == "BH":
            # sort p-values, find largest k where p_(k) <= k/m * alpha
            m = len(p_values)
            order = np.argsort(p_values)
            sorted_p = p_values[order]
            bh_line = (np.arange(1, m + 1) / m) * self.alpha
            below = np.where(sorted_p <= bh_line)[0]
            rejected = np.zeros(m, dtype=bool)
            if len(below) > 0:
                rejected[order[:below[-1] + 1]] = True

        else:
            raise ValueError(
                f"Unknown multiple testing correction strategy "
                f"{self.correction_strategy}. Use 'Bonferroni' or 'BH'."
            )

        return rejected

    def results(self) -> dict:
        """
        Returns a structured array with one row per bar.
        Cols: birth, death, pi, p_value, significant
        """
        p_values = self._calculate_p_values_for_persistence_diagram()
        rejected = p_values < self._correct_alpha()
        births = self.dgm[:, 0]
        deaths = self.dgm[:, 1]
        pi_values = np.where(
            np.isfinite(deaths),
            deaths / births,
            self.max_threshold / births,
        )
        if self.correction_strategy == "Bonferroni":
            alpha_thresh = self._correct_alpha()

        elif self.correction_strategy == "BH":
            # alpha_thresh = k/m * alpha where k = number of rejections
            # this is the BH threshold that was actually applied
            m = len(p_values)
            k = int(rejected.sum())
            if k > 0:
                alpha_thresh = (k / m) * self.alpha
            else:
                alpha_thresh = 0.0   # nothing rejected, threshold is below all bars

        threshold = self._pi_min(alpha_thresh)

        return {
            "results_array": np.column_stack([
                births,
                deaths,
                pi_values,
                p_values,
                rejected.astype(float),
            ]),
            "threshold": threshold


        }
