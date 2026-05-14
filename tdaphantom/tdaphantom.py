import numpy as np
import warnings
from .hypothesis_tests.universal_null_tests.universal_null_hypothesis_test import UNTest
from .hypothesis_tests.bottleneck_distance_tests.bottleneck_distance_test import BNTest
from typing import List
import matplotlib.pyplot as plt


class Phantom:
    """
    Persistence diagram container.

    dgm: np.ndarray, shape (n, 2)
        Persistence diagram. Each row is [birth, death].
        Infinite deaths (np.inf) are permitted.
    k: int
        Homological dimension of the diagram.
        0 = connected components (H_0)
        1 = loops / holes      (H_1)
        2 = voids / cavities   (H_2)
        etc
    """

    def __init__(self, dgm: np.ndarray, k: int):

        if not isinstance(k, (int, np.integer)):
            raise TypeError(
                f"k must be an integer, got {type(k).__name__}."
            )
        if k < 0:
            raise ValueError(
                f"k must be non-negative, got k={k}."
            )

        try:
            dgm = np.asarray(dgm, dtype=float)
        except (TypeError, ValueError) as e:
            raise TypeError(
                f"dgm could not be converted to a numpy float array: {e}"
            ) from e

        if dgm.ndim != 2:
            raise ValueError(
                f"dgm must be a 2D array of shape (n, 2), "
                f"got shape {dgm.shape}."
            )
        if dgm.shape[1] != 2:
            raise ValueError(
                f"dgm must have exactly 2 columns [birth, death] got {dgm.shape[1]} columns."
            )

        if dgm.shape[0] == 0:
            warnings.warn(
                "dgm is empty. All tests will return trivially ",
                UserWarning, stacklevel=2,
            )

        births = dgm[:, 0]
        deaths = dgm[:, 1]

        if not np.all(np.isfinite(births)):
            raise ValueError(
                f"All birth values must be finite. Found {np.sum(~np.isfinite(births))} non-finite birth(s)."
            )

        if np.any(births < 0):
            raise ValueError(
                f"All birth values must be positive. Found {np.sum(births < 0)} negative birth(s)."
            )

        finite_mask = np.isfinite(deaths)
        n_bad = np.sum(deaths[finite_mask] <= births[finite_mask])
        if n_bad > 0:
            raise ValueError(
                f"All death values must be strictly greater than their corresponding birth values. Found {n_bad} bar(s) where death <= birth."
            )

        if np.any(np.isnan(dgm)):
            raise ValueError(
                "dgm contains NaN values. Deaths may be np.inf but not NaN."
            )

        # H_0 specific: should have only one infinite bar
        n_inf = int(np.sum(~np.isfinite(deaths)))
        if k == 0 and n_inf != 1:
            warnings.warn(
                f"H_0 diagrams contain exactly 1 infinite bar. The last surviving component. Found {n_inf}.",
                UserWarning, stacklevel=2,
            )

        self.dgm = dgm
        self.k = k

    @property
    def finite(self) -> np.ndarray:
        """Rows where death is finite """
        return self.dgm[np.isfinite(self.dgm[:, 1])]

    @property
    def infinite(self) -> np.ndarray:
        """Rows where death is infinite """
        return self.dgm[~np.isfinite(self.dgm[:, 1])]

    @property
    def persistences(self) -> np.ndarray:
        """death − birth - may contain infinite persistence """
        return self.dgm[:, 1] - self.dgm[:, 0]

    def __len__(self) -> int:
        return len(self.dgm)

    def __repr__(self) -> str:
        n_fin = len(self.finite)
        n_inf = len(self.infinite)
        dim_name = f"H_{self.k}"
        return (
            f"Phantom({dim_name}, {len(self.dgm)} bars: "
            f"{n_fin} finite, {n_inf} infinite)"
        )

    def hypothesis_test(
        self,
        alpha:             float = 0.05,
        methods:           list[str] = ["universal_null", "bottleneck"],
        correction_method: str = "BH",
    ) -> dict:
        """
        Calculates the p_values and checks significance for the persistence diagrams.
        Alpha is the significance.
        Methods include:
            universal_null test by Omer Bobrowski & Primoz Skraba
            bottleneck test by Fasy et al.
        """
        if not isinstance(alpha, float):
            raise TypeError(
                f"alpha must be a float, got {type(alpha).__name__}."
            )
        if alpha <= 0 or alpha >= 1:
            raise ValueError(
                f"alpha, the significance, must be between 0 and 1, got alpha={alpha}."
            )

        results = {}

        if "universal_null" in methods:
            test = UNTest(
                dgm=self.dgm,
                k=self.k,
                alpha=alpha,
                correction_strategy=correction_method,
            )
            results["universal_null"] = test.results()

        if "bottleneck" in methods:
            test = BNTest(
                dgm=self.dgm,
                k=self.k,
                alpha=alpha,
            )
            results["bottleneck"] = test.results()

        self._cached_results = results
        return results

    def display_results(
        self,
        results: dict = None,
        method:  str = "all",
        plot:    str = "both",
    ) -> None:
        """
        Visualise hypothesis test results
        """
        if results is None:
            if not hasattr(self, "_cached_results"):
                raise ValueError(
                    "require results passed in or a previously ran hypothesis_test.")
            results = self._cached_results

        if not isinstance(results, dict):
            raise ValueError(
                "results must be a dict returned by hypothesis_test.")

        if plot not in ("diagram", "barcode", "both"):
            raise ValueError(
                f"plot must be 'diagram', 'barcode', or 'both', got {plot!r}.")

        available = list(results.keys())
        if method == "all":
            methods_to_plot = available
        else:
            if method not in available:
                raise ValueError(
                    f"method {method!r} not found in results. Available: {available}")
            methods_to_plot = [method]

        n_cols = 2 if plot == "both" else 1
        n_rows = len(methods_to_plot)

        fig, axes = plt.subplots(n_rows, n_cols, figsize=(
            6 * n_cols, 5 * n_rows), squeeze=False)
        fig.suptitle(f"H_{self.k} persistence results", fontsize=14)

        for row, mname in enumerate(methods_to_plot):
            res = results[mname]
            results_array = res["results_array"]
            thr = res.get("threshold", np.nan)
            births = results_array[:, 0]
            deaths = results_array[:, 1]
            pers = deaths - births
            sig = results_array[:, 4].astype(bool)
            ax_idx = 0

            if plot in ("diagram", "both"):
                ax = axes[row, ax_idx]
                lim = deaths[np.isfinite(deaths)].max() * 1.05
                xs = np.linspace(0, lim, 300)

                ax.plot([0, lim], [0, lim], "k--", lw=0.8,
                        alpha=0.4, label="diagonal")
                if not np.isnan(thr):
                    if mname == "universal_null":
                        # d = b * pi*  — ray from origin
                        ax.plot(xs, thr * xs, color="steelblue", lw=1.2,
                                linestyle="--", alpha=0.7,
                                label=f"π* = {thr:.2f}")
                        ax.fill_between(xs, xs, thr * xs,
                                        color="steelblue", alpha=0.07,
                                        label="noise band")
                    else:
                        # d = b + 2*c_n  — parallel to diagonal
                        ax.plot(xs, xs + thr, color="steelblue", lw=1.2,
                                linestyle="--", alpha=0.7,
                                label=f"2c_n = {thr:.3f}")
                        ax.fill_between(xs, xs, xs + thr,
                                        color="steelblue", alpha=0.07,
                                        label="noise band")

                ax.scatter(births[~sig], deaths[~sig], s=8,  alpha=0.4,
                           color="steelblue", label="noise")
                ax.scatter(births[sig],  deaths[sig],  s=9, alpha=0.9,
                           color="crimson", label=f"significant ({sig.sum()})", zorder=5)

                ax.set_xlabel("birth")
                ax.set_ylabel("death")
                ax.set_title(f"{mname} — persistence diagram")
                ax.set_aspect("equal")
                ax.set_xlim(0, lim)
                ax.set_ylim(0, lim)
                ax.legend(fontsize=8)
                ax_idx += 1

            if plot in ("barcode", "both"):
                ax = axes[row, ax_idx]
                order = np.argsort(pers)[::-1]
                for rank, idx in enumerate(order):
                    color = "crimson" if sig[idx] else "steelblue"
                    av = 0.9 if sig[idx] else 0.25
                    lw = 3.5 if sig[idx] else 1.0
                    ax.hlines(rank, births[idx], deaths[idx],
                              colors=color, linewidth=lw, alpha=av)

                ax.set_xlabel("filtration value epsilon")
                ax.set_ylabel("bar rank")
                ax.set_title(f"{mname} — barcode ({sig.sum()} significant)")
                ax.invert_yaxis()

        plt.tight_layout()
        plt.show()
