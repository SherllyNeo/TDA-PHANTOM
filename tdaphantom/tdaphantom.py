import numpy as np
import warnings
from tdaphantom.hypothesis_tests.universal_null_tests.universal_null_hypothesis_test import UNTest
from tdaphantom.hypothesis_tests.bottleneck_distance_tests.bottleneck_distance_test import BNTest
import matplotlib.pyplot as plt
import gudhi
from ripser import ripser
import pickle
import os


class Phantom:
    """
    Persistence diagram container.

    point_cloud : np.ndarray
        Either an (n, d) point cloud or an (n, n) distance matrix.

    is_distance_matrix : bool
        Set True when point_cloud is a distance matrix.

    dgms : dict[int, np.ndarray]
        Persistence diagrams keyed by homological dimension.
        Each value has shape (n, 2) — columns are [birth, death].
        Populated by calculate_dgms_from_point_cloud.
    """

    def __init__(
        self,
        point_cloud: np.ndarray,
        is_distance_matrix: bool = False,
    ):
        self.pc: np.ndarray = point_cloud
        self.is_dist: bool = is_distance_matrix
        self.dgms: dict[int, np.ndarray] = {}
        self._cached_results: dict = {}
        self.k: int = None

        self.allowed_methods: list[str] = [
            "universal_null",
            "universal_null:median",
            "universal_null:mean",
            "bottleneck",
            "bottleneck:subsample",
            # "bottleneck:shells",
            # "bottleneck:density",
            # "bottleneck:concentration",
        ]
        self.allowed_methods_descriptions: dict[str, str] = {
            "universal_null":           "Alias for universal_null:median.",
            "universal_null:median":    "Assumes noise follows a Gumbel distribution (Bobrowski & Skraba); uses median normalisation.",
            "universal_null:mean":      "Assumes noise follows a Gumbel distribution (Bobrowski & Skraba); uses mean normalisation.",
            "bottleneck":               "Alias for bottleneck:subsample. All bottleneck methods aim to bound the bottleneck distance between your samples diagram and the ideal hypothetical diagram using confidence intervals. This is how the hypothesis test is designed.",
            "bottleneck:subsample":     "Bootstrap confidence band via subsampling (Fasy et al.).",
            # "bottleneck:shells":        "Bottleneck test using shell decomposition.",
            # "bottleneck:density":       "Bottleneck test using density estimation.",
            # "bottleneck:concentration": "Bottleneck test using concentration inequalities.",
        }
        self.defaults = {
            "universal_null":           {"correction_strategy": "BH", "max_threshold": None, "max_depth": 1000},
            "universal_null:median":    {"correction_strategy": "BH", "max_threshold": None, "max_depth": 1000},
            "universal_null:mean":      {"correction_strategy": "BH", "max_threshold": None, "max_depth": 1000},
            "bottleneck":               {"max_depth": 50, "b_multiplier": 0.8},
            "bottleneck:subsample":     {"max_depth": 50, "b_multiplier": 0.8},
            "bottleneck:shells":        {"max_depth": 50, "b_multiplier": 0.8},
            "bottleneck:density":       {"max_depth": 50, "b_multiplier": 0.8},
            "bottleneck:concentration": {"max_depth": 50, "b_multiplier": 0.8},
        }

    def __repr__(self) -> str:
        sizes = {k: len(v) for k, v in self.dgms.items()}
        return (
            f"Phantom("
            f"n_points={len(self.pc)}, "
            f"is_distance_matrix={self.is_dist}, "
            f"computed_dims={list(self.dgms.keys())}, "
            f"diagram_sizes={sizes})"
        )

    def calculate_dgms_from_point_cloud(
        self,
        point_cloud: np.ndarray = None,
        is_distance_matrix: bool = None,
        max_dim: int = None,
        max_eps: float = None,
        k: int = None,
    ) -> dict[int, np.ndarray]:
        """
        Build a Vietoris-Rips complex and compute persistence diagrams for
        every homological dimension 0 … max_dim.

        max_dim defaults to k when provided, otherwise 1.  This lets
        you write calculate_dgms_from_point_cloud(k=2) and have exactly
        the diagrams needed for a subsequent hypothesis_test(k=2)

        point_cloud : np.ndarray, optional
            Overrides self.pc when provided.
        is_distance_matrix : bool, optional
            Overrides self.is_dist when provided.
        max_dim : int, optional
            Highest homological dimension to compute.
            Defaults to k if given, otherwise 1.
        max_eps : float, optional
            Maximum edge length for the Rips filtration (default np.inf).
        k : int, optional
            Convenience alias: sets max_dim when max_dim is not
            explicitly supplied.

        Returns:
            dict[int, np.ndarray]
                Persistence diagrams keyed by homological dimension.
        """

        if point_cloud is None:
            point_cloud = self.pc
        if is_distance_matrix is None:
            is_distance_matrix = self.is_dist
        if max_dim is None:
            max_dim = k if k is not None else 1

        # Default max_eps to the diameter of the data (matching ripser's
        # behaviour).  np.inf is intentionally avoided: gudhi will enumerate
        # every possible simplex up to max_dim+1, which is O(n^(max_dim+2))
        # and will exhaust memory on any moderately sized point cloud.
        if max_eps is None:
            if is_distance_matrix:
                max_eps = float(np.max(point_cloud))
            else:
                # Diameter via broadcasting; O(n²) memory — warn for large inputs.
                if len(point_cloud) > 5000:
                    warnings.warn(
                        f"Computing the diameter of {len(point_cloud)} points "
                        f"requires an O(n²) distance matrix. Consider passing "
                        f"max_eps explicitly to avoid this.",
                        UserWarning,
                        stacklevel=2,
                    )
                D = np.linalg.norm(
                    point_cloud[:, None, :] - point_cloud[None, :, :], axis=-1
                )
                max_eps = float(D.max())

        self.max_eps = max_eps

        self.defaults["universal_null"]["max_threshold"] = self.max_eps
        self.defaults["universal_null:median"]["max_threshold"] = self.max_eps
        self.defaults["universal_null:mean"]["max_threshold"] = self.max_eps

        if is_distance_matrix:
            rc = gudhi.RipsComplex(
                distance_matrix=point_cloud.tolist(),
                max_edge_length=max_eps,
                sparse=0.3,
            )
        else:
            rc = gudhi.RipsComplex(
                points=point_cloud.tolist(),
                max_edge_length=max_eps,
                sparse=0.3,
            )

        # create_simplex_tree needs max_dimension = max_dim + 1 to compute
        # homology up to degree max_dim
        st = rc.create_simplex_tree(max_dimension=max_dim + 1)
        st.compute_persistence()

        self.dgms = {}
        for dim in range(max_dim + 1):
            intervals = st.persistence_intervals_in_dimension(dim)
            if len(intervals) == 0:
                self.dgms[dim] = np.empty((0, 2))
            else:
                dgm = np.array(intervals, dtype=float)
                # Remove degenerate bars (numerical artefacts)
                self.dgms[dim] = dgm[dgm[:, 1] > dgm[:, 0]]

        return self.dgms

    def calculate_dgms_from_point_cloud_ripser(
        self,
        point_cloud: np.ndarray = None,
        is_distance_matrix: bool = None,
        max_dim: int = None,
        max_eps: float = None,
        k: int = None,
    ) -> dict[int, np.ndarray]:
        """
        Ripser backend for computing persistence diagrams.  Equivalent
        interface to calculate_dgms_from_point_cloud but uses ripser
        instead of gudhi's RipsComplex.

        Ripser is ignificantly faster than gudhi for Vietoris-Rips
        persistence, especially at higher dimensions, because it exploits
        the implicit representation of the Rips complex and uses
        cohomology rather than homology internally.

        point_cloud : np.ndarray, optional
            Overrides self.pc when provided.
        is_distance_matrix : bool, optional
            Overrides self.is_dist when provided.
        max_dim : int, optional
            Highest homological dimension to compute.
            Defaults to k if given, otherwise 1.
        max_eps : float, optional
            Maximum edge length / filtration threshold.
            Defaults to the diameter of the data.
        k : int, optional
            sets max_dim when max_dim is not
            explicitly supplied.

        Returns
            Persistence diagrams keyed by homological dimension,
            stored in self.dgms.
        """
        if point_cloud is None:
            point_cloud = self.pc
        if is_distance_matrix is None:
            is_distance_matrix = self.is_dist
        if max_dim is None:
            max_dim = k if k is not None else 1
        if max_eps is None:
            max_eps = np.inf  # ripser handles inf safely via its internal algorithms

        result = ripser(
            point_cloud,
            maxdim=max_dim,
            # thresh=max_eps,
            distance_matrix=is_distance_matrix,
        )

        self.dgms = {}
        for dim, dgm in enumerate(result["dgms"]):
            if len(dgm) == 0:
                self.dgms[dim] = np.empty((0, 2))
            else:
                dgm = np.array(dgm, dtype=float)
                # Remove potential degenerate bars (numerical artefacts)
                self.dgms[dim] = dgm[dgm[:, 1] > dgm[:, 0]]

        self.max_eps = max_eps
        return self.dgms

    def display_dgms(
        self,
        dgms: np.ndarray = None,
        plot: str = "both",
    ) -> None:
        """
        Plot the computed persistence diagrams.

        Each homological dimension is drawn in a distinct colour.  Call
        calculate_dgms_from_point_cloud before this method or pass in a diagram.

        plot : str
            "diagram"  — persistence diagram only (birth vs death).
            "barcode"  — barcode only (one horizontal bar per feature).
            "both"     — diagram and barcode side by side (default).
        """
        if not self.dgms:
            if dgms is None:
                raise ValueError(
                    "No persistence diagrams found. "
                    "Call calculate_dgms_from_point_cloud first."
                    "Or pass in your own diagram with the form {0: dgm_0, 1: dgm_2, ...}"
                )
        if plot not in ("diagram", "barcode", "both"):
            raise ValueError(
                f"plot must be 'diagram', 'barcode', or 'both', got {plot!r}."
            )

        dims = sorted(self.dgms.keys())
        colours = plt.cm.tab10.colors

        n_cols = 2 if plot == "both" else 1
        fig, axes = plt.subplots(
            1, n_cols, figsize=(6 * n_cols, 5), squeeze=False)

        all_finite = np.concatenate(
            [dgm[np.isfinite(dgm[:, 1])] for dgm in self.dgms.values()
             if len(dgm) > 0],
            axis=0,
        ) if any(len(d) > 0 for d in self.dgms.values()) else np.empty((0, 2))

        lim = all_finite[:, 1].max() * 1.05 if len(all_finite) else 1.0

        if plot in ("diagram", "both"):
            ax = axes[0, 0]
            ax.plot([0, lim], [0, lim], "k--", lw=0.8,
                    alpha=0.4, label="diagonal")

            for dim in dims:
                dgm = self.dgms[dim]
                if len(dgm) == 0:
                    continue
                births = dgm[:, 0]
                deaths = dgm[:, 1].copy()

                inf_mask = ~np.isfinite(deaths)
                deaths[inf_mask] = lim
                colour = colours[dim % len(colours)]
                ax.scatter(
                    births, deaths,
                    s=10, alpha=0.8, color=colour,
                    label=f"H_{dim} ({len(dgm)})",
                    zorder=3,
                )

                if inf_mask.any():
                    ax.scatter(
                        births[inf_mask], deaths[inf_mask],
                        s=30, marker="^", color=colour, zorder=4,
                    )

            ax.set_xlabel("birth")
            ax.set_ylabel("death")
            ax.set_title("Persistence diagram")
            ax.set_aspect("equal")
            ax.set_xlim(0, lim)
            ax.set_ylim(0, lim)
            ax.legend(fontsize=8)

        if plot in ("barcode", "both"):
            ax = axes[0, 1 if plot == "both" else 0]

            rank = 0
            tick_positions = []
            tick_labels = []

            for dim in dims:
                dgm = self.dgms[dim]
                if len(dgm) == 0:
                    continue
                colour = colours[dim % len(colours)]

                pers = dgm[:, 1] - dgm[:, 0]
                order = np.argsort(pers)[::-1]
                dim_start = rank

                for idx in order:
                    birth = dgm[idx, 0]
                    death = dgm[idx, 1] if np.isfinite(dgm[idx, 1]) else lim
                    ax.hlines(rank, birth, death, colors=colour,
                              linewidth=1.5, alpha=0.8)
                    rank += 1

                mid = (dim_start + rank - 1) / 2
                tick_positions.append(mid)
                tick_labels.append(f"H_{dim}")

            ax.set_xlabel("filtration value epsilon")
            ax.set_yticks(tick_positions)
            ax.set_yticklabels(tick_labels)
            ax.set_title("Barcode")
            ax.invert_yaxis()

        plt.tight_layout()
        plt.show()

    def _get_options(self, method_name: str, methods: list, options: list) -> dict:
        aliases = {
            "universal_null": "universal_null:median",
            "bottleneck":     "bottleneck:subsample",
        }
        lookup = aliases.get(method_name, method_name)
        idx = next((i for i, m in enumerate(methods)
                   if aliases.get(m, m) == lookup), None)
        user_opts = (
            options[idx]
            if idx is not None
            and options is not None
            and idx < len(options)
            and options[idx] is not None
            else {}
        )
        return {**self.defaults[method_name], **user_opts}

    def hypothesis_test(
        self,
        alpha: float = 0.05,
        methods: list[str] = None,
        options: list[dict] = None,
        k: int = 1,
    ) -> dict:
        """
        Run significance tests on the persistence diagram for dimension k.

        Requires calculate_dgms_from_point_cloud to have been called with
        max_dim >= k (or equivalently k >= k).

        alpha : float
            Significance level in (0, 1).  Default 0.05.
        methods : list[str], optional
            One or more of self.allowed_methods.
            Defaults to ["universal_null", "bottleneck"].
        options : list[dict], optional
            Per-method advanced options. Each entry corresponds to the method
            at the same index in methods. Pass None for a method to use defaults.
        k : int
            Homological dimension to test.  Default 1.
        """
        if not isinstance(k, int) or k < 0:
            raise TypeError(
                f"k must be a non-negative integer homological dimension, got {k!r}."
            )
        if not isinstance(alpha, float):
            raise TypeError(
                f"alpha must be a float, got {type(alpha).__name__}."
            )
        if alpha <= 0 or alpha >= 1:
            raise ValueError(
                f"alpha must be strictly between 0 and 1, got {alpha}."
            )

        if methods is None:
            methods = ["universal_null", "bottleneck"]

        invalid = [m for m in methods if m not in self.allowed_methods]
        if invalid:
            raise ValueError(
                f"Unknown method(s): {invalid}. "
                f"Allowed methods: {self.allowed_methods}."
            )

        if k not in self.dgms:
            raise ValueError(
                f"No persistence diagram found for dimension k={k}. "
                f"Call calculate_dgms_from_point_cloud(k={k}) first."
            )

        self.k = k
        dgm_k = self.dgms[k]
        results = {}

        if "universal_null" in methods or "universal_null:median" in methods:
            mname = "universal_null:median" if "universal_null:median" in methods else "universal_null"
            opts = self._get_options(mname, methods, options)
            test = UNTest(
                dgm=dgm_k,
                k=k,
                alpha=alpha,
                method="universal_null:median",
                options=opts,
            )
            results["universal_null:median"] = test.results()

        if "universal_null:mean" in methods:
            opts = self._get_options("universal_null:mean", methods, options)
            test = UNTest(
                dgm=dgm_k,
                k=k,
                alpha=alpha,
                method="universal_null:mean",
                options=opts,
            )
            results["universal_null:mean"] = test.results()

        if "bottleneck" in methods or "bottleneck:subsample" in methods:
            mname = "bottleneck:subsample" if "bottleneck:subsample" in methods else "bottleneck"
            opts = self._get_options(mname, methods, options)
            test = BNTest(
                point_cloud=self.pc,
                dgm=dgm_k,
                alpha=alpha,
                method="bottleneck:subsample",
                is_distance_matrix=self.is_dist,
                options=opts,
            )
            results["bottleneck:subsample"] = test.results()

        if "bottleneck:shells" in methods:
            opts = self._get_options("bottleneck:shells", methods, options)
            test = BNTest(
                point_cloud=self.pc,
                dgm=dgm_k,
                alpha=alpha,
                method="bottleneck:shells",
                is_distance_matrix=self.is_dist,
                options=opts,
            )
            results["bottleneck:shells"] = test.results()

        if "bottleneck:density" in methods:
            opts = self._get_options("bottleneck:density", methods, options)
            test = BNTest(
                point_cloud=self.pc,
                dgm=dgm_k,
                alpha=alpha,
                method="bottleneck:density",
                is_distance_matrix=self.is_dist,
                options=opts,
            )
            results["bottleneck:density"] = test.results()

        if "bottleneck:concentration" in methods:
            opts = self._get_options(
                "bottleneck:concentration", methods, options)
            test = BNTest(
                point_cloud=self.pc,
                dgm=dgm_k,
                alpha=alpha,
                method="bottleneck:concentration",
                is_distance_matrix=self.is_dist,
                options=opts,
            )
            results["bottleneck:concentration"] = test.results()

        self._cached_results = results
        return results

    def display_results(
        self,
        results: dict = None,
        method: str = "all",
        plot: str = "both",
    ) -> None:
        """
        Visualise hypothesis test results.

        results : dict, optional
            Output of hypothesis_test.  Uses the most recent cached run
            when omitted.
        method : str
            Which method to plot, or "all" for every method in results.
        plot : str
            "diagram", "barcode", or "both".
        """
        if results is None:
            if not self._cached_results:
                raise ValueError(
                    "No results to display. Pass a results dict or call "
                    "hypothesis_test first."
                )
            results = self._cached_results

        if not isinstance(results, dict):
            raise ValueError(
                "results must be a dict returned by hypothesis_test.")

        if plot not in ("diagram", "barcode", "both"):
            raise ValueError(
                f"plot must be 'diagram', 'barcode', or 'both', got {plot!r}."
            )

        available = list(results.keys())
        if method == "all":
            methods_to_plot = available
        else:
            if method not in available:
                raise ValueError(
                    f"method {method!r} not found in results. Available: {available}."
                )
            methods_to_plot = [method]

        n_cols = 2 if plot == "both" else 1
        n_rows = len(methods_to_plot)

        fig, axes = plt.subplots(
            n_rows, n_cols,
            figsize=(6 * n_cols, 5 * n_rows),
            squeeze=False,
        )
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
                finite_deaths = deaths[np.isfinite(deaths)]
                lim = finite_deaths.max() * 1.05 if len(finite_deaths) else 1.0
                xs = np.linspace(0, lim, 300)

                ax.plot([0, lim], [0, lim], "k--", lw=0.8,
                        alpha=0.4, label="diagonal")

                if not np.isnan(thr):
                    if np.isinf(thr):
                        # inf threshold = everything is noise: shade the entire upper triangle
                        ax.fill_between(xs, xs, lim,
                                        color="steelblue", alpha=0.07, label="noise band (all)")
                    elif "universal_null" in mname:
                        # Threshold is a multiplicative ratio: death = (pi*)(birth)
                        ax.plot(xs, thr * xs, color="steelblue", lw=1.2,
                                linestyle="--", alpha=0.7, label=f"pi_min = {thr:.2f}")
                        ax.fill_between(xs, xs, thr * xs,
                                        color="steelblue", alpha=0.07, label="noise band")
                    else:
                        # Threshold is an additive offset: death = birth + 2c_n
                        ax.plot(xs, xs + thr, color="steelblue", lw=1.2,
                                linestyle="--", alpha=0.7, label=f"2c_n = {thr:.3f}")
                        ax.fill_between(xs, xs, xs + thr,
                                        color="steelblue", alpha=0.07, label="noise band")

                ax.scatter(births[~sig], deaths[~sig], s=8,  alpha=0.4,
                           color="steelblue", label="noise")
                ax.scatter(births[sig],  deaths[sig],  s=9,  alpha=0.9,
                           color="crimson", label=f"significant ({sig.sum()})", zorder=5)

                ax.set_xlabel("birth")
                ax.set_ylabel("death")
                ax.set_title(
                    f"{mname} — significance persistence diagram ({sig.sum()} significant)")
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
                    alpha_val = 0.9 if sig[idx] else 0.25
                    lw = 3.5 if sig[idx] else 1.0
                    ax.hlines(rank, births[idx], deaths[idx],
                              colors=color, linewidth=lw, alpha=alpha_val)

                ax.set_xlabel("filtration value epsilon")
                ax.set_ylabel("bar rank")
                ax.set_title(
                    f"{mname} — significance barcode ({sig.sum()} significant)")
                ax.invert_yaxis()

        plt.tight_layout()
        plt.show()

    def save(self, path: str) -> None:
        """
        Save the Phantom instance to disk using pickle.
        """
        os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
        with open(path, "wb") as f:
            pickle.dump(self, f)
        print(f"Saved Phantom to {path}")

    @classmethod
    def load(cls, path: str) -> "Phantom":
        """
        Load a Phantom instance from disk.
        """
        if not os.path.exists(path):
            raise FileNotFoundError(f"No Phantom file found at {path!r}.")
        with open(path, "rb") as f:
            obj = pickle.load(f)
        if not isinstance(obj, cls):
            raise TypeError(
                f"Loaded object is {type(obj).__name__}, expected Phantom."
            )
        print(f"Loaded Phantom from {path}")
        return obj
