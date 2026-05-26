import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

class DisplayPersistenceDiagram:
    def __init__(self,point_cloud: np.ndarray, is_distance_matrix: bool = False, dgms: np.ndarray = None, plot: str = "both"):
        self.point_cloud = point_cloud
        self.is_distance_matrix = is_distance_matrix
        self.dgms = dgms 
        self.plot = plot

class DisplaySignificancePersistenceDiagram:
    def __init__(self,results: dict = None, method: str = "all", plot: str = "both"):
        self.results = results
        self.method = method
        self.plot = plot

    def main(self):
        '''INFORMATIVE DOCSTRING'''
    
        ''''''
        available = list(self.results.keys())
        if self.method == "all":
            methods_to_plot = available
        else:
            if self.method not in available:
                raise ValueError(
                    f"method {self.method!r} not found in results. Available: {available}."
                )
            methods_to_plot = [self.method]
     
        n_cols = 2 if self.plot == "both" else 1
        n_rows = len(methods_to_plot)

        fig, axes = plt.subplots(
            n_rows, n_cols,
            figsize=(6 * n_cols, 5 * n_rows),
            squeeze=False,
        )
        fig.suptitle(f"H_{self.k} persistence results", fontsize=14)

        for row, mname in enumerate(methods_to_plot):
            res = self.results[mname]
            results_array = res["results_array"]
            thr = res.get("threshold", np.nan)

            births = results_array[:, 0]
            deaths = results_array[:, 1]
            pers = deaths - births
            sig = results_array[:, 4].astype(bool)
            ax_idx = 0

            if self.plot in ("diagram", "both"):
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

            if self.plot in ("barcode", "both"):
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

        breakpoint()