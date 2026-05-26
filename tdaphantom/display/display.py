import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns


def is_interactive(self):
    '''Returns true if running an interactive/notebook session'''
    import __main__ as main
    return not hasattr(main, '__file__')


class DisplayPersistenceDiagram:
    def __init__(self, point_cloud: np.ndarray, is_distance_matrix: bool = False, dgms: np.ndarray = None, plot: str = "both"):
        self.point_cloud = point_cloud
        self.is_distance_matrix = is_distance_matrix
        self.dgms = dgms
        self.plot = plot


class DisplaySignificancePersistenceDiagram:
    def __init__(self, results: dict = None, method: str = "all", plot: str = "both", k: int = 1, save_fpath: str = None):
        self.results = results
        self.method = method
        self.plot = plot
        self.k = k
        self.label_size = 14
        if not save_fpath:
            self.save_fpath = f"Hypothesis_test_results_H{self.k}.png"
        else:
            self.save_fpath = save_fpath

    def main(self):
        '''
        Plots persistence diagrams on hypothesis test output
        Users may choose to plot either/both persistence diagrams and/or barcodes.
        '''

        '''Count n types of analysis, informs which plots to make'''
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

        '''Detect if run from notebook to determine if we're .show()ing or .plot()ting.'''
        is_notebook = is_interactive()

        '''Initialise subplots'''
        sns.set_theme(rc={"xtick.labelsize": self.label_size, "ytick.labelsize": self.label_size,
                      "axes.labelsize": self.label_size}, style="whitegrid")
        fig, axes = plt.subplots(
            n_rows, n_cols,
            figsize=(6.5 * n_cols, 6 * n_rows),
            squeeze=False,
        )
        fig.suptitle(f"H$_{self.k}$ persistence results",
                     fontsize=self.label_size+2)

        for row, mname in enumerate(methods_to_plot):
            fig.supylabel(f"{mname}", fontsize=self.label_size)
            res = self.results[mname]
            results_array = res["results_array"]
            thr = res.get("threshold", np.nan)

            births = results_array[:, 0]
            deaths = results_array[:, 1]
            pers = deaths - births
            sig = results_array[:, 4].astype(bool)
            ax_idx = 0

            if self.plot in ("diagram", "both"):
                '''Plot persistence diagram with threshold and significant points highlighted'''
                ax = axes[row, ax_idx]
                finite_deaths = deaths[np.isfinite(deaths)]
                lim = finite_deaths.max() * 1.05 if len(finite_deaths) else 1.0
                xs = np.linspace(0, lim, 300)

                ax.plot([0, lim], [0, lim], "k--", lw=0.8,
                        alpha=0.4, label="Diagonal")

                if not np.isnan(thr):
                    if np.isinf(thr):
                        '''inf threshold = everything is noise: shade the entire upper triangle'''
                        ax.fill_between(xs, xs, lim,
                                        color="steelblue", alpha=0.07, label="Noise band (all)")
                    elif "universal_null" in mname:
                        '''Threshold is a multiplicative ratio: death = (pi*)(birth)'''
                        ax.plot(xs, thr * xs, color="steelblue", lw=1.2,
                                linestyle="--", alpha=0.7, label=f"\u03C0$_{{min}}$ = {thr:.2f}")
                        ax.fill_between(xs, xs, thr * xs,
                                        color="steelblue", alpha=0.07, label="Noise band")
                    else:
                        '''Threshold is an additive offset: death = birth + 2c_n'''
                        ax.plot(xs, xs + thr, color="steelblue", lw=1.2,
                                linestyle="--", alpha=0.7, label=f"2c_n = {thr:.3f}")
                        ax.fill_between(xs, xs, xs + thr,
                                        color="steelblue", alpha=0.07, label="Noise band")

                ax.scatter(births[~sig], deaths[~sig], s=8,  alpha=0.4,
                           color="steelblue", label="Noise")
                ax.scatter(births[sig],  deaths[sig],  s=9,  alpha=0.9,
                           color="crimson", label=f"Significant ({sig.sum()})", zorder=5)

                ax.set_xlabel("Birth", fontsize=self.label_size)
                ax.set_ylabel("Death", fontsize=self.label_size)
                ax.set_title(
                    f"Significance persistence diagram", fontsize=self.label_size)
                ax.set_aspect("equal")
                ax.set_xlim(0, lim)
                ax.set_ylim(0, lim)
                ax.legend(fontsize=self.label_size)
                ax_idx += 1

            if self.plot in ("barcode", "both"):
                '''If user selects, plot barcode to RHS'''
                ax = axes[row, ax_idx]
                order = np.argsort(pers)[::-1]

                for rank, idx in enumerate(order):
                    color = "crimson" if sig[idx] else "steelblue"
                    alpha_val = 0.9 if sig[idx] else 0.25
                    lw = 1.0 if sig[idx] else 1.0
                    ax.hlines(rank, births[idx], deaths[idx],
                              colors=color, linewidth=lw, alpha=alpha_val)

                ax.set_xlabel("Filtration value (\u03B5)",
                              fontsize=self.label_size)
                ax.set_ylabel("Bar rank", fontsize=self.label_size)
                ax.set_title(
                    f"Significance barcode", fontsize=self.label_size)
                ax.invert_yaxis()

        plt.tight_layout()
        if is_notebook:
            plt.show()

        else:
            plt.savefig(f"Hypothesis_test_results_H{self.k}.png", dpi=300)
