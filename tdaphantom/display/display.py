import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns


def is_interactive():
    '''Returns true if running an interactive/notebook session'''
    import __main__ as main
    return not hasattr(main, '__file__')


def get_rcparams(label_size=14, n_cols=1, n_rows=1):
    '''Set RC params for consistent plot styling across modules'''
    plt_params = ["xtick.labelsize", "ytick.labelsize",
                  "axes.labelsize", "axes.titlesize", "axes.labelsize",
                  "figure.labelsize",
                  "legend.fontsize"]
    plt_rcparams = {i: label_size for i in plt_params}
    plt_rcparams["figure.figsize"] = (6 * n_cols, 6 * n_rows)
    plt_rcparams["figure.labelsize"] = label_size + 2
    return plt_rcparams


class DisplayPersistenceDiagram:
    def __init__(self, point_cloud: np.ndarray, is_distance_matrix: bool = False, dgms: np.ndarray = None, plot: str = "both", save_fpath: str = None):
        self.point_cloud = point_cloud
        self.is_distance_matrix = is_distance_matrix
        self.dgms = dgms
        self.plot = plot
        self.label_size = 14
        if not save_fpath:
            self.save_fpath = f"Persistence_diagram_H{self.k}.png"
        else:
            self.save_fpath = save_fpath

    def main(self):
        '''
        Draw persistence diagram and/or barcode.
        Users may choose to plot either/both persistence diagrams and/or barcodes.
        Plots will either .show() in ipynb or be saved when called as a module.
        '''
        dims = sorted(self.dgms.keys())
        colours = plt.cm.tab10.colors

        n_cols = 2 if self.plot == "both" else 1
        sns.set_theme(rc=get_rcparams(label_size=self.label_size,
                      n_cols=n_cols, n_rows=1), style="whitegrid")

        fig, axes = plt.subplots(
            1, n_cols, squeeze=False)
        fig.suptitle(f"Persistence diagrams",
                     fontsize=self.label_size+2)

        all_finite = np.concatenate(
            [dgm[np.isfinite(dgm[:, 1])] for dgm in self.dgms.values()
             if len(dgm) > 0],
            axis=0,
        ) if any(len(d) > 0 for d in self.dgms.values()) else np.empty((0, 2))

        lim = all_finite[:, 1].max() * 1.05 if len(all_finite) else 1.0

        if self.plot in ("diagram", "both"):
            ax = axes[0, 0]
            ax.plot([0, lim], [0, lim], "k--", lw=0.8,
                    alpha=0.4, label="Diagonal")

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
                    label=f"H$_{dim}$ ({len(dgm)})",
                    zorder=3,
                )

                if inf_mask.any():
                    ax.scatter(
                        births[inf_mask], deaths[inf_mask],
                        s=30, marker="^", color=colour, zorder=4,
                    )

            ax.set_xlabel("Birth")
            ax.set_ylabel("Death")
            ax.set_title("Persistence diagram", )
            ax.set_aspect("equal")
            ax.set_xlim(0, lim)
            ax.set_ylim(0, lim)
            ax.legend(fontsize=self.label_size)

        if self.plot in ("barcode", "both"):
            ax = axes[0, 1 if self.plot == "both" else 0]

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
                tick_labels.append(f"H$_{dim}$")

            ax.set_xlabel("Filtration value (\u03B5)")
            ax.set_yticks(tick_positions)
            ax.set_yticklabels(tick_labels,)
            ax.set_title("Barcode")
            ax.invert_yaxis()

        plt.tight_layout()
        if is_interactive():
            plt.show()

        else:
            plt.savefig(self.save_fpath, dpi=300)


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
        Plots will either .show() in ipynb or be saved when called as a module.
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

        '''Initialise subplots'''
        sns.set_theme(rc=get_rcparams(self.label_size,
                      n_cols, n_rows), style="whitegrid")
        fig, axes = plt.subplots(
            n_rows, n_cols,
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

                ax.set_xlabel("Birth")
                ax.set_ylabel("Death")
                ax.set_title(
                    f"Significance persistence diagram")
                ax.set_aspect("equal")
                ax.set_xlim(0, lim)
                ax.set_ylim(0, lim)
                ax.legend()
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

                ax.set_xlabel("Filtration value (\u03B5)")
                ax.set_ylabel("Bar rank")
                ax.set_title(
                    f"Significance barcode")
                ax.invert_yaxis()

        plt.tight_layout()
        if is_interactive():
            plt.show()

        else:
            plt.savefig(self.save_fpath, dpi=300)
