# TDA-PHANTOM
Topological data analysis -
Persistent Homology Analysis via Null Testing On Manifolds (TDA-PHANTOM)
is a tool for statistically analysing significance of persistence diagrams and barcodes.

This project implements hypothesis tests from:
- *Confidence Sets for Persistence Diagrams*
  Fasy et al. (2014)
  DOI: https://doi.org/10.1214/14-AOS1252
- *A Universal Null-Distribution for Topological Data Analysis*
  Bobrowski and Skraba (2023)
  DOI: https://doi.org/10.1038/s41598-023-37842-2

## Installation
Via [PyPI](https://pypi.org/project/tdaphantom/):
```bash
pip install tdaphantom
```
Or you can clone this repository -  [TDA-PHANTOM](https://github.com/SherllyNeo/TDA-PHANTOM) and install it manually:
```bash
python setup.py install
```

## Overview
This tool can build a Vietoris-Rips complex from either a point cloud or distance matrix.
It can then be used to visualise the persistence diagram for that complex, and run various hypothesis tests for it.
The results of these hypothesis tests can be analysed via a return results array, or visualised in a signifiance persistence diagram.

## Example Usage
### Generate data
```python
def _make_circle(n=2000, noise=0.03, seed=1):
    rng   = np.random.default_rng(seed)
    theta = rng.uniform(0, 2 * np.pi, n)
    pts   = np.stack([np.cos(theta), np.sin(theta)], axis=1)
    return pts + rng.normal(0, noise, pts.shape)
pc_circle = _make_circle()
```

### Init Phantom class
```python
phantom_circle = Phantom(pc_circle)
```

### Calculate persistence diagram
Here we go up to homological dimension 1
```python
phantom_circle.calculate_dgms_from_point_cloud_ripser(k=1)
```

### Display persistence diagram
```python
phantom_circle.display_dgms()
```
![Persistence diagram for a circle using phatom](![Persistence diagram](https://raw.githubusercontent.com/SherllyNeo/TDA-PHANTOM/main/README_SRC/circle_dgms.png))

### Run hypothesis test
```python
alpha      = 0.01
correction = None
methods    = ["universal_null"]
phantom_circle.hypothesis_test(alpha, methods=methods, k=1)
```

### Display signifiance persistence diagram
```python
phantom_circle.display_results()
```
![signifiance Persistence diagram for a circle using phatom](![Persistence diagram](https://raw.githubusercontent.com/SherllyNeo/TDA-PHANTOM/main/README_SRC/circle_sig.png))

## Sample size

Each hypothesis method requires a specific minimum sample size to be reliable, in terms of the amount of points in the point cloud and potentially the amount of points in the persistence diagram/bars in the barcode.

**This is checked by each hypothesis test automatically when a hypothesis test is ran.**

### Universal null sample size

TODO

### Bottleneck subsampling sample size

TODO

### Bottleneck shells sample size

TODO

### Bottleneck density sample size

TODO

### Bottleneck concentration sample size

TODO


## Basic useage
The general workflow is:

### 1. Initalise Phantom
Initalise with either a point cloud or distance matrix. If using a distance matrix, set `is_distance_matrix=True`.

```python
from tdaphantom.tdaphantom import Phantom

# from a point cloud
phantom = Phantom(point_cloud)

# from a distance matrix
phantom = Phantom(distance_matrix, is_distance_matrix=True)
```

### 2. Generate persistence diagrams
Two backends are available — GUDHI and Ripser. Ripser is faster for Vietoris-Rips persistence.
Both take `k` as a convenience argument to set the maximum homological dimension to compute.

This will work even if you have inputted a distance matrix

```python
# gudhi backend
phantom.calculate_dgms_from_point_cloud(k=1)

# ripser backend
phantom.calculate_dgms_from_point_cloud_ripser(k=1)
```

### 3. Display persistence diagrams
Call `display_dgms` to plot the persistence diagram and/or barcode.
Each homological dimension is shown in a different colour.
You may also pass in a pre-made diagram dict of the form `{0: dgm_0, 1: dgm_1, ...}`.

```python
phantom.display_dgms()                  # both diagram and barcode
phantom.display_dgms(plot="diagram")    # persistence diagram only
phantom.display_dgms(plot="barcode")    # barcode only
```

### 4. Run hypothesis tests
Run one or more hypothesis tests on the persistence diagram for a given dimension `k`.
You may optionally pass per-method options as a list of dicts aligned to the methods list.
Note: bottleneck methods require the point cloud or distance matrix to be available.

The below code snippet shows the default values

```python
results = phantom.hypothesis_test(
    alpha   = 0.05,
    methods = ["universal_null", "bottleneck"],
    k = 1,
)
```

With custom options:

```python
results = phantom.hypothesis_test(
    alpha   = 0.05,
    methods = ["universal_null", "bottleneck"],
    options = [
        {"correction_strategy": "Bonferroni"},   # options for universal_null
        {"max_depth": 100, "b_multiplier": 3.5}, # options for bottleneck
    ],
    k = 1,
)
```

### 5. Display results
Call `display_results` to visualise which bars are significant.
Significant bars/points are shown in red, noise in blue.
A threshold line is drawn on the signifiance persistence diagram showing the significance boundary.

```python
phantom.display_results()                        # all methods, both plots
phantom.display_results(method="universal_null") # specific method
phantom.display_results(plot="diagram")          # signifiance persistence diagram only
```

### 6. Save and load
At any point you can save the full state of Phantom to disk and reload it later.

```python
phantom.save("./saved/circle.phantom")
phantom = Phantom.load("./saved/circle.phantom")
```

## Avaliable hypothesis methods

| Method | Alias | Requires |
|---|---|---|
| `universal_null:median` | `universal_null` | diagram only |
| `universal_null:mean` | — | diagram only |
| `bottleneck:subsample` | `bottleneck` | point cloud or distance matrix |
| `bottleneck:subsample_kdtree` | `bottleneck` | point cloud or distance matrix |

## Universal null median

### Useage


```python
phantom.hypothesis_test(alpha=0.05, methods=["universal_null"], k=1)
```

With options:
```python
phantom.hypothesis_test(
    alpha   = 0.05,
    methods = ["universal_null"],
    options = [{"correction_strategy": "Bonferroni", "max_threshold": 10.0}],
    k       = 1,
)
```

Available options:

| Option | Default | Description |
|---|---|---|
| `correction_strategy` | `None` | Multiple testing correction. `"Bonferroni"` or `"BH"` for Benjamini hochberg or `None`. |
| `max_threshold` | 10 * max finite death| Threshold substituted for infinite death values. |
| `max_depth` | `1000` | Maximum iterations for the infinite cycle threshold algorithm. |

### Theory
Bobrowski and Skraba (2023) show that the death/birth ratio $\pi = \text{death}/\text{birth}$ for noise bars
follows a universal limiting distribution — the left-skewed Gumbel (LGumbel) — regardless
of the underlying point cloud distribution or geometry.

The normalised statistic for each bar is:

```math
\ell(p) = A \cdot \log \log \pi(p) + B
```

where:

```math
\pi(p) = \frac{\text{death}}{\text{birth}}, \quad
A = 1 \text{ (Vietoris-Rips)}, \quad
B = -\gamma - A \cdot \hat{L}, \quad
\hat{L} = \text{median}(\log \log \pi) \text{ over all finite bars}, \quad
\gamma \approx 0.5772  \text{ (Euler-Mascheroni constant)},
```

Under the null hypothesis (noise), $\ell(p)$ follows $\text{LGumbel}(0,1)$.
The p-value for each bar is the LGumbel survival function:

```math
p_i = \exp(-\exp(\ell_i))
```

Significance is assessed after Bonferroni or BH correction across all bars.
The threshold line on the persistence diagram is $\text{death} = \text{birth} \cdot \pi^*$ where $\pi^*$ is the
minimum $\pi$ such that a bar would be rejected at the corrected alpha level.

Infinite bars are handled with Bobrowski and Skraba (2023) algorithm 1
```
τ ← τ₀
do
    D ← dgmₖ(τ)
    I ← { b : (b, d) ∈ D, d = ∞, and τ/b < π_min(α/|D|) }
    τ ← τ_new
while |I| > 0
return τ
```

where:

```math
\tau_{\text{new}} = \begin{cases} \min(I) \cdot \pi_{\min}(\alpha / |D|) & I \neq \emptyset \\ \tau & I = \emptyset \end{cases}
```

where $\tau_0$ is the initial threshold, $\text{dgm}_k(\tau)$ is the persistence diagram truncated at $\tau$,
and $\pi_{\min}(\alpha/|D|)$ is the minimum death/birth ratio such that a bar is significant at the Bonferroni-corrected level $\alpha/|D|$.


## Universal null mean

### Useage
```python
phantom.hypothesis_test(alpha=0.05, methods=["universal_null:mean"], k=1)
```

### Theory
Identical to universal null median, but $\hat{L}$ is estimated using the mean of $\log \log \pi$
rather than the median. The paper's exact formula uses the mean. The median is more
robust to contamination from signal bars, but the mean is theoretically motivated by
the paper's universality result.

## Bottleneck subsampling

### Useage
```python
phantom.hypothesis_test(alpha=0.05, methods=["bottleneck"], k=1)
```

With options:
```python
phantom.hypothesis_test(
    alpha   = 0.05,
    methods = ["bottleneck"],
    options = [{"max_depth": 100, "b_multiplier": 3.5}],
    k       = 1,
)
```

Available options:

| Option | Default | Description |
|---|---|---|
| `max_depth` | `50` | Number of bootstrap subsamples N. |
| `b_multiplier` | `3.5` | The constant used for $O(\frac{n}{\log(n)})$. Larger values give smaller c_n and more power but looser theoretical guarantees. |

### Theory
Fasy et al. (2014) 4.1 derive a confidence set for the persistence diagram of the form:

```math
\mathcal{C} = \{ Q : W_\infty(\hat{P}, P) \leq c_n \}
```

where $W_\infty$ is the bottleneck distance and $c_n$ is estimated by subsampling.

For each of $N$ subsamples $S^*_b$ of size $b$ drawn without replacement from $S_n$:

```math
T_j = H(S_n, S^*_b) \quad \text{(Hausdorff distance between point sets)}
```

```math
c_n = \text{quantile}(\{T_j\},\, 1 - \alpha)
```

By the bottleneck stability theorem, $W_\infty(\hat{P}, P) \leq H(S_n, M)$, so bars with
persistence $> 2 c_n$ are significant at level $\alpha$.

The p-value for bar $i$ is the fraction of bootstrap distances exceeding half its persistence:

```math
p_i = \frac{1}{N} \sum_j \mathbf{1}\!\left(T_j \geq \frac{\ell_i}{2}\right)
```

The error bound from the paper is:

```math
P(H(S_n, M) > c_n) \leq \alpha + O\!\left(\left(\frac{b}{n}\right)^{1/4}\right)
```

so the bias term shrinks as $b/n \to 0$. In practice, larger $b$ gives more power at the
cost of strict theoretical guarantees on the type I error. The default
```python
b = int(3.5*(n / np.log(n)))
```
is a practical choice.

## Bottleneck subsampling kdtree

### Useage
```python
phantom.hypothesis_test(alpha=0.05, methods=["bottleneck:subsampling_kdtree"], k=1)
```

With options:
```python
phantom.hypothesis_test(
    alpha   = 0.05,
    methods = ["bottleneck:subsampling_kdtree"],
    options = [{"max_depth": 100}],
    k       = 1,
)
```

Available options:

| Option | Default | Description |
|---|---|---|
| `max_depth` | `50` | Number of bootstrap subsamples N. |

### Theory
Fasy et al. (2014) 4.1 derive a confidence set for the persistence diagram of the form:

```math
\mathcal{C} = \{ Q : W_\infty(\hat{P}, P) \leq c_n \}
```

where $W_\infty$ is the bottleneck distance and $c_n$ is estimated by subsampling.

For each of $N$ subsamples $S^*_b$ of size $b$ drawn without replacement from $S_n$:

```math
T_j = H(S_n, S^*_b) \quad \text{(Hausdorff distance between point sets)}
```

```math
c_n = \text{quantile}(\{T_j\},\, 1 - \alpha)
```

By the bottleneck stability theorem, $W_\infty(\hat{P}, P) \leq H(S_n, M)$, so bars with
persistence $> 2 c_n$ are significant at level $\alpha$.

The p-value for bar $i$ is the fraction of bootstrap distances exceeding half its persistence:

```math
p_i = \frac{1}{N} \sum_j \mathbf{1}\!\left(T_j \geq \frac{\ell_i}{2}\right)
```

The error bound from the paper is:

```math
P(H(S_n, M) > c_n) \leq \alpha + O\!\left(\left(\frac{b}{n}\right)^{1/4}\right)
```

so the bias term shrinks as $b/n \to 0$. In practice, larger $b$ gives more power at the
cost of strict theoretical guarantees on the type I error. The default
```python
b = int(n / np.log(n))
```
is enforced for a clean theoretical guarantee

## Bottleneck shells

### Useage

> **TODO**: not yet implemented.

```python
# phantom.hypothesis_test(alpha=0.05, methods=["bottleneck:shells"], k=1)
```

Available options:

| Option | Default | Description |
|---|---|---|
| `max_depth` | `50` | Number of bootstrap subsamples N. |


### Theory
> **TODO**: not yet implemented.

Fasy et al. (2014) 4.3
Note: the method of shells requires that $P$ satisfies a uniform lower bound on local
density within each shell. It does not apply to distributions with arbitrarily sparse regions.

## Bottleneck density

### Useage

> **TODO**: not yet implemented.

```python
# phantom.hypothesis_test(alpha=0.05, methods=["bottleneck:density"], k=1)
```

Available options:

| Option | Default | Description |
|---|---|---|
| `max_depth` | `50` | Number of bootstrap subsamples N. |
| `bandwidth` | `None` | KDE bandwidth h. Estimated from data if not supplied. |

### Theory
> **TODO**: not yet implemented.


Fasy et al. (2014) 4.4 estimates the underlying density $p$ using a kernel
density estimator $\hat{p}_h$ and computes the persistence diagram of the upper level
sets of $\hat{p}_h$.

Let $\hat{p}_h$ be a KDE with bandwidth $h$:



The density method is robust to outliers because the KDE 'smooths them out',
unlike the distance-function methods which are sensitive to isolated points far from $M$.

## Bottleneck concentration

### Useage

> **TODO**: not yet implemented.

```python
# phantom.hypothesis_test(alpha=0.05, methods=["bottleneck:concentration"], k=1)
```

Available options:

| Option | Default | Description |
|---|---|---|
| `max_depth` | `50` | Number of bootstrap subsamples N. |
| `rho` | `None` | Minimum local density of P on M. Estimated from data if not supplied. |
| `d` | `None` | Intrinsic dimension of M. Inferred from point cloud if not supplied. |

### Theory
> **TODO**: not yet implemented.
Fasy et al. (2014) 4.2
## TODO
* Add bottleneck shells
* Add bottleneck density
* Add bottleneck concentration
* Add more integeration tests
* Add more unit tests
