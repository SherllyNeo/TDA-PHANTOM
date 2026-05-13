# TDA-PHANTOM
Topological data analysis - Persistent Homology Analysis via Null Testing On Manifolds (TDA-PHANTOM) is a tool for statistically analysing significance of persistence diagrams and barcodes.


## Quickstart


```{python}
from ripser import ripser
from tdaphamtom import Phantom


def _make_circle(n=2000, noise=0.03, seed=1):
    rng   = np.random.default_rng(seed)
    theta = rng.uniform(0, 2 * np.pi, n)
    pts   = np.stack([np.cos(theta), np.sin(theta)], axis=1)
    return pts + rng.normal(0, noise, pts.shape)

from scipy.spatial.distance import cdist

def _ripser_h1(pts):
    D   = cdist(pts, pts)
    dgm = ripser(D, distance_matrix=True, maxdim=1)["dgms"][1]
    # ripser may return bars with birth == death, strip them
    return dgm[dgm[:, 1] > dgm[:, 0]]

pts_circle = _make_circle()

dgm_circle = _ripser_h1(pts_circle)

phantom_circle = Phantom(dgm_circle, k=1)

results_circle = phantom_circle.hypothesis_test(alpha,correction_method=correction)

phantom_circle.display_results()
```

