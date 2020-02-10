# ahoi (A Horrible Optimisation Instrument)

This module contains a few python functions to run Brute-force scans for rectangular cut optimization.

# Installation

To install ahoi run

```sh
python3 -m pip install [--user] ahoi
```

Use `--user` if not in a virtual environment or conda environment.

It's recommended to use python3, but currently python2 is also supported.

# Example
The basic functionality uses a `masks_list` which is a list of lists or a list
of 2D numpy arrays that represent pass flags for selection criteria.

For example, the following represents pass flags for the criteria `>0`, `>0.1`,
`>0.2`, ..., `>0.9` for 5 random uniform variables in 10000 events:

```python
import numpy as np
np.random.seed(42)
x = np.random.rand(10000, 5)
masks_list = [[x[:,i] > v for v in np.linspace(0, 0.9, 10)] for i in range(x.shape[1])]
```

To count all matching combinations for all criteria on each variable run

```
import ahoi
counts = ahoi.scan(masks_list)
```

The entry `[0, 1, 2, 3, 4]` of `counts` will contain the number of matching
events where the first column of `x` is `>0`, the second one `>0.1`, the third
one `>0.2` etc.

```python
>>> counts[0, 1, 2, 3, 4]
3032
>>> np.count_nonzero((x[:,0] > 0) & (x[:,1] > 0.1) & (x[:,2] > 0.2) & (x[:,3] > 0.3) & (x[:,4] > 0.4))
3032
```

You can also pass weights

```python
weights = np.random.normal(loc=1, size=len(x))
counts, sumw, sumw2 = ahoi.scan(masks_list, weights=weights)
```

The arrays `sumw` and `sumw2` will contain the sum of weights and sum of squares
of weights for matching combinations. The sum of squares of weights can be used
to estimate the statistical uncertainty on the sum of weights ($`\sigma = \sqrt{\sum w_i^2}`$).

```python
>>> sumw[0, 1, 2, 3, 4]
3094.2191136427627
>>> np.dot(
...     (x[:,0] > 0) & (x[:,1] > 0.1) & (x[:,2] > 0.2) & (x[:,3] > 0.3) & (x[:,4] > 0.4),
...     weights
... )
3094.219113642755
>>> np.sqrt(sumw2[0, 1, 2, 3, 4])
78.5528532026876
>>> np.sqrt(
...     np.dot(
...         (x[:,0] > 0) & (x[:,1] > 0.1) & (x[:,2] > 0.2) & (x[:,3] > 0.3) & (x[:,4] > 0.4),
...         weights ** 2
...     )
... )
78.55285320268761
```

# Tests/Coverage

Run the tests and coverage report inside the project directory with

```sh
python3 -m pytest --cov=ahoi --doctest-modules
coverage html
```
