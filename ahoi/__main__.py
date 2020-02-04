import ahoi
import numpy as np

np.random.seed(42)
x = np.random.rand(10000, 7)
w = np.random.normal(loc=1, size=len(x))
masks_list = [
    # [x[:, 0] > i for i in np.arange(0, 1, 0.1)],
    [x[:, 0] > i for i in np.linspace(0.9, 1, 10)],
    [x[:, 1] > i for i in np.arange(0, 1, 0.1)],
    [x[:, 2] > i for i in np.arange(0, 1, 0.1)],
    [x[:, 3] > i for i in np.arange(0, 1, 0.1)],
    [x[:, 4] > i for i in np.arange(0, 1, 0.1)],
    [x[:, 5] > i for i in np.arange(0, 1, 0.1)],
    [x[:, 6] > i for i in np.arange(0, 1, 0.1)],
]

counts_c, sumw_c, sumw2_c = ahoi.run_c(masks_list, w)
