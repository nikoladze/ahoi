import ahoi
import numpy as np

x = np.random.rand(1000, 7)

masks_list = [
    [x[:, 0] > i for i in np.linspace(0.9, 1, 10)],
    [x[:, 1] > i for i in np.arange(0, 1, 0.1)],
    [x[:, 2] > i for i in np.arange(0, 1, 0.1)],
    [x[:, 3] > i for i in np.arange(0, 1, 0.1)],
    [x[:, 4] > i for i in np.arange(0, 1, 0.1)],
    [x[:, 5] > i for i in np.arange(0, 1, 0.1)],
    [x[:, 6] > i for i in np.arange(0, 1, 0.1)],
]

w = np.random.normal(size=len(x))
counts, sumw, sumw2 = ahoi.scan(masks_list, w, progress=True, workers=2)
