import ahoi
import numpy as np

def test_examples():
    test_values = [[0.1, 0.5, 0.8], [0.6, 0.8], [0.1, 0.5, 0.8, 0.9]]
    for run_f in [ahoi.scan]:
        x = np.random.rand(10000, 3)
        w = np.random.normal(loc=1, size=len(x))
        masks_list = [
            [x[:, 0] > i for i in test_values[0]],
            [x[:, 1] < i for i in test_values[1]],
            [x[:, 2] > i for i in test_values[2]],
        ]
        counts, sumw, sumw2 = run_f(masks_list, weights=w)
        for i0, v0 in enumerate(test_values[0]):
            for i1, v1 in enumerate(test_values[1]):
                for i2, v2 in enumerate(test_values[2]):
                    mask = (x[:, 0] > v0) & (x[:, 1] < v1) & (x[:, 2] > v2)
                    assert counts[i0][i1][i2] == np.count_nonzero(mask)
                    assert np.isclose(sumw[i0][i1][i2], np.dot(mask, w))
                    assert np.isclose(sumw2[i0][i1][i2], np.dot(mask, w ** 2))
