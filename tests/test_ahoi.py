import ahoi
import numpy as np

scanner_methods = ["c", "numpy", "numpy_reduce"]

def test_examples():
    test_values = [[0.1, 0.5, 0.8], [0.6, 0.8], [0.1, 0.5, 0.8, 0.9]]
    for method in scanner_methods:
        x = np.random.rand(10000, 3)
        w = np.random.normal(loc=1, size=len(x))
        masks_list = [
            [x[:, 0] > i for i in test_values[0]],
            [x[:, 1] < i for i in test_values[1]],
            [x[:, 2] > i for i in test_values[2]],
        ]
        counts, sumw, sumw2 = ahoi.scan(masks_list, weights=w, method=method)
        for i0, v0 in enumerate(test_values[0]):
            for i1, v1 in enumerate(test_values[1]):
                for i2, v2 in enumerate(test_values[2]):
                    mask = (x[:, 0] > v0) & (x[:, 1] < v1) & (x[:, 2] > v2)
                    assert counts[i0][i1][i2] == np.count_nonzero(mask)
                    assert np.isclose(sumw[i0][i1][i2], np.dot(mask, w))
                    assert np.isclose(sumw2[i0][i1][i2], np.dot(mask, w ** 2))


def test_noweights():
    for method in scanner_methods:
        x = np.random.rand(1000, 3)
        masks_list = [
            [x[:, j] > i for i in np.linspace(0, 1, 10)] for j in range(x.shape[1])
        ]
        w = np.random.normal(size=1000)
        counts, _, _ = ahoi.scan(masks_list, w, method=method)
        counts2 = ahoi.scan(masks_list, method=method)
        assert (counts == counts2).all()


def test_singlecut():
    for method in scanner_methods:
        masks_list = [[[0]]]
        counts = ahoi.scan(masks_list, method=method)
        assert ahoi.scan([[[0]]], method=method) == [0]
        assert ahoi.scan([[[1]]], method=method) == [1]
        assert ahoi.scan([[[1, 0, 0, 1]]], method=method) == [2]
        assert np.all(ahoi.scan([[[1, 0, 0, 1], [1, 1, 1, 0], [1, 0, 0, 0]]]) == [2, 3, 1])
        counts, sumw, sumw2 = ahoi.scan(
            [[[1, 0, 0, 1], [1, 1, 1, 0], [1, 0, 0, 0]]], weights=[2, 3, 4, 5], method=method
        )
        assert np.all(sumw == [7, 9, 2])
        assert np.all(sumw2 == [29, 29, 4])


def test_consistency():
    x = np.random.rand(1000, 5)
    w = np.random.normal(loc=1, size=len(x))
    masks_list = [
        [x[:, 0] > i for i in np.arange(0, 1, 0.01)],
        [x[:, 1] > i for i in np.arange(0, 1, 0.1)],
        [x[:, 2] > i for i in np.arange(0, 1, 0.3)],
        [x[:, 3] > i for i in np.arange(0, 1, 0.1)],
        [x[:, 4] > i for i in np.arange(0, 1, 0.5)],
    ]
    counts_ref, sumw_ref, sumw2_ref = ahoi.scan(masks_list, weights=w, method=scanner_methods[0])
    for method in scanner_methods[1:]:
        counts, sumw, sumw2 = ahoi.scan(masks_list, weights=w, method=method)
        assert (counts_ref == counts).all()
        assert np.allclose(sumw_ref, sumw)
        assert np.allclose(sumw2_ref, sumw2)
