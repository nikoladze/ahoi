import ctypes
from numpy.ctypeslib import ndpointer
import os
import sys
import glob
import numpy as np
from tqdm import tqdm


def scan(masks_list, weights=None, method="c", progress=False):
    """
    Scan all combinations of matching flags

    Parameters:
    -----------
    masks_list: list of array_like
        A list of masks where masks are 2d arrays with shape (n_criteria,
        n_events) of match flags for all selection criteria for a certain
        selection.
    weights: array_like, optional
        An array of weights for all events. If given, in addition to the counts of
        matching combinations, the sum of weights and the sum of squares of
        weights will be filled and returned.
    method: {"c", "numpy", "numpy_reduce"}, optional
        Method to use for the scan. "c" (default) uses a precompiled c function
        to perform the scan on a per-event basis, "numpy" and "numpy_reduce"
        use numpy functions to perform the main loop over all combinations. In
        most cases "c" is the fastest.
    progress: bool, optional
        If True, show progress bar

    Returns:
    --------
    counts: ndarray
        A multi-dimensional array of counts for matching combinations.
    sumw: ndarray, optional
        A multi-dimensional array of the sum of weights for matching
        combinations. Only provided if weights is not None.
    sumw2: ndarray, optional
        A multi-dimensional array of the sum of squares of weights for matching
        combinations. Only provided if weights is not None.

    Examples:
    ---------
    Scan 4 events for combinations of two selections (e.g. cut variables). The
    first selection has two criteria (e.g. cut values), where the first
    criterion matches for the first 3 events, the second one for the first and
    third event. The second selection has 3 criteria with events (0, 1, 3), (1,
    3) and 4 matching. That results in combinations for which the counts of
    matching events will be returned.

    >>> scan([[[1, 1, 1, 0], [1, 0, 1, 0]],
    ...       [[1, 1, 0, 1], [0, 1, 0, 1], [0, 0, 0, 1]]])
    array([[2, 1, 0],
           [1, 0, 0]])

    It is possible to pass weights. In this case the sum of weights and sum of
    squares of weights for each combination will be returned as well.

    >>> scan([[[1, 1, 1, 0], [1, 0, 1, 0]],
    ...       [[1, 1, 0, 1], [0, 1, 0, 1], [0, 0, 0, 1]]], weights=[1.2, 5., 0.1, 1.])
    ... # doctest:+NORMALIZE_WHITESPACE
    (array([[2, 1, 0],
           [1, 0, 0]]),
    array([[6.2, 5. , 0. ],
           [1.2, 0. , 0. ]]),
    array([[26.44, 25.  ,  0.  ],
           [ 1.44,  0.  ,  0.  ]]))
    """
    scanner_dict = {
        "c": PerEventScannerC,
        "numpy": ScannerNumpy,
        "numpy_reduce": ScannerNumpyReduce,
    }
    scanner = scanner_dict[method](masks_list, weights=weights)
    scanner.run(progress=progress)
    if weights is None:
        return scanner.counts
    else:
        return scanner.counts, scanner.sumw, scanner.sumw2


class Scanner(object):
    "Base class"

    def __init__(self, masks_list, weights=None):

        # convert masks to 2D np arrays if not yet in that format
        if not all([isinstance(masks, np.ndarray) for masks in masks_list]):
            masks_list = [np.array(masks, dtype=np.bool) for masks in masks_list]
        self.masks_list = masks_list

        self.weights = weights
        if (self.weights is not None) and (not isinstance(self.weights, np.ndarray)):
            self.weights = np.array(self.weights, dtype=np.float64)

        self.shape = np.array([len(masks) for masks in masks_list], dtype=np.int64)
        self.counts = np.zeros(self.shape, dtype=np.int64)
        if self.weights is not None:
            self.sumw = np.zeros(self.shape, dtype=np.float64)
            self.sumw2 = np.zeros(self.shape, dtype=np.float64)
        else:
            self.sumw = None
            self.sumw2 = None


class PerEventScanner(Scanner):
    "Base class for per-event methods"

    def __init__(self, masks_list, weights=None):
        super(PerEventScanner, self).__init__(masks_list, weights=weights)

        # contiguous per event buffer (probably better for CPU cache)
        self.masks_buffer = np.empty(
            (len(self.masks_list), max([len(masks) for masks in self.masks_list])),
            dtype=np.bool,
        )

    def run(self, progress=True):
        for i in tqdm(range(len(self.masks_list[0][0])), disable=not progress, desc="Events"):

            # fill per event buffer
            for i_mask, masks in enumerate(self.masks_list):
                self.masks_buffer[i_mask][: len(masks)] = masks[:, i]

            if self.weights is None:
                w = None
            else:
                w = self.weights[i]

            self.run_event(self.masks_buffer, w=w)


class PerEventScannerC(PerEventScanner):
    "per-event scan with compiled c function"

    def __init__(self, masks_list, weights=None):
        super(PerEventScannerC, self).__init__(masks_list, weights=weights)

        # ... not sure if this is the right way to find the library
        if sys.version_info[0] < 3:
            import pkgutil
            lib_filename = pkgutil.get_loader("ahoi_scan").filename
        else:
            import importlib
            lib_filename = importlib.util.find_spec("ahoi_scan").origin
        lib = ctypes.cdll.LoadLibrary(lib_filename)
        self._fill_matching = lib.fill_matching
        self._fill_matching.restype = None
        self._fill_matching.argtypes = [
            ndpointer(dtype=np.uintp, ndim=1, flags="C_CONTIGUOUS"),  # masks
            ctypes.c_double,  # wi
            ctypes.c_int,  # j
            ndpointer(dtype=ctypes.c_int, ndim=1, flags="C_CONTIGUOUS"),  # multi_index
            ndpointer(dtype=ctypes.c_int, ndim=1, flags="C_CONTIGUOUS"),  # shape
            ctypes.c_size_t,  # ndims
            ndpointer(dtype=ctypes.c_long, ndim=1, flags="C_CONTIGUOUS"),  # counts
            ndpointer(dtype=ctypes.c_double, ndim=1, flags="C_CONTIGUOUS"),  # sumw
            ndpointer(dtype=ctypes.c_double, ndim=1, flags="C_CONTIGUOUS"),  # sumw2
            ctypes.c_bool,  # use_weights
        ]

    def run_event(self, masks_buffer, w=None):
        "Wrap around c function"

        # prepare array of pointers for 2D per-event masks buffer
        p_masks = np.array(
            masks_buffer.__array_interface__["data"][0]
            + (np.arange(masks_buffer.shape[0]) * masks_buffer.strides[0]).astype(
                np.uintp
            )
        )

        counts = self.counts.ravel()
        use_weights = w is not None
        sumw = np.empty(0) if self.sumw is None else self.sumw.ravel()
        sumw2 = np.empty(0) if self.sumw2 is None else self.sumw2.ravel()
        if w is None:
            w = 0
        multi_index = np.zeros_like(self.shape, dtype=np.int32)

        self._fill_matching(
            p_masks,
            w,
            0,
            multi_index,
            self.shape.astype(np.int32),
            self.shape.size,
            counts,
            sumw,
            sumw2,
            use_weights,
        )


class ScannerNumpy(Scanner):
    def run(self, progress=True):

        current_mask = np.ones_like(self.masks_list[0][0], dtype=np.bool)
        multi_index = np.zeros_like(self.shape, dtype=np.int32)
        if self.weights is not None:
            w = self.weights
            w2 = self.weights ** 2

        def fill(j, current_mask):
            for i, mask in enumerate(self.masks_list[j]):
                multi_index[j] = i
                new_mask = current_mask & mask
                if j != (len(self.masks_list) - 1):
                    for _ in fill(j + 1, new_mask):
                        yield 1
                else:
                    self.counts[tuple(multi_index)] = np.count_nonzero(new_mask)
                    if self.weights is not None:
                        self.sumw[tuple(multi_index)] = np.dot(new_mask, w)
                        self.sumw2[tuple(multi_index)] = np.dot(new_mask, w2)
                    yield 1

        for i in tqdm(
                fill(0, current_mask), total=len(self.counts.ravel()),
                desc="Combinations",
                disable=not progress
        ):
            pass


class ScannerNumpyReduce(Scanner):
    def run(self, progress=True):

        current_mask = np.ones_like(self.masks_list[0][0], dtype=np.bool)
        multi_index = np.zeros_like(self.shape, dtype=np.int32)
        w = None
        w2 = None
        if self.weights is not None:
            w = self.weights
            w2 = self.weights ** 2

        def fill(masks_list, j, w=None, w2=None):
            for i, mask in enumerate(masks_list[0]):
                multi_index[j] = i
                new_w = None
                new_w2 = None
                if self.weights is not None:
                    new_w = w[mask]
                    new_w2 = w2[mask]
                if j != (len(self.masks_list) - 1):
                    new_masks_list = [
                        [new_mask[mask] for new_mask in masks] for masks in masks_list[1:]
                    ]
                    for _ in fill(new_masks_list, j + 1, new_w, new_w2):
                        yield 1
                else:
                    self.counts[tuple(multi_index)] = np.count_nonzero(mask)
                    if self.weights is not None:
                        self.sumw[tuple(multi_index)] = new_w.sum()
                        self.sumw2[tuple(multi_index)] = new_w2.sum()
                    yield 1

        for i in tqdm(
                fill(self.masks_list, 0, w, w2), total=len(self.counts.ravel()),
                desc="Combinations",
                disable=not progress
        ):
            pass
