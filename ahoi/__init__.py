import ctypes
from numpy.ctypeslib import ndpointer
import os
import sys
import glob
import numpy as np
from tqdm import tqdm
from multiprocessing import Process, Queue


def scan(
    masks_list,
    weights=None,
    counts=None,
    sumw=None,
    sumw2=None,
    method="c",
    progress=False,
    workers=1,
    trim_masks_list=True,
):
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
    counts: ndarray, optional
        Fill this counts array in-place instead of allocating a new one. Can be used to
        fill the counts chunkwise. The array has to have a shape corresponding
        to the lengths of the masks in masks_list. Has to be int64.
    sumw: ndarray, optional
        See counts - for the case that weights and counts are passed, sumw has
        to be passed as well. Has to be float64.
    sumw2: ndarray, optional
        See counts - for the case that weights and counts are passed, sumw2 has
        to be passed as well. Has to be float64.
    method: {"c", "numpy", "numpy_reduce"}, optional
        Method to use for the scan. "c" (default) uses a precompiled c function
        to perform the scan on a per-event basis, "numpy" and "numpy_reduce"
        use numpy functions to perform the outer loop over all combinations. In
        case of a large number of combinations "c" is the fastest.
    progress: bool, optional
        If True, show progress bar
    workers: int, optional
        If > 1 then use this number of processes to parallelize over events.
        Note that this will also multiply the memory consumption by the number
        of workers.
    trim_masks_list: bool, optional
        Before scanning, internally reduce masks_list to only contain entries that pass
        any combination.

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
        "c": ScannerC,
        "numpy": ScannerNumpy,
        "numpy_reduce": ScannerNumpyReduce,
    }

    if trim_masks_list:
        pass_any = get_pass_any(masks_list)
        if weights is not None:
            pass_any &= np.array(weights, copy=False) != 0
        masks_list = [
            np.array(
                [np.array(mask, dtype=np.bool, copy=False)[pass_any] for mask in masks]
            )
            for masks in masks_list
        ]
        if weights is not None:
            weights = np.array(weights, copy=False)[pass_any]

    if workers < 2:
        # if 1 worker, just run directly
        scanner = scanner_dict[method](
            masks_list, weights=weights, counts=counts, sumw=sumw, sumw2=sumw2
        )
        scanner.run(progress=progress)
        counts, sumw, sumw2 = scanner.counts, scanner.sumw, scanner.sumw2
    else:
        # otherwise spawn subprocesses
        queue_counts = Queue(1)
        queue_sumw = None if weights is None else Queue(1)
        queue_sumw2 = None if weights is None else Queue(1)

        # split masks_list into number of workers parts
        masks_list_dict = {}
        for j, masks in enumerate(masks_list):
            worker_masks = np.array_split(np.array(masks), workers, axis=1)
            for i_worker, worker_mask in enumerate(worker_masks):
                if not i_worker in masks_list_dict:
                    masks_list_dict[i_worker] = {}
                masks_list_dict[i_worker][j] = worker_masks[i_worker]

        # split weights into number of workers parts
        if weights is not None:
            weights_list = np.array_split(weights, workers)
        else:
            weights_list = [None for i in range(workers)]

        def run_scanner(masks_list, weights, queue_counts, queue_sumw, queue_sumw2):
            scanner = scanner_dict[method](masks_list, weights=weights)
            scanner.run(progress=progress)
            queue_counts.put(scanner.counts)
            if weights is not None:
                queue_sumw.put(scanner.sumw)
                queue_sumw2.put(scanner.sumw2)

        # start workers
        for i_worker in range(workers):
            masks_list_worker = [
                masks_list_dict[i_worker][i] for i in range(len(masks_list))
            ]
            weights_worker = weights_list[i_worker]
            p = Process(
                target=run_scanner,
                args=(
                    masks_list_worker,
                    weights_worker,
                    queue_counts,
                    queue_sumw,
                    queue_sumw2,
                ),
            )
            p.start()

        # sum results
        for i_worker in tqdm(range(workers), disable=not progress, desc="Summing"):
            worker_counts = np.array(queue_counts.get())
            if weights is not None:
                worker_sumw = np.array(queue_sumw.get())
                worker_sumw2 = np.array(queue_sumw2.get())
            if counts is None:
                counts = worker_counts
                if weights is None:
                    continue
            if sumw is None and weights is not None:
                sumw = worker_sumw
                sumw2 = worker_sumw2
                continue
            counts += worker_counts
            if weights is None:
                continue
            sumw += worker_sumw
            sumw2 += worker_sumw2

    if weights is None:
        return counts
    else:
        return counts, sumw, sumw2


def get_pass_any(masks_list):
    "Return mask for entries that pass any combination."
    pass_any_combination = None
    for masks in masks_list:
        pass_any_cut = None
        for mask in masks:
            mask = np.array(mask, dtype=np.bool)
            if pass_any_cut is None:
                pass_any_cut = mask
            else:
                pass_any_cut |= mask
        if pass_any_combination is None:
            pass_any_combination = pass_any_cut
        else:
            pass_any_combination &= pass_any_cut
    return pass_any_combination


class Scanner(object):
    "Base class"

    def __init__(self, masks_list, weights=None, counts=None, sumw=None, sumw2=None):

        # convert masks to 2D np arrays if not yet in that format
        for i, masks in enumerate(masks_list):
            masks_list[i] = np.array(masks, dtype=np.bool, copy=False)
        self.masks_list = masks_list

        # convert weights to np.ndarray if not yet of that type
        self.weights = weights
        if self.weights is not None:
            self.weights = np.array(self.weights, copy=False)

        self.shape = np.array([len(masks) for masks in masks_list], dtype=np.int64)

        # counts, sumw, sumw2 can be passed to be filled in place
        self.counts = counts
        self.sumw = sumw
        self.sumw2 = sumw2
        if (
            self.weights is not None
            and self.counts is not None
            and (self.sumw is None or self.sumw2 is None)
        ):
            raise ValueError(
                "`sumw` and `sumw2` are required if `counts` and `weights` are passed"
            )

        # otherwise new arrays are allocated
        if self.counts is None:
            self.counts = np.zeros(self.shape, dtype=np.int64)
        if self.weights is not None and self.sumw is None:
            self.sumw = np.zeros(self.shape, dtype=np.float64)
            self.sumw2 = np.zeros(self.shape, dtype=np.float64)

        # check if shape and dtype is correct (important if they were passed)
        self._check_shape_and_type("counts", np.int64)
        if self.weights is not None:
            self._check_shape_and_type("sumw", np.float64)
            self._check_shape_and_type("sumw2", np.float64)

    def _check_shape_and_type(self, array_name, dtype):
        if not isinstance(getattr(self, array_name), np.ndarray):
            raise TypeError("`{}` has to be of type `np.ndarray`".format(array_name))
        if not getattr(self, array_name).dtype == dtype:
            raise TypeError(
                "`{}` has to be `{}`, but is `{}`".format(
                    array_name, dtype, getattr(self, array_name).dtype
                )
            )
        if (len(getattr(self, array_name).shape) != len(self.shape)) or (
            not all(getattr(self, array_name).shape == self.shape)
        ):
            raise TypeError(
                "the shape `{}` of `{}` doesn't match the expected shape "
                "determined from `masks_list` ({})".format(
                    getattr(self, array_name).shape, array_name, self.shape
                )
            )


class ScannerC(Scanner):
    "per-event scan with compiled c function"

    def __init__(self, *args, **kwargs):
        super(ScannerC, self).__init__(*args, **kwargs)

        # contiguous per event buffer (probably better for CPU cache)
        self.masks_buffer = np.empty(
            (len(self.masks_list), max([len(masks) for masks in self.masks_list])),
            dtype=np.bool,
        )

        # ... not sure if this is the right way to find the library
        if sys.version_info[0] < 3:
            import pkgutil

            lib_filename = pkgutil.get_loader("ahoi.ahoi_scan").filename
        else:
            import importlib

            lib_filename = importlib.util.find_spec(".ahoi_scan", "ahoi").origin

        lib = ctypes.cdll.LoadLibrary(lib_filename)
        self._fill_matching = lib.fill_matching
        self._fill_matching.restype = None
        self._fill_matching.argtypes = [
            ndpointer(dtype=np.uintp, ndim=1, flags="C_CONTIGUOUS"),  # char **masks
            ctypes.c_double,  # double wi
            ctypes.c_int,  # size_t j
            ctypes.c_size_t,  # size_t combination_index
            ctypes.c_int,  # int index_factor
            ndpointer(
                dtype=ctypes.c_size_t, ndim=1, flags="C_CONTIGUOUS"
            ),  # size_t *shape
            ctypes.c_size_t,  # size_t ndims
            ndpointer(
                dtype=ctypes.c_long, ndim=1, flags="C_CONTIGUOUS"
            ),  # long *counts
            ndpointer(
                dtype=ctypes.c_double, ndim=1, flags="C_CONTIGUOUS"
            ),  # double *sumw
            ndpointer(
                dtype=ctypes.c_double, ndim=1, flags="C_CONTIGUOUS"
            ),  # double *sumw2
            ctypes.c_bool,  # char use_weights
        ]

        # prepare array of pointers for 2D per-event masks buffer
        self._p_masks = np.array(
            self.masks_buffer.__array_interface__["data"][0]
            + (
                np.arange(self.masks_buffer.shape[0]) * self.masks_buffer.strides[0]
            ).astype(np.uintp)
        )
        # the other pointers
        self._p_counts = self.counts.ravel()
        self._p_sumw = np.empty(0) if self.sumw is None else self.sumw.ravel()
        self._p_sumw2 = np.empty(0) if self.sumw2 is None else self.sumw2.ravel()
        self._p_shape = self.shape.astype(ctypes.c_size_t)

    def run(self, progress=True):
        for i in tqdm(
            range(len(self.masks_list[0][0])), disable=not progress, desc="Events"
        ):
            # fill per event buffer
            for i_mask, masks in enumerate(self.masks_list):
                self.masks_buffer[i_mask][: len(masks)] = masks[:, i]

            if self.weights is None:
                w = None
            else:
                w = self.weights[i]

            self.run_event(self.masks_buffer, w=w)

    def run_event(self, masks_buffer, w=None):
        "Wrap around c function"

        use_weights = w is not None
        if w is None:
            w = 0

        self._fill_matching(
            self._p_masks,  # char **masks
            w,  # double wi
            0,  # size_t j
            0,  # size_t combination_index
            np.prod(self.shape[1:]),  # int index_factor
            self._p_shape,  # size_t *shape
            self._p_shape.size,  # size_t ndims
            self._p_counts,  # long *counts
            self._p_sumw,  # double *sumw
            self._p_sumw2,  # double *sumw2
            use_weights,  # char use_weights
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
                    self.counts[tuple(multi_index)] += np.count_nonzero(new_mask)
                    if self.weights is not None:
                        self.sumw[tuple(multi_index)] += np.dot(new_mask, w)
                        self.sumw2[tuple(multi_index)] += np.dot(new_mask, w2)
                    yield 1

        for i in tqdm(
            fill(0, current_mask),
            total=len(self.counts.ravel()),
            desc="Combinations",
            disable=not progress,
        ):
            pass


class ScannerNumpyReduce(Scanner):
    def run(self, progress=True):

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
                if j != (len(self.shape) - 1):
                    new_masks_list = [
                        [new_mask[mask] for new_mask in masks]
                        for masks in masks_list[1:]
                    ]
                    for _ in fill(new_masks_list, j + 1, new_w, new_w2):
                        yield 1
                else:
                    self.counts[tuple(multi_index)] += np.count_nonzero(mask)
                    if self.weights is not None:
                        self.sumw[tuple(multi_index)] += new_w.sum()
                        self.sumw2[tuple(multi_index)] += new_w2.sum()
                    yield 1

        for i in tqdm(
            fill(self.masks_list, 0, w, w2),
            total=len(self.counts.ravel()),
            desc="Combinations",
            disable=not progress,
        ):
            pass
