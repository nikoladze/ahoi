import ctypes
from numpy.ctypeslib import ndpointer
import os
import glob
import numpy as np
from tqdm import tqdm


def scan(masks_list, weights=None):
    """
    This function is the main interface.
    """
    scanner = PerEventScannerC(masks_list, weights=weights)
    scanner.run()
    if weights is None:
        return scanner.counts
    else:
        return scanner.counts, scanner.sumw, scanner.sumw2


class Scanner:
    "Base class"

    def __init__(self, masks_list, weights=None):

        # convert masks to 2D np arrays if not yet in that format
        if not all([isinstance(masks, np.ndarray) for masks in masks_list]):
            masks_list = [np.array(masks, dtype=np.bool) for masks in masks_list]
        self.masks_list = masks_list

        self.weights = weights

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

    def run(self):
        for i in tqdm(range(len(self.masks_list[0][0]))):

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

        # not sure if this is the right way to find the compiled library ...
        import importlib

        lib = ctypes.cdll.LoadLibrary(importlib.util.find_spec("ahoi_scan").origin)
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
