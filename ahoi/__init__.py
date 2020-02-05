import ctypes
from numpy.ctypeslib import ndpointer
import os
import glob
import numpy as np
from tqdm import tqdm


def run_per_event(masks_list, weights, event_function):

    # convert masks to 2D np arrays if not yet in that format
    if not all([isinstance(masks, np.ndarray) for masks in masks_list]):
        masks_list = [np.array(masks, dtype=np.bool) for masks in masks_list]

    shape = np.array([len(masks) for masks in masks_list], dtype=np.int64)
    counts = np.zeros(shape, dtype=np.int64)
    if weights is not None:
        sumw = np.zeros(shape, dtype=np.float64)
        sumw2 = np.zeros(shape, dtype=np.float64)
    else:
        sumw = None
        sumw2 = None

    # contiguous per event buffer (probably better for CPU cache)
    masks_buffer = np.empty(
        (len(masks_list), max([len(masks) for masks in masks_list])), dtype=np.bool
    )

    for i in tqdm(range(len(masks_list[0][0]))):

        # fill per event buffer
        for i_mask, masks in enumerate(masks_list):
            masks_buffer[i_mask][: len(masks)] = masks[:, i]

        if weights is None:
            w = None
        else:
            w = weights[i]

        event_function(masks_buffer, shape, counts, w=w, sumw=sumw, sumw2=sumw2)

    if weights is not None:
        return counts, sumw, sumw2
    else:
        return counts


def run_c(masks_list, weights):

    import importlib

    # not sure if this is the right way to find the compiled library ...
    lib = ctypes.cdll.LoadLibrary(importlib.util.find_spec("ahoi_scan").origin)
    _fill_matching = lib.fill_matching
    _fill_matching.restype = None
    _fill_matching.argtypes = [
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

    # wrap the c function
    def fill_matching(masks, shape, counts, w=None, sumw=None, sumw2=None):
        # prepare array of pointers for 2D per-event masks buffer
        p_masks = np.array(
            masks.__array_interface__["data"][0]
            + (np.arange(masks.shape[0]) * masks.strides[0]).astype(np.uintp)
        )

        counts = counts.ravel()
        use_weights = w is not None
        sumw = np.empty(0) if sumw is None else sumw.ravel()
        sumw2 = np.empty(0) if sumw2 is None else sumw2.ravel()
        if w is None:
            w = 0
        multi_index = np.zeros_like(shape, dtype=np.int32)

        _fill_matching(
            p_masks,
            w,
            0,
            multi_index,
            shape.astype(np.int32),
            shape.size,
            counts,
            sumw,
            sumw2,
            use_weights,
        )

    # run for all events
    return run_per_event(masks_list, weights, event_function=fill_matching)


def scan(masks_list, weights=None):
    return run_c(masks_list, weights)
