import ctypes
from numpy.ctypeslib import ndpointer
import os
import glob
import numpy as np
from tqdm import tqdm

def run_per_event(masks_list, weights, event_function):

    if weights is None:
        weights = np.ones(len(masks_list[0][0]), dtype=np.float32)

    # convert masks to 2D np arrays if not yet in that format
    if not all([isinstance(masks, np.ndarray) for masks in masks_list]):
        masks_list = [np.array(masks, dtype=np.bool) for masks in masks_list]

    dims = np.array([len(masks) for masks in masks_list], dtype=np.int64)
    inds = np.zeros_like(dims, dtype=np.int64)
    counts = np.zeros(dims, dtype=np.int64)
    sumw = np.zeros(dims, dtype=np.float64)
    sumw2 = np.zeros(dims, dtype=np.float64)

    # contiguous per event buffer (probably better for CPU cache)
    masks_buffer = np.empty(
        (len(masks_list), max([len(masks) for masks in masks_list])), dtype=np.bool
    )

    for i in tqdm(range(len(masks_list[0][0]))):

        # fill per event buffer
        for i_mask, masks in enumerate(masks_list):
            masks_buffer[i_mask][: len(masks)] = masks[:, i]

        event_function(
            masks_buffer,
            weights[i],
            0,
            inds,
            dims,
            counts.reshape(-1),
            sumw.reshape(-1),
            sumw2.reshape(-1),
        )
    return counts, sumw, sumw2


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
        ndpointer(dtype=ctypes.c_int, ndim=1, flags="C_CONTIGUOUS"),  # inds
        ndpointer(dtype=ctypes.c_int, ndim=1, flags="C_CONTIGUOUS"),  # dims
        ctypes.c_size_t,  # ndims
        ndpointer(dtype=ctypes.c_long, ndim=1, flags="C_CONTIGUOUS"),  # counts
        ndpointer(dtype=ctypes.c_double, ndim=1, flags="C_CONTIGUOUS"),  # sumw
        ndpointer(dtype=ctypes.c_double, ndim=1, flags="C_CONTIGUOUS"),  # sumw2
    ]

    def fill_matching(masks, wi, j, inds, dims, counts, sumw, sumw2):
        p_masks = np.array(
            masks.__array_interface__["data"][0]
            + (np.arange(masks.shape[0]) * masks.strides[0]).astype(np.uintp)
        )
        _fill_matching(
            p_masks,
            wi,
            j,
            inds.astype(np.int32),
            dims.astype(np.int32),
            dims.size,
            counts.ravel(),
            sumw.ravel(),
            sumw2.ravel(),
        )

    return run_per_event(masks_list, weights, event_function=fill_matching)


def scan(masks_list, weights=None):
    return run_c(masks_list, weights)
