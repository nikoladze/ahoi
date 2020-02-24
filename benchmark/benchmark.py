#!/usr/bin/env python

import ahoi
import numpy as np
from timeit import timeit
import click


@click.command()
@click.option("--nevent", default=10000)
@click.option("--nvar", default=5)
@click.option("--ncut", default=10)
@click.option("--method", default="c")
@click.option("--progress/--no-progress", default=True)
def run_benchmark(*args, **kwargs):
    time = get_time_benchmark(*args, **kwargs)
    print(f"Took {time} seconds")


def get_time_benchmark(nevent=10000, nvar=5, ncut=10, method="c", progress=False):
    print(f"nevent: {nevent}, nvar: {nvar}, ncut: {ncut}")
    x = np.random.rand(nevent, nvar)
    w = np.random.normal(loc=1, size=len(x))
    cuts = [np.linspace(0, 1, ncut) for i in range(nvar)]
    masks_list = [[x[:, j] > i for i in cut] for j, cut in enumerate(cuts)]
    time = timeit(
        lambda: ahoi.scan(masks_list, weights=w, method=method, progress=progress),
        number=1,
    )
    return time


if __name__ == "__main__":
    run_benchmark()
