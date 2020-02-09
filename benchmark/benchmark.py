import ahoi
import numpy as np
import pandas as pd
from timeit import timeit


def run_benchmark(nevent_range, nvar_range, ncut_range, methods):
    df = {f"time_{method}": [] for method in methods}
    df.update({k: [] for k in ["nvar", "ncut", "nevent"]})
    for nevent in nevent_range:
        for nvar in nvar_range:
            for ncut in ncut_range:
                print(f"nevent: {nevent}, nvar: {nvar}, ncut: {ncut}")
                x = np.random.rand(nevent, nvar)
                w = np.random.normal(loc=1, size=len(x))
                cuts = [np.linspace(0, 1, ncut) for i in range(nvar)]
                masks_list = [[x[:, j] > i for i in cut] for j, cut in enumerate(cuts)]
                df["ncut"].append(ncut)
                df["nvar"].append(nvar)
                df["nevent"].append(nevent)
                for method in methods:
                    print(f'Running method "{method}"')
                    df[f"time_{method}"].append(
                        timeit(
                            lambda: ahoi.scan(masks_list, weights=w, method=method),
                            number=1,
                        )
                    )
    df = pd.DataFrame(df)
    return df


if __name__ == "__main__":

    # df = run_benchmark(nevent_range=[1000], nvar_range=np.arange(1, 10, 1), ncut_range=[5], methods=["c", "numpy", "numpy_reduce"])
    # df.to_hdf("benchmark.h5", "nvar")

    # df = run_benchmark(nevent_range=[10000], nvar_range=np.arange(1, 10, 1), ncut_range=[5], methods=["c", "numpy", "numpy_reduce"])
    # df.to_hdf("benchmark.h5", "nvar_10k")

    # df = run_benchmark(nevent_range=[100000], nvar_range=np.arange(1, 10, 1), ncut_range=[5], methods=["c", "numpy", "numpy_reduce"])
    # df.to_hdf("benchmark.h5", "nvar_100k")

    df = run_benchmark(nevent_range=np.arange(1000, 100000, 10000), nvar_range=[5], ncut_range=[10], methods=["c", "numpy", "numpy_reduce"])
    df.to_hdf("benchmark.h5", "nevent")
