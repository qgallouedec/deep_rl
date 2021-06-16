"""Convert results into .dat files, headed by
timestep med lowq highq
"""

import numpy as np


def process(results):
    results = np.array(results, dtype=np.float)
    med = np.median(results, axis=0)
    l = results.shape[1]
    timesteps = np.arange(0, l) * 10000
    lowq = np.quantile(results, 0.33, axis=0)
    highq = np.quantile(results, 0.66, axis=0)
    out = np.vstack((timesteps, med, lowq, highq)).transpose()

    np.savetxt(
        "results.dat",
        out,
        fmt="%.3f",
        header="timestep med lowq highq",
        comments="",
    )


if __name__ == "__main__":
    results = np.random.randn(100,101)
    print(process(results))
