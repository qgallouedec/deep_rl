import numpy as np


def result_to_dat(timesteps, results, filename="results.dat"):
    """Generate a .dat with 4 columns : timestep, median value, low and high quantile value.

    Args:
        timesteps (list): Timesteps.
        results (list[list]): The the results from multiple experiences.
        filename (str, optional): The output file name. Defaults to "results.dat".
    """
    results = np.array(results, dtype=float)
    timesteps = np.array(timesteps, dtype=float)
    med = np.median(results, axis=0)
    lowq = np.quantile(results, 0.33, axis=0)
    highq = np.quantile(results, 0.66, axis=0)
    out = np.vstack((timesteps, med, lowq, highq)).transpose()

    np.savetxt(
        filename,
        out,
        fmt=["%d", "%.3f", "%.3f", "%.3f"],
        header="timestep med lowq highq",
        comments="",
    )


if __name__ == "__main__":
    results = np.round(np.random.randn(8, 10), 3)
    print(results)
    timesteps = np.arange(10) * 100
    result_to_dat(timesteps, results)
