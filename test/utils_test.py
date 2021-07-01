from deep_rl.utils import result_to_dat
import numpy as np

def test_results_to_dat():
    timesteps = np.arange(10) * 100
    results = np.array(
        [
            [-0.776, 0.578, 1.351, 0.686, -0.105, -0.619, -0.818, 1.005, -0.294, 1.736],
            [0.328, -0.023, -0.856, -1.333, -0.055, -0.03, -1.703, -0.029, 0.191, 1.025],
            [-1.184, 2.28, -2.86, -0.367, -2.509, -1.259, 0.36, 0.046, 1.076, -1.604],
            [0.761, -0.357, -0.942, -0.588, -0.383, 1.982, 0.808, 0.758, -0.176, -0.025],
            [-0.495, -0.285, -0.78, -1.327, -1.85, 0.521, -0.877, -0.156, -0.605, 1.225],
            [0.461, -0.193, -0.379, 0.075, 0.66, -0.366, 1.077, -0.325, 0.381, -1.013],
            [0.275, 0.67, 0.289, 1.073, -0.513, -0.467, -0.815, -1.874, 1.773, 1.518],
            [1.052, 2.403, -0.317, -1.053, -0.436, -1.593, 0.195, 0.639, 0.688, 0.574],
        ]
    )
    result_to_dat(timesteps, results, filename="/tmp/test_results.dat")
    with open("/tmp/test_results.dat", "r") as file:
        dat = file.readlines()
    first_line = dat.pop(0)
    # check the content of the first line
    assert first_line == "timestep med lowq highq\n"
    # convert list of string into a numpy float array
    dat = [line.split(" ") for line in dat]
    dat = np.array(dat).astype(float)
    expected_dat = np.array(
        [
            [0.000e00, 3.010e-01, -2.560e-01, 4.100e-01],
            [1.000e02, 2.770e-01, -1.400e-01, 6.350e-01],
            [2.000e02, -5.800e-01, -8.320e-01, -3.410e-01],
            [3.000e02, -4.770e-01, -9.090e-01, -9.300e-02],
            [4.000e02, -4.090e-01, -4.890e-01, -2.110e-01],
            [5.000e02, -4.160e-01, -5.720e-01, -1.580e-01],
            [6.000e02, -3.100e-01, -8.170e-01, 2.970e-01],
            [7.000e02, 8.000e-03, -1.170e-01, 4.140e-01],
            [8.000e02, 2.860e-01, -6.200e-02, 5.710e-01],
            [9.000e02, 7.990e-01, 1.610e-01, 1.149e00],
        ]
    )
    assert np.allclose(expected_dat, dat)
