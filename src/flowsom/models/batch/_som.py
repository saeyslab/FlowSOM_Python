"""Code adapted from student assignment Computational Biology 2024, Ghent University."""

from typing import Callable

import numpy as np
from numba import jit, prange
from sklearn.neighbors import BallTree

from flowsom.models.numpy_numba import nb_median_axis_0


@jit(nopython=True, fastmath=True)
def eucl_without_sqrt(p1: np.ndarray, p2: np.ndarray):
    """Function that computes the Euclidean distance between two points without taking the square root.

    For performance reasons, the square root is not taken. This is useful when comparing distances, because the square
    root is a monotonic function, meaning that the order of the distances is preserved.

    Args:
        p1 (np.ndarray): The first point.
        p2 (np.ndarray): The second point.

    Returns
    -------
        float: The Euclidean distance between the two points.

    >>> eucl_without_sqrt(np.array([1, 2, 3]), np.array([4, 5, 6]))
    27.0
    """
    distance = 0.0
    for j in range(p1.shape[0]):
        diff = p1[j] - p2[j]
        distance += diff * diff
    return distance


@jit(nopython=True, parallel=True, fastmath=True)
def SOM_Batch(
    data: np.ndarray,
    codes: np.ndarray,
    nhbrdist: np.ndarray,
    alphas: tuple,
    radii: tuple,
    ncodes: int,
    rlen: int,
    num_batches: int = 10,
    distf: Callable[[np.ndarray, np.ndarray], float] = eucl_without_sqrt,
    seed=None,
):
    """Function that computes the Self-Organizing Map.

    Args:
        data (np.ndarray): The data to be clustered.
        codes (np.ndarray): The initial codes.
        nhbrdist (np.ndarray): The neighbourhood distances.
        alphas (tuple): The alphas.
        radii (tuple): The radii.
        ncodes (int): The number of codes.
        rlen (int): The number of iterations.
        num_batches (int): The number of batches.
        distf (function): The distance function.
        seed (int): The seed for the random number generator.

    Returns
    -------
        np.ndarray: The computed codes.
    """
    if seed is not None:
        np.random.seed(seed)

    # Number of data points
    n = data[-1].shape[0]

    # Dimension of the data
    px = data[0].shape[1]

    # Number of iterations
    niter = n

    # The threshold is the radius of the neighbourhood, meaning in which range codes are updated.
    # The threshold step decides how much the threshold is decreased each iteration.
    treshold_step = (radii[0] - radii[1]) / niter

    # Keep the temporary codes, using the given codes as the initial codes, for every batch
    tmp_codes_all = np.empty((num_batches, ncodes, px), dtype=np.float64)

    # Copy the codes as a float64, because the codes are updated in the algorithm
    copy_codes = codes.copy().astype(np.float64)

    # Execute some initial serial iterations to get a good init clustering
    xdist = np.empty(ncodes, dtype=np.float64)
    init_threshold = radii[0]
    init_alpha = alphas[0]

    for i in range(niter):
        # Choose a random data point
        i = np.random.choice(n)

        # Compute the nearest code
        nearest = 0
        for cd in range(ncodes):
            xdist[cd] = distf(data[0][i, :], copy_codes[cd, :])
            if xdist[cd] < xdist[nearest]:
                nearest = cd

        init_alpha = alphas[0] - (alphas[0] - alphas[1]) * i / (niter * rlen)

        for cd in range(ncodes):
            # The neighbourhood distance decides whether the code is updated. This states that the code is only updated
            # if they are close enough to each other. Otherwise, the value stays the same.
            if nhbrdist[cd, nearest] <= init_threshold:
                # Update the code based on the difference between the used data point and the code.
                for j in range(px):
                    tmp = data[0][i, j] - copy_codes[cd, j]
                    copy_codes[cd, j] += tmp * init_alpha

        init_threshold -= treshold_step

    # Choose random data points, for the different batches, and the rlen iterations
    data_points_random = np.random.choice(n, num_batches * rlen * n, replace=True)

    # Decrease the number of iterations, because the first iterations are already done
    rlen = int(rlen / 2)

    for iteration in range(rlen):
        # Execute the batches in parallel
        for batch_nr in prange(num_batches):
            # Keep the temporary codes, using the given codes as the initial codes
            tmp_codes = copy_codes.copy()

            # Array for the distances
            xdists = np.empty(ncodes, dtype=np.float64)

            # IMPORTANT: When setting the threshold to radii[0], this causes big changes every iteration. This is not
            # wanted, because the algorithm should converge. Therefore, the threshold is decreased every iteration.
            # Update: factor 2 is added, to make the threshold decrease faster.
            threshold = init_threshold - radii[0] * 2 * iteration / rlen

            for k in range(iteration * niter, (iteration + 1) * niter):
                # Get the data point
                i = data_points_random[n * rlen * batch_nr + k]

                # Compute the nearest code
                nearest = 0
                for cd in range(ncodes):
                    xdists[cd] = distf(data[batch_nr][i, :], tmp_codes[cd, :])
                    if xdists[cd] < xdists[nearest]:
                        nearest = cd

                if threshold < 1.0:
                    threshold = 0.5
                alpha = init_alpha - (alphas[0] - alphas[1]) * k / (niter * rlen)

                for cd in range(ncodes):
                    # The neighbourhood distance decided whether the code is updated. This states that the code is only updated
                    # if they are close enough to each other. Otherwise, the value stays the same.
                    if nhbrdist[cd, nearest] <= threshold:
                        # Update the code based on the difference between the used data point and the code.
                        for j in range(px):
                            tmp = data[batch_nr][i, j] - tmp_codes[cd, j]
                            tmp_codes[cd, j] += tmp * alpha

                threshold -= treshold_step

            tmp_codes_all[batch_nr] = tmp_codes

        # Merge the different SOM's together
        copy_codes = nb_median_axis_0(tmp_codes_all).astype(np.float64)

    return copy_codes


# ChatGPT generated alternative to map_data_to_codes
def map_data_to_codes(data, codes):
    """Returns a tuple with the indices and distances of the nearest code for each data point.

    Args:
        data (np.ndarray): The data points.
        codes (np.ndarray): The codes that the data points are mapped to.

    Returns
    -------
        np.ndarray: The indices of the nearest code for each data point.
        np.ndarray: The distances of the nearest code for each data point.

    >>> data_ = np.array([[1, 2, 3], [4, 5, 6]])
    >>> codes_ = np.array([[1, 2, 3], [4, 5, 6]])
    >>> map_data_to_codes(data_, codes_)
    (array([0, 1]), array([0., 0.]))
    """
    # Create a BallTree for the codes (this is an efficient data structure for nearest neighbor search)
    tree = BallTree(codes, metric="euclidean")

    # Query the BallTree to find the nearest code for each data point (k=1 means we only want the nearest neighbor)
    dists, indices = tree.query(data, k=1)

    # Flatten the results and return them
    return indices.flatten(), dists.flatten()
