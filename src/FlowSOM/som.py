import numpy as np
from numba import jit


@jit(nopython=True, parallel=True)
def eucl(p1, p2):
    distance = 0.0
    for j in range(len(p1)):
        diff = p1[j] - p2[j]
        distance += diff * diff
    return np.sqrt(distance)


@jit(nopython=True, parallel=True)
def manh(p1, p2):
    return np.sum(np.abs(p1 - p2))


@jit(nopython=True)
def chebyshev(p1, p2, px, n, ncodes):
    distance = 0.0
    for j in range(px):
        diff = abs(p1[j * n] - p2[j * ncodes])
        if diff > distance:
            distance = diff
    return distance


@jit(nopython=True, parallel=True)
def cosine(p1, p2, px, n, ncodes):
    nom = 0.0
    denom1 = 0.0
    denom2 = 0.0
    for j in range(px):
        nom += p1[j * n] * p2[j * ncodes]
        denom1 += p1[j * n] * p1[j * n]
        denom2 += p2[j * ncodes] * p2[j * ncodes]

    return (-nom / (np.sqrt(denom1) * np.sqrt(denom2))) + 1


@jit(nopython=True, parallel=True)
def SOM(data, codes, nhbrdist, alphas, radii, ncodes, rlen, distf=eucl, seed=None):
    if seed is not None:
        np.random.seed(seed)
    xdists = np.zeros((ncodes))
    n = data.shape[0]
    px = data.shape[1]
    niter = rlen * n
    threshold = radii[0]
    thresholdStep = (radii[0] - radii[1]) / niter
    change = 1.0

    for k in range(niter):
        if k % n == 0:
            if change < 1:
                k = niter
            change = 0.0

        i = np.random.randint(n)

        nearest = 0
        for cd in range(ncodes):
            xdists[cd] = distf(data[i, :], codes[cd, :])
            if xdists[cd] < xdists[nearest]:
                nearest = cd

        if threshold < 1.0:
            threshold = 0.5
        alpha = alphas[0] - (alphas[0] - alphas[1]) * k / niter

        for cd in range(ncodes):
            if nhbrdist[cd, nearest] > threshold:
                continue

            for j in range(px):
                tmp = data[i, j] - codes[cd, j]
                change += abs(tmp)
                codes[cd, j] += tmp * alpha

        threshold -= thresholdStep
    return codes


@jit(nopython=True, parallel=True)
def map_data_to_codes(data, codes, distf=eucl):
    counter = -1
    n_codes = codes.shape[0]
    nd = data.shape[0]
    nn_codes = np.zeros((nd))
    nn_dists = np.zeros((nd))
    for i in range(nd):
        minid = -1
        mindist = np.inf
        for cd in range(n_codes):
            tmp = distf(data[i, :], codes[cd, :])
            if tmp < mindist:
                mindist = tmp
                minid = cd
        counter += 1
        nn_codes[counter] = minid
        nn_dists[counter] = mindist
    return nn_codes, nn_dists
