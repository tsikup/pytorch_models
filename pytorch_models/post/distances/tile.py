import numpy as np
import scipy.spatial


def get_tile_coordinates(tile_center, tileSize=512):
    return tuple(np.floor_divide(np.array(tile_center), tileSize - 1))


def single_gaussian_function(x, sigma=10):
    return np.exp(-(1 / (2 * sigma) ** 2) * x)


def spatial_ranking_gaussian_squared_manhattan(point1, point2, sigma=10):
    i1, j1 = point1
    i2, j2 = point2
    return single_gaussian_function((i1 - i2) ** 2 + (j1 - j2) ** 2, sigma=sigma)


def spatial_ranking_gaussian_manhattan(point1, point2, sigma=10):
    return single_gaussian_function(
        scipy.spatial.distance.cityblock(point1, point2), sigma=sigma
    )


def spatial_ranking_gaussian_euclidean(point1, point2, sigma=10):
    return single_gaussian_function(
        scipy.spatial.distance.euclidean(point1, point2), sigma=sigma
    )


def spatial_ranking_gaussian_minkowski(point1, point2, p=3, sigma=10):
    return single_gaussian_function(
        scipy.spatial.distance.minkowski(point1, point2, p=p), sigma=sigma
    )


def qSpat(patch, I, distance=spatial_ranking_gaussian_squared_manhattan, sigma=0.5):
    def qSpatOne(I, tile1, tile2):
        if tuple(tile2) == tuple(tile1):
            return 0.0
        i = I[tile2[0], tile2[1]]
        g = distance(tile1, tile2, sigma=sigma)
        return i * g

    qSpat_sum = 0
    nonZero = np.transpose(I.nonzero())
    for (i, j) in nonZero:
        qSpat_sum = qSpat_sum + qSpatOne(I, patch, (i, j))
    return qSpat_sum
