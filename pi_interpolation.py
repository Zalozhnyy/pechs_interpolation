from typing import Optional, Union

import numpy as np
from scipy.spatial import KDTree

from pi_datatypes import Grid, Detector, ToManyActiveDetectors, CantFindPillarAxe
from pi_utility import find_nearest


class Interpolation:
    def __init__(self, grd: Grid, det: Union[Grid, Detector]):

        self.mesh: Grid = grd
        self.detectors: Union[Grid, Detector] = det

    def _allow_interpolation(self, k):
        if isinstance(self.detectors, Grid):
            values = self.detectors.array.ravel()

        elif isinstance(self.detectors, Detector):
            values = self.detectors.results

        else:
            raise TypeError('Unknown detector data structure')

        if k > values.shape[0]:
            raise ToManyActiveDetectors(values.shape[0])

        elif k == values.shape[0]:
            return True, k - 1

        else:
            return True, k

    def _get_tree(self):
        if isinstance(self.detectors, Grid):
            values = self.detectors.array.ravel()
            tree = KDTree(np.c_[self.detectors.x.ravel(), self.detectors.y.ravel(), self.detectors.z.ravel()])

        elif isinstance(self.detectors, Detector):
            values = self.detectors.results
            tree = KDTree(self.detectors.coordinates)

        else:
            raise TypeError('Unknown detector data structure')

        return values, tree


class NearestInterpolation(Interpolation):
    def nearest(self):
        result = self.mesh.array
        values, tree = self._get_tree()
        ni, nj, nk = self.mesh.x.shape[0], self.mesh.y.shape[0], self.mesh.z.shape[0],

        for i in range(ni):
            if (i + 1) % 10 == 0 or i == ni - 1:
                print(f'Progress {i + 1}/{ni}')
            yield
            for j in range(nj):
                for k in range(nk):
                    if self.mesh.space[i, j, k] != self.detectors.layer:
                        continue
                    dd, ii = tree.query([self.mesh.x[i], self.mesh.y[j], self.mesh.z[k]], k=1)
                    result[i, j, k] = values[ii]

    def _k_nearest_calculations(self, distances, ii, k_, values):
        distances_set = set(distances)

        max_distance = np.max(distances)
        weights = 1. - distances / max_distance

        if len(distances_set) > 1:
            weights = weights / np.sum(weights)
        else:
            weights = np.array([1 / k_ for _ in range(weights.shape[0])])

        return np.sum([original_value * weights[v] for v, original_value in enumerate(values[ii])])

    def k_nearest(self, k_: int):
        if k_ < 1:
            raise Exception('k must be bigger that 1')
        k_ += 1

        flag, k_ = self._allow_interpolation(k_)
        result = self.mesh.array
        values, tree = self._get_tree()

        ni, nj, nk = self.mesh.x.shape[0], self.mesh.y.shape[0], self.mesh.z.shape[0],

        for i in range(ni):
            if (i + 1) % 10 == 0 or i == ni - 1:
                print(f'Progress {i + 1}/{ni}')
            yield
            for j in range(nj):
                for k in range(nk):
                    if self.mesh.space[i, j, k] != self.detectors.layer:
                        continue

                    distances, ii = tree.query([self.mesh.x[i], self.mesh.y[j], self.mesh.z[k]], k=k_)

                    if 0 <= distances[0] <= 1e-6:
                        result[i, j, k] = values[ii][0]
                        continue

                    result[i, j, k] = self._k_nearest_calculations(distances, ii, k_, values)


class PillarInterpolation(Interpolation):
    def __init__(self, grd: Grid, det: Union[Grid, Detector]):
        super().__init__(grd, det)

        self._arr_ax_slices = {
            0: lambda index_, array: array[index_, :, :],
            1: lambda index_, array: array[:, index_, :],
            2: lambda index_, array: array[:, :, index_],
        }

        self._space_ax_slices = {
            0: lambda index_: self.mesh.space[index_, :, :],
            1: lambda index_: self.mesh.space[:, index_, :],
            2: lambda index_: self.mesh.space[:, :, index_],
        }

        self._grd_dict = {
            0: self.mesh.x,
            1: self.mesh.y,
            2: self.mesh.z,
        }

    def _get_detectors_axis(self):
        """получаем ось вдоль которой расположены детекторы"""
        x = len(set(self.detectors.coordinates[:, 0]))
        y = len(set(self.detectors.coordinates[:, 1]))
        z = len(set(self.detectors.coordinates[:, 2]))

        if x != 1 and y == 1 and z == 1:
            return 0
        elif x == 1 and y != 1 and z == 1:
            return 1
        elif x == 1 and y == 1 and z != 1:
            return 2
        else:
            raise CantFindPillarAxe()

    def _fill_results(self, array: np.ndarray, space: np.ndarray, value: float):

        for i in range(array.shape[0]):
            for j in range(array.shape[1]):
                if space[i, j] == self.detectors.layer:
                    array[i, j] = value

    def _sort_detectors(self, ax):
        coord_and_res = np.c_[self.detectors.coordinates, self.detectors.results]
        coord_and_res = np.array(sorted(coord_and_res.tolist(), key=lambda x: x[ax]))
        self.detectors.coordinates = coord_and_res[:, :3]
        self.detectors.results = coord_and_res[:, -1]

    def pillar(self):
        assert isinstance(self.detectors, Detector)

        result = self.mesh.array
        ax = self._get_detectors_axis()
        self._sort_detectors(ax)

        fr = lambda value, index: self._fill_results(
            self._arr_ax_slices[ax](index, result),
            self._space_ax_slices[ax](index),
            value)

        for k_d in range(self.detectors.coordinates.shape[0] - 1):
            yield

            y2 = self.detectors.results[k_d + 1]
            x2 = self.detectors.coordinates[k_d + 1, ax]
            y1 = self.detectors.results[k_d]
            x1 = self.detectors.coordinates[k_d, ax]

            k1 = find_nearest(self._grd_dict[ax], x1)
            k2 = find_nearest(self._grd_dict[ax], x2)

            for k in range(k1, k2 + 1):
                x = round(self._grd_dict[ax][k], 6)
                if x1 <= x <= x2:
                    v = y1 + (x - x1) * (y2 - y1) / (x2 - x1)
                    fr(v, k)

                # elif self._grd_dict[ax][k] < x1:
                #     fr(y1, k)
                #
                # elif self._grd_dict[ax][k] > x2:
                #     fr(y2, k)

        return result
