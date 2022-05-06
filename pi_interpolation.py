from typing import Optional, Union, List

import time
import numpy as np
from scipy.spatial import KDTree

import pi_utility
from pi_datatypes import Grid, Detector, ToManyActiveDetectors, CantFindPillarAxe, FluxDetector, InterpolationMethods, \
    DetectorValuesProjections
from pi_utility import find_nearest, print_progress
from pi_interpolation_save import SaveFlux


def log_time(func, *args, **kwargs):
    def wrapper():
        t = time.time()
        res = func(*args, **kwargs)
        print(f'{func.__name__} exec time: {time.time() - t}')
        return res

    return wrapper


class Interpolation:
    def __init__(self, grd: Grid, det: Union[Grid, Detector, List[FluxDetector]]):

        self.mesh: Grid = grd
        self.detectors: Union[Grid, Detector, List[FluxDetector]] = det

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
            print_progress(i + 1, ni)
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
            print_progress(i + 1, ni)

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


class FluxInterpolation(Interpolation):
    def __init__(self, grd: Grid, det: List[FluxDetector], meta):
        super().__init__(grd, det)

        self.meta_data = meta
        self.from_l, self.to_l = [int(i) for i in self.meta_data['name'].split('_')[1:3]]

        self.xc, self.yc, self.zc, self._xc, self._yc, self._zc = [], [], [], [], [], []  # cell
        self.xd, self.yd, self.zd, self._xd, self._yd, self._zd = [], [], [], [], [], []  # detectors
        self.xr, self.yr, self.zr, self._xr, self._yr, self._zr = [], [], [], [], [], []  # results

    def nearest(self):
        yield

        self._find_detectors()
        self._find_mesh_points()

        yield

        self._interpolate_nearest(self.xc, self.xd, self.xr)
        self._interpolate_nearest(self.yc, self.yd, self.yr)
        self._interpolate_nearest(self.zc, self.zd, self.zr)
        self._interpolate_nearest(self._xc, self._xd, self._xr)
        self._interpolate_nearest(self._yc, self._yd, self._yr)
        self._interpolate_nearest(self._zc, self._zd, self._zr)

        yield

        self._save()

    def n_nearest(self):
        yield

        self._find_detectors()
        self._find_mesh_points()

        yield

        self._interpolate_n_nearest(self.xc, self.xd, self.xr)
        self._interpolate_n_nearest(self.yc, self.yd, self.yr)
        self._interpolate_n_nearest(self.zc, self.zd, self.zr)
        self._interpolate_n_nearest(self._xc, self._xd, self._xr)
        self._interpolate_n_nearest(self._yc, self._yd, self._yr)
        self._interpolate_n_nearest(self._zc, self._zd, self._zr)

        yield

        self._save()

    def _save(self):
        res = [self.xr, self.yr, self.zr, self._xr, self._yr, self._zr]
        det = [self.xd, self.yd, self.zd, self._xd, self._yd, self._zd]
        dim = [1, 3, 5, 0, 2, 4]
        arg_ = ['wx', 'wy', 'wz', 'w_x', 'w_y', 'w_z']
        dim2 = [1, 2, 3, -1, -2, -3]

        for r, d, _dim1, _arg, _dim2 in zip(res, det, dim, arg_, dim2):
            if len(r) > 0:
                SaveFlux(r, **self.meta_data).save_one_dim(_dim1, _arg, _dim2)
            else:
                print(f'Для спектра по {_arg} не найдены необходимые точки в файле CELL. ')
                SaveFlux(d, **self.meta_data).save_one_dim(_dim1, _arg, _dim2)

    def _find_mesh_points(self):

        for i in range(self.mesh.space.shape[0]):
            for j in range(self.mesh.space.shape[1]):
                for k in range(self.mesh.space.shape[2]):

                    if i < self.mesh.space.shape[0] - 1 \
                            and self.mesh.space[i, j, k] == self.from_l \
                            and self.mesh.space[i + 1, j, k] == self.to_l:
                        self.xc.append((i, j, k))

                    if i > 0 and self.mesh.space[i, j, k] == self.from_l \
                            and self.mesh.space[i - 1, j, k] == self.to_l:
                        self._xc.append((i, j, k))

                    if j < self.mesh.space.shape[1] - 1 \
                            and self.mesh.space[i, j, k] == self.from_l \
                            and self.mesh.space[i, j + 1, k] == self.to_l:
                        self.yc.append((i, j, k))

                    if j > 0 and self.mesh.space[i, j, k] == self.from_l \
                            and self.mesh.space[i, j - 1, k] == self.to_l:
                        self._yc.append((i, j, k))

                    if k < self.mesh.space.shape[2] - 1 \
                            and self.mesh.space[i, j, k] == self.from_l \
                            and self.mesh.space[i, j, k + 1] == self.to_l:
                        self.zc.append((i, j, k))

                    if k > 0 and self.mesh.space[i, j, k] == self.from_l \
                            and self.mesh.space[i, j, k - 1] == self.to_l:
                        self._zc.append((i, j, k))

    def _find_detectors(self):

        for detector in self.detectors:

            if detector.nx > 0:
                self.xd.append(detector)
            if detector.nx < 0:
                self._xd.append(detector)
            if detector.ny > 0:
                self.yd.append(detector)
            if detector.ny < 0:
                self._yd.append(detector)
            if detector.nz > 0:
                self.zd.append(detector)
            if detector.nz < 0:
                self._zd.append(detector)

    def _get_axe_name(self, ax):

        if ax is self.xc:
            return 'x', 'wx'
        elif ax is self.yc:
            return 'y', 'wy'
        elif ax is self.zc:
            return 'z', 'wz'
        elif ax is self._xc:
            return '-x', 'w_x'
        elif ax is self._yc:
            return '-y', 'w_y'
        elif ax is self._zc:
            return '-z', 'w_z'

    def _n_nearest_calculations(self, distances, ii, k_, values):

        energies = self.meta_data['template']['energies']
        distances_set = set(distances)

        max_distance = np.max(distances)
        weights = 1. - distances / max_distance

        if len(distances_set) > 1:
            weights = weights / np.sum(weights)
        else:
            weights = np.array([1 / k_ for _ in range(weights.shape[0])])

        detectors = [values[i] for i in ii]

        nx = sum([det.x * w for det, w in zip(detectors, weights)])
        ny = sum([det.y * w for det, w in zip(detectors, weights)])
        nz = sum([det.z * w for det, w in zip(detectors, weights)])
        nx, ny, nz = pi_utility.norm_vector(nx, ny, nz)

        projections = []
        for i in range(len(energies)):
            p = DetectorValuesProjections()
            for _attr in DetectorValuesProjections.__annotations__.keys():
                value = sum([det.projections[i].__getattribute__(_attr) * w for det, w in zip(detectors, weights)])
                p.__setattr__(_attr, value)
            projections.append(p)

        if self.meta_data['measure'] == 'DETAILED':
            nx_d, ny_d, nz_d = [], [], []

            for i in range(len(energies)):
                nx_d.append(sum([det.nx_d[i] * w for det, w in zip(detectors, weights)]))
                ny_d.append(sum([det.ny_d[i] * w for det, w in zip(detectors, weights)]))
                nz_d.append(sum([det.nz_d[i] * w for det, w in zip(detectors, weights)]))

                nx_d[i], ny_d[i], nz_d[i] = pi_utility.norm_vector(nx_d[i], ny_d[i], nz_d[i])

            return FluxDetector(0., 0., 0., nx, ny, nz, None, projections, nx_d, ny_d, nz_d)

        elif self.meta_data['measure'] == 'BASIC':
            return FluxDetector(0., 0., 0., nx, ny, nz, None, projections)

    def _interpolate_nearest(self, cells, detectors: List[FluxDetector], result: List):

        if len(detectors) == 0:
            return

        tree = KDTree(
            np.array([[det.x, det.y, det.z] for det in detectors])
        )

        for i, j, k in cells:
            distances, ii = tree.query([self.mesh.x[i], self.mesh.y[j], self.mesh.z[k]], k=1)

            d = FluxDetector(self.mesh.x[i], self.mesh.y[j], self.mesh.z[k],
                             detectors[ii].nx, detectors[ii].ny, detectors[ii].nz,
                             detectors[ii].results, detectors[ii].projections,
                             detectors[ii].nx_d, detectors[ii].ny_d, detectors[ii].nz_d, )

            result.append(d)

    def _interpolate_n_nearest(self, cells, detectors: List[FluxDetector], result: List):

        if len(detectors) == 0:
            return

        axe, attr = self._get_axe_name(cells)
        if self.meta_data['n'] <= len(detectors):
            k_ = self.meta_data['n']
        else:
            print(f'Недостаточно детекторов по оси {axe}.'
                  f' Количество действующих детекторов уменьшено до {len(detectors)}')
            k_ = len(detectors)

        tree = KDTree(np.array([[det.x, det.y, det.z] for det in detectors]))

        for i, j, k in cells:
            distances, ii = tree.query([self.mesh.x[i], self.mesh.y[j], self.mesh.z[k]], k=k_)

            if 0 <= distances[0] <= 1e-6:
                result.append(detectors[ii])
                continue

            detector = self._n_nearest_calculations(distances, ii, k_, detectors)
            detector.x, detector.y, detector.z = self.mesh.x[i], self.mesh.y[j], self.mesh.z[k]
            result.append(detector)
