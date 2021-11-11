from typing import Union

from datatypes import Grid

import numpy as np
import matplotlib.pyplot as plt


class TrilinearInterpolation:
    def __init__(self, grd: Grid, det: Grid):
        self.mesh: Grid = grd
        self.detectors: Grid = det

    @staticmethod
    def _find_nearest(axe: np.ndarray, point: Union[int, float]) -> int:
        if point <= axe[0]:
            return 0

        for i in range(axe.shape[0] - 1):
            if point < axe[i + 1]:
                return i

        return axe.shape[0] - 1

    def _get_cube(self, i, j, k):

        x0 = self.detectors.x[i, 0, 0]
        y0 = self.detectors.y[0, j, 0]
        z0 = self.detectors.z[0, 0, k]
        x1 = self.detectors.x[i + 1, 0, 0]
        y1 = self.detectors.y[0, j + 1, 0]
        z1 = self.detectors.z[0, 0, k + 1]

        corners = [
            self.detectors.array[i, j, k],
            self.detectors.array[i, j + 1, k],
            self.detectors.array[i + 1, j + 1, k],
            self.detectors.array[i + 1, j, k],
            self.detectors.array[i, j, k + 1],
            self.detectors.array[i, j + 1, k + 1],
            self.detectors.array[i + 1, j + 1, k + 1],
            self.detectors.array[i + 1, j, k + 1],
        ]

        return x0, y0, z0, x1, y1, z1, corners

    @staticmethod
    def calc(x, y, z, x0, y0, z0, x1, y1, z1, *corner_values):
        # https://en.wikipedia.org/wiki/Trilinear_interpolation
        c000, c010, c110, c100, c001, c011, c111, c101 = corner_values
        xd = (x - x0) / (x1 - x0)
        yd = (y - y0) / (y1 - y0)
        zd = (z - z0) / (z1 - z0)

        c00 = c000 * (1 - xd) + c100 * xd
        c01 = c001 * (1 - xd) + c101 * xd
        c10 = c010 * (1 - xd) + c110 * xd
        c11 = c011 * (1 - xd) + c111 * xd

        c0 = c00 * (1 - yd) + c10 * yd
        c1 = c01 * (1 - yd) + c11 * yd

        c = c0 * (1 - zd) + c1 * zd
        return c

    def _interpolate(self, result, x0, y0, z0, x1, y1, z1, corners):
        x0i = self._find_nearest(self.mesh.x[:, 0, 0], x0)
        x1i = self._find_nearest(self.mesh.x[:, 0, 0], x1)
        y0i = self._find_nearest(self.mesh.y[0, :, 0], y0)
        y1i = self._find_nearest(self.mesh.y[0, :, 0], y1)
        z0i = self._find_nearest(self.mesh.z[0, 0, :], z0)
        z1i = self._find_nearest(self.mesh.z[0, 0, :], z1)

        for i in range(x0i, x1i + 1):
            x = self.mesh.x[i, 0, 0]

            for j in range(y0i, y1i + 1):
                y = self.mesh.y[0, j, 0]

                for k in range(z0i, z1i + 1):
                    z = self.mesh.z[0, 0, k]

                    result[i, j, k] = self.calc(x, y, z, x0, y0, z0, x1, y1, z1, *corners)

    def interpolation(self):
        result = np.copy(self.mesh.array)

        for k in range(self.detectors.z.shape[2] - 1):
            print(f'{k}/{self.detectors.z.shape[2] - 2}')
            for i in range(self.detectors.x.shape[0] - 1):
                for j in range(self.detectors.y.shape[1] - 1):
                    x0, y0, z0, x1, y1, z1, corners = self._get_cube(i, j, k)
                    self._interpolate(result, x0, y0, z0, x1, y1, z1, corners)

        return result


if __name__ == '__main__':

    xg = yg = zg = np.arange(-10, 10 + 0.01, 0.2)
    xd = yd = zd = np.arange(-10, 10 + 0.01, 1)

    xg, yg, zg = np.meshgrid(xg, yg, zg, indexing='ij')
    xd, yd, zd = np.meshgrid(xd, yd, zd, indexing='ij')

    ar_g = np.zeros((xg.shape[0], xg.shape[1], xg.shape[2]))
    ar_d = (xd ** 2 + yd ** 2 + zd ** 2) ** 0.5

    # ar_d = np.zeros((xd.shape[0], yd.shape[0], yd.shape[2]))
    # for i in range(ar_d.shape[0]):
    #     for j in range(ar_d.shape[0]):
    #         ar_d[i, j, 0] = 1
    #         ar_d[i, j, 1] = 0
    #         ar_d[i, j, 2] = 2

    mesh, detector = Grid(xg, yg, zg, ar_g), Grid(xd, yd, zd, ar_d)

    print(f'GRID | {mesh.array.shape}')
    print(f'DET  | {detector.array.shape}')

    ex = TrilinearInterpolation(mesh, detector)

    res = ex.interpolation()

    for q in range(0, mesh.z.shape[2], 50):
        plt.contourf(mesh.x[:, :, q], mesh.y[:, :, q], res[:, :, q], cmap=plt.cm.bone)
        plt.title(f'k={q}')
        plt.colorbar()
        plt.show()
        print(q)
        # print(res[:, :, q])
