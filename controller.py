import os

import numpy as np

import matplotlib.pyplot as plt

from interpolation import NearestInterpolation, PillarInterpolation
from datatypes import Grid, Detector, InterpolationMethods, ToManyActiveDetectors
from read_REMP import read_REMP
from interpolation_save import Save


class DataCreator:

    @classmethod
    def create_data(cls):
        delta = 2.5
        x0d = 10
        xd = yd = zd = np.arange(-x0d, x0d + 1, delta)

        x0g = 10
        xg = yg = zg = np.arange(-x0g, x0g + 0.01, 0.5)

        xg, yg, zg = np.meshgrid(xg, yg, zg, indexing='ij')
        xd, yd, zd = np.meshgrid(xd, yd, zd, indexing='ij')

        ar_g = np.zeros((xg.shape[0], xg.shape[1], xg.shape[2]))
        ar_d = (xd ** 2 + yd ** 2 + zd ** 2) ** 0.5

        return Grid(xg, yg, zg, ar_g), Grid(xd, yd, zd, ar_d)

    @classmethod
    def create_data_2d(cls):
        delta = 2.5
        x0d = 10
        xd = yd = np.arange(-x0d, x0d + 1, delta)

        x0g = 10
        xg = yg = np.arange(-x0g, x0g + 0.01, 0.5)

        xg, yg = np.meshgrid(xg, yg, indexing='ij')
        xd, yd = np.meshgrid(xd, yd, indexing='ij')

        ar_g = np.zeros((xg.shape[0], xg.shape[1]))
        ar_d = (xd ** 2 + yd ** 2) ** 0.5

        return Grid(xg, yg, None, ar_g), Grid(xd, yd, None, ar_d)

    @classmethod
    def create_pillar(cls):
        x0g = 10
        xg = yg = zg = np.arange(0, x0g + 0.01, 0.5)
        xg, yg, zg = np.meshgrid(xg, yg, zg, indexing='ij')
        ar_g = np.zeros((xg.shape[0], xg.shape[1], xg.shape[2]))

        zd = np.arange(0, 10 + 0.01, 2)
        xd = np.full_like(zd, 5)
        yd = np.full_like(zd, 5)

        lst = np.c_[xd, yd, zd]
        res = zd ** 2

        return Grid(xg, yg, zg, ar_g), Detector(lst, res)


class GridProcessing:

    @classmethod
    def process_remp(cls, x: list, y: list, z: list, space):
        xg, yg, zg = np.array(x), np.array(y), np.array(z),
        ar_g = np.zeros((xg.shape[0], yg.shape[0], zg.shape[0]))
        return Grid(xg, yg, zg, ar_g, space)

    @classmethod
    def process_pechs(cls, res_root: str, ini_root: str, filename: str):
        coords = np.loadtxt(os.path.join(ini_root, filename + '.lst'), dtype=float)[:, :3]
        res = np.loadtxt(os.path.join(res_root, filename + '.res'), dtype=float)
        layer = int(filename.split('_')[-2])
        return Detector(coords, res, layer)

    @classmethod
    def process_pechs_direct_files(cls, lst: str, res: str, layer: int):
        coords = np.loadtxt(lst, dtype=float)[:, :3]
        res = np.loadtxt(res, dtype=float)
        return Detector(coords, res, layer)

    @classmethod
    def process_pechs_current(cls, lst: str, res: str, layer: int, direction: str):
        assert direction in ('x', 'y', 'z')
        coords = np.loadtxt(lst, dtype=float)[:, :3]
        ax = {'x': 0, 'y': 1, 'z': 2}
        results = np.loadtxt(res, dtype=float)[:, ax[direction]]
        return Detector(coords, results, layer)


class Calculations:
    def _coroutine(self, *args, **kwargs):
        mesh, detectors = args
        pb = kwargs['progress_bar']
        pb.configure(maximum=mesh.x.shape[0])

        if kwargs['method'] == InterpolationMethods.n_nearest:
            gen = NearestInterpolation(mesh, detectors).k_nearest(kwargs['n'])
            v = pb['length'] // mesh.x.shape[0]

        elif kwargs['method'] == InterpolationMethods.nearest:
            gen = NearestInterpolation(mesh, detectors).nearest()
            v = pb['length'] // mesh.x.shape[0]

        elif kwargs['method'] == InterpolationMethods.pillar:
            gen = PillarInterpolation(mesh, detectors).pillar()
            v = pb['length'] // detectors.coordinates.shape[0]

        else:
            raise Exception('Unknown interpolation method')

        while True:
            try:
                next(gen)
                pb['value'] += v
                kwargs['widget'].update_idletasks()
            except StopIteration:
                print(f'Завершён расчет {kwargs["name"]}')
                break

    def _calculate_current(self, *args, **kwargs):
        layer = int(kwargs['name'].split('_')[1])
        _, grid, space, _, _, _ = read_REMP(kwargs['remp_dir'])

        cases = {
            'x': lambda: GridProcessing.process_remp(grid[0]['i05'], grid[1]['i'], grid[2]['i'], space),
            'y': lambda: GridProcessing.process_remp(grid[0]['i'], grid[1]['i05'], grid[2]['i'], space),
            'z': lambda: GridProcessing.process_remp(grid[0]['i'], grid[1]['i'], grid[2]['i05'], space),
        }

        for case in cases.keys():
            detectors = GridProcessing.process_pechs_current(kwargs['lst'], kwargs['res'], layer, case)
            mesh = cases[case]()

            kwargs['progress_bar']['value'] = 0
            self._coroutine(mesh, detectors, **kwargs)

            filename = 'j' + case + '_' + '_'.join(kwargs['name'].split('_')[1:])
            Save().save_remp(mesh, kwargs['remp_dir'], filename)

    def _calculate_energy(self, *args, **kwargs):
        layer = int(kwargs['name'].split('_')[1])
        _, grid, space, _, _, _ = read_REMP(kwargs['remp_dir'])

        mesh = GridProcessing.process_remp(grid[0]['i05'], grid[1]['i05'], grid[2]['i05'], space)
        detectors = GridProcessing.process_pechs_direct_files(kwargs['lst'], kwargs['res'], layer)

        self._coroutine(mesh, detectors, **kwargs)
        Save().save_remp(mesh, kwargs['remp_dir'], 'en_' + '_'.join(kwargs['name'].split('_')[1:]))

    def calculate(self, *args, **kwargs):
        if kwargs['type'] == 'ENERGY':
            self._calculate_energy(**kwargs)
        elif kwargs['type'] == 'CURRENT':
            self._calculate_current(**kwargs)
        else:
            raise Exception(f'Unsupported type {kwargs["type"]}')


def debug_plot(mesh: Grid):
    for k in range(0, mesh.y.shape[0], 10):
        # for k in range(55, 115, 5):
        plt.contourf(mesh.x, mesh.z, mesh.array[:, k, :], cmap=plt.cm.bone)
        plt.title(f'k={k}')
        plt.colorbar()
        plt.show()


def main():
    # mesh, detectors = DataCreator.create_data_2d()

    project_path = r'C:\Users\niczz\Dropbox\work_cloud\projects\grant_project'

    lst_file = r'C:\Users\niczz\Dropbox\work_cloud\projects\grant_project\pechs\pechs_box\initials\energy_4_1.lst'
    res_file = r'C:\Users\niczz\Dropbox\work_cloud\projects\grant_project\pechs\pechs_box\results\energy_4_1.res'

    _, grid, space, _, _, _ = read_REMP(project_path)

    mesh = GridProcessing.process_remp(grid[0]['i05'], grid[1]['i05'], grid[2]['i05'], space)

    detectors = GridProcessing.process_pechs_direct_files(lst_file, res_file, 4)
    # detectors = GridProcessing.process_pechs_current(lst_file, res_file, 4, 'z')

    # mesh, detectors = grd_creator.create_pillar()
    print(f'GRID | {mesh.array.shape}')

    # gen = PillarInterpolation(mesh, detectors).pillar()
    gen = NearestInterpolation(mesh, detectors).k_nearest(2)

    while True:
        try:
            next(gen)
        except StopIteration:
            print(f'Завершён расчет')
            break
        except Exception('Error while calculation') as e:
            print(e)
            break

    # for i, (a, v) in enumerate(
    #         zip(mesh.array[10, 10, 121:121 - detectors.results.shape[0]:-1][::-1], detectors.results)):
    #     print(f'det={v:10f}  arr={a:10f}  {round(v, 6) == round(a, 6)}  {detectors.coordinates[i, 2]}')
    # for k in range(0, mesh.y.shape[1], 20):
    for k in range(110, 121, 1):
        plt.contourf(mesh.x, mesh.y, mesh.array[:, :, k], cmap=plt.cm.bone)
        plt.title(f'k={k}')
        plt.colorbar()
        plt.show()


if __name__ == '__main__':
    main()
