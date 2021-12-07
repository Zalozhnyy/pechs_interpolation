import os
from collections import defaultdict

import numpy as np

import matplotlib.pyplot as plt

from pi_interpolation import NearestInterpolation, PillarInterpolation, FluxInterpolation
from pi_datatypes import Grid, Detector, InterpolationMethods
import pi_utility
import pi_datatypes
from read_REMP import read_REMP
from pi_interpolation_save import Save, SaveFlux


class DataGenerator:

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

    @classmethod
    def process_pechs_flux_basic(cls, lst: str, res: str):
        data = np.loadtxt(lst, dtype=float)
        results = np.loadtxt(res, dtype=float)  # basic results

        detectors = []
        for d, r in zip(data, results):
            detector = pi_datatypes.FluxDetector(*d[:6], results=r)
            norm = (detector.nx ** 2 + detector.ny ** 2 + detector.nz ** 2) ** 0.5

            detector.nx /= norm
            detector.ny /= norm
            detector.nz /= norm

            detector.projections = pi_utility.get_projection_basic(detector)
            detectors.append(detector)

        return detectors

    @classmethod
    def process_pechs_flux_detailed(cls, lst: str, res: str, en_count: int):

        with open(lst, 'r', encoding='utf-8') as lf, open(res, 'r', encoding='utf-8') as rf:
            coordinates = lf.readlines()
            res_data = np.array([i.strip().split() for i in rf.readlines() if i != '\n'], dtype=float)

            if len(coordinates) == res_data.shape[0]:
                raise Exception(f'Результаты не похожи на детальный вывод. {os.path.basename(res)}')

            res_pointer = 0
            detectors = []

            for det in range(len(coordinates)):
                x, y, z, nx, ny, nz, r = [float(i) for i in coordinates[det].strip().split()]
                nx, ny, nz = pi_utility.norm_vector(nx, ny, nz)

                res_slice = res_data[res_pointer:res_pointer + en_count]
                res_pointer += en_count

                nx_d = res_slice[:, 2]
                ny_d = res_slice[:, 3]
                nz_d = res_slice[:, 4]
                results = res_slice[:, 1]

                for i in range(len(nx_d)):
                    nx_d[i], ny_d[i], nz_d[i] = pi_utility.norm_vector(nx_d[i], ny_d[i], nz_d[i])

                projections = []
                for j in range(results.shape[0]):
                    projections.append(pi_utility.get_projection_detailed((nx_d[i], ny_d[i], nz_d[i]), results[j]))

                detectors.append(
                    pi_datatypes.FluxDetector(
                        x,
                        y,
                        z,
                        nx,
                        ny,
                        nz,
                        results,
                        projections,
                        nx_d.tolist(),
                        ny_d.tolist(),
                        nz_d.tolist(),
                    )
                )

            return detectors


class Calculations:
    def _coroutine(self, *args, **kwargs):
        mesh, detectors = args
        pb, gen, v = self._pick_interpolation_method(mesh, detectors, **kwargs)

        while True:
            try:
                next(gen)
                pb['value'] += v
                kwargs['widget'].update_idletasks()
            except StopIteration:
                print(f'Завершён расчет {kwargs["name"]}')
                break

    def _pick_interpolation_method(self, mesh, detectors, **kwargs):
        pb = kwargs['progress_bar']
        pb.configure(maximum=mesh.x.shape[0])

        if kwargs['type'] != 'FLUX':

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

        elif kwargs['type'] == 'FLUX':

            if kwargs['method'] == InterpolationMethods.flux_translation:
                gen = SaveFlux(detectors, **kwargs).translation_save()
                pb.configure(maximum=6)
                v = 1

            elif kwargs['method'] == InterpolationMethods.nearest:
                gen = FluxInterpolation(mesh, detectors, kwargs).nearest()
                pb.configure(maximum=3)
                v = 1

            elif kwargs['method'] == InterpolationMethods.n_nearest:
                gen = FluxInterpolation(mesh, detectors, kwargs).n_nearest()
                pb.configure(maximum=3)
                v = 1

            else:
                raise Exception('Unknown interpolation method')

        else:
            raise Exception('Unknown interpolation method')

        return pb, gen, v

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

    def _calculate_flux(self, *args, **kwargs):
        _, grid, space, _, _, _ = read_REMP(kwargs['remp_dir'])
        mesh = GridProcessing.process_remp(grid[0]['i05'], grid[1]['i05'], grid[2]['i05'], space)

        if kwargs['measure'] == 'BASIC':
            detectors = GridProcessing.process_pechs_flux_basic(kwargs['lst'], kwargs['res'])
        elif kwargs['measure'] == 'DETAILED':
            detectors = GridProcessing.process_pechs_flux_detailed(kwargs['lst'],
                                                                   kwargs['res'],
                                                                   len(kwargs['template']['energies']))
        else:
            raise Exception('unknown measure type')

        self._coroutine(mesh, detectors, **kwargs)

    def calculate(self, *args, **kwargs):
        if kwargs['type'] == 'ENERGY':
            self._calculate_energy(**kwargs)

        elif kwargs['type'] == 'CURRENT':
            self._calculate_current(**kwargs)

        elif kwargs['type'] == 'FLUX':
            self._calculate_flux(**kwargs)

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

    project_path = r'C:\Users\\niczz\Dropbox\work_cloud\projects\project_test_interpolation'

    lst_file = r'C:\Users\niczz\Dropbox\work_cloud\projects\project_test_interpolation\pechs\pe_flux\initials\flux_2_4_1.lst'
    res_file = r'C:\Users\niczz\Dropbox\work_cloud\projects\project_test_interpolation\pechs\pe_flux\results\flux_2_4_1.res'

    _, grid, space, _, _, _ = read_REMP(project_path)

    mesh = GridProcessing.process_remp(grid[0]['i05'], grid[1]['i05'], grid[2]['i05'], space)

    detectors = GridProcessing.process_pechs_flux_detailed(lst_file, res_file, 5)
    # detectors = GridProcessing.process_pechs_flux_basic(lst_file, res_file)
    # detectors = GridProcessing.process_pechs_current(lst_file, res_file, 4, 'z')

    # mesh, detectors = grd_creator.create_pillar()
    print(f'GRID | {mesh.array.shape}')
    meta = {
        'name': 'flux_2_4_1',
        'template': {
            'energies': [50, 150, 250, 350, 450],
        },
        'measure': 'DETAILED',
        'remp_dir': project_path,
        'n': 3,
    }
    # gen = PillarInterpolation(mesh, detectors).pillar()
    gen = FluxInterpolation(mesh, detectors, meta).nearest()

    while True:
        try:
            next(gen)
        except StopIteration:
            print(f'Завершён расчет')
            break
        except Exception('Error while calculation') as e:
            print(e)
            break
    exit(0)

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
