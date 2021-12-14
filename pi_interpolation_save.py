import os
from copy import copy
from typing import List

from pi_datatypes import Grid, InterpolationMethods
import pi_datatypes


class Save:

    def save_remp(self, mesh: Grid, remp_path, name):

        ni, nj, nk = mesh.array.shape

        print(f'Начато сохранение {name}')

        with open(os.path.join(remp_path, name), 'w', encoding='utf-8') as f:
            for i in range(ni):
                for j in range(nj):
                    for k in range(nk):
                        if mesh.array[i, j, k] != 0:

                            if 'en' in name and mesh.array[i, j, k] < 0:
                                continue

                            f.write(f'{i} {j} {k} {mesh.array[i, j, k]}\n')

        print(f'Сохранение {name} завершено')


class SaveFlux:

    def __init__(self, detectors: List[pi_datatypes.FluxDetector], **kwargs):
        self.detectors: List[pi_datatypes.FluxDetector] = detectors

        self.meta_data = kwargs
        self.meta_data['energy'] = kwargs['template']['energies']

        self.TEMPLATE_CAP = '''{particle_name:s}
        Spectre number
        {spc_number}
        Spectre power (cnt/cm**2/s)- surface, (cnt/cm**3/s)- volumeric
        1
        Spectr type (0-fixed, 1-random, 3-detectors, 31-detectors interpolated)
        {spectr_type}
        Particle count
        {en_count}
        Detectors count
        {detectors_count}
        Energy + normal'''

    def translation_save(self):
        generators = (
            self._saver(1, 'wx', 1),
            self._saver(3, 'wy', 2),
            self._saver(5, 'wz', 3),
            self._saver(0, 'w_x', -1),
            self._saver(2, 'w_y', -2),
            self._saver(4, 'w_z', -3),
        )

        for generator in generators:
            self._exec_generator(generator)
            yield

    def save_one_dim(self, indx: int, attr, direction):
        generator = self._saver(indx, attr, direction)
        self._exec_generator(generator)

    def _exec_generator(self, generator):
        while True:
            try:
                next(generator)
            except StopIteration:
                break

    def _saver(self, indx: int, attr, direction):
        from_, to_, particle = list(map(int, self.meta_data['name'].split('_')[1:]))

        fname = '_'.join(list(map(str, [from_, to_, indx, particle]))) + '.spc'

        sp_number = f'{int(from_):0>2d}{int(to_):0>2d}{indx}{particle}'
        spectr_type = 3 if self.meta_data['method'] == InterpolationMethods.flux_translation else 31

        lines = copy(self.TEMPLATE_CAP).format(
            particle_name='electron',
            spc_number=sp_number,
            en_count=len(self.meta_data['energy']),
            detectors_count=len(self.detectors),
            spectr_type=spectr_type
        ).split('\n')
        lines = [i.replace(' ', '') for i in lines]

        spaces = ' ' * 4

        yield

        for detector in self.detectors:
            lines.append(f'{detector.x} {detector.y} {detector.z} {direction}')

            for i, projection in enumerate(detector.projections):

                if self.meta_data['measure'] == 'BASIC':
                    p = projection.__getattribute__(attr)
                    lines.append(
                        self._format_string(
                            self.meta_data['energy'][i], detector.nx, detector.ny, detector.nz, p, spaces)
                    )

                elif self.meta_data['measure'] == 'DETAILED':
                    p = projection.__getattribute__(attr)

                    lines.append(
                        self._format_string(
                            self.meta_data['energy'][i], detector.nx_d[i], detector.ny_d[i], detector.nz_d[i], p,
                            spaces)
                    )

                else:
                    raise Exception('unknown measure type')

            yield

        with open(os.path.join(self.meta_data['remp_dir'], fname), 'w', encoding='utf-8') as f:

            for line in lines:
                f.write(line + '\n')

        print(f'Save {fname} done')
        yield

    def _format_string(self, en, nx, ny, nz, value, spaces):

        s = f"{en * 1e-3:15.8E}{spaces}" \
            f"{nx:10.5E}{spaces}" \
            f"{ny:10.5E}{spaces}" \
            f"{nz:10.5E}{spaces}" \
            f"{value:15.8E}{spaces}"

        return s
