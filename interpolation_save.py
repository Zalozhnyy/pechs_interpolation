import os

from datatypes import Grid, InterpolationMethods


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
