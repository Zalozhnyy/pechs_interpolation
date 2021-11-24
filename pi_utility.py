import numpy as np

from collections import defaultdict
from typing import Union, List
import pi_datatypes


def find_nearest(axe: np.ndarray, point: Union[int, float]) -> int:
    for i in range(axe.shape[0] - 1):
        if round(axe[i], 6) == point:
            return i
        elif round(axe[i], 6) < point < round(axe[i + 1], 6):
            return i
    return axe.shape[0] - 1


def find_nearest_binary(axe: np.ndarray, point: Union[int, float]) -> int:
    l, r = 0, axe.shape[0] - 1

    while l < r:

        mid = (l + r) // 2

        if axe[mid] < point:
            l = mid + 1
        else:
            r = mid

    if l == 0:
        return 0
    elif l == axe.shape[0] - 1:
        return axe.shape[0] - 1
    elif axe[l] <= point < axe[l + 1]:
        return l
    elif axe[l + 1] <= point < axe[l + 2]:
        return l + 1
    else:
        return l - 1


def get_projection_basic(detector: pi_datatypes.FluxDetector) -> List[pi_datatypes.DetectorValuesProjections]:
    arr = defaultdict(np.ndarray)

    def foo(vector, axe):
        if vector > 0:
            arr[axe] = detector.results[:] * vector
            arr['_' + axe] = detector.results[:] * 0
        elif vector < 0:
            arr['_' + axe] = detector.results[:] * abs(vector)
            arr[axe] = detector.results[:] * 0
        else:
            arr['_' + axe] = detector.results[:] * 0
            arr[axe] = detector.results[:] * 0

    foo(detector.nx, 'nx')
    foo(detector.ny, 'ny')
    foo(detector.nz, 'nz')

    projections = []

    for i in range(detector.results.shape[0]):
        projections.append(
            pi_datatypes.DetectorValuesProjections(
                arr['nx'][i],
                arr['ny'][i],
                arr['nz'][i],
                arr['_nx'][i],
                arr['_ny'][i],
                arr['_nz'][i],
            )
        )

    return projections


def get_projection_detailed(vector, value) -> pi_datatypes.DetectorValuesProjections:
    nx, ny, nz = vector

    arr = defaultdict(float)

    def foo(v, axe, value_):
        if v > 0:
            arr[axe] = value_ * v
            arr['_' + axe] = value_ * 0
        elif v < 0:
            arr['_' + axe] = value_ * abs(v)
            arr[axe] = value_ * 0
        else:
            arr['_' + axe] = value_ * 0
            arr[axe] = value_ * 0

    foo(nx, 'nx', value)
    foo(ny, 'ny', value)
    foo(nz, 'nz', value)

    return pi_datatypes.DetectorValuesProjections(
        arr['nx'],
        arr['ny'],
        arr['nz'],
        arr['_nx'],
        arr['_ny'],
        arr['_nz'],
    )
