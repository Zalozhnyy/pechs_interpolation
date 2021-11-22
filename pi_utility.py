import numpy as np

from typing import Union


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
