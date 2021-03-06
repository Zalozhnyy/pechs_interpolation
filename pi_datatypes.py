from typing import Dict, List, Union

from dataclasses import dataclass
from enum import Enum
import numpy as np


class InterpolationMethods(Enum):
    nearest = 0
    n_nearest = 1
    pillar = 2
    flux_translation = 3
    flux_nearest = 4


@dataclass
class Grid:
    x: np.ndarray
    y: np.ndarray
    z: np.ndarray
    array: np.ndarray

    space: np.ndarray = None


@dataclass
class Detector:
    coordinates: np.ndarray
    results: np.ndarray
    layer: int


@dataclass
class UiConfiguration:
    remp_path: str = None
    pechs_path: str = None
    calculation_method: InterpolationMethods = None

    pechs_detectors_families: Dict = None

    count_detectors: int = 0


@dataclass
class DetectorValuesProjections:
    wx: float = 0.
    wy: float = 0.
    wz: float = 0.

    w_x: float = 0.
    w_y: float = 0.
    w_z: float = 0.


@dataclass
class FluxDetector:
    x: float
    y: float
    z: float
    nx: float
    ny: float
    nz: float

    results: np.ndarray
    projections: List[DetectorValuesProjections] = None

    nx_d: List[float] = None
    ny_d: List[float] = None
    nz_d: List[float] = None



class ToManyActiveDetectors(Exception):
    def __init__(self, max_detectors: int):
        self.n_det = max_detectors

    def __str__(self):
        return f'Max detectors count is {self.n_det}'


class CantFindPillarAxe(Exception):
    def __str__(self):
        return f'Не найдена ось симметрии. Исользуйте другой метод.'
