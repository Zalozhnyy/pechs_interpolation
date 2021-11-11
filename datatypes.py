from typing import Dict

from dataclasses import dataclass
from enum import Enum
import numpy as np


class InterpolationMethods(Enum):
    nearest = 0
    n_nearest = 1
    pillar = 2


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
