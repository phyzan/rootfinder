import numpy as np
import numpy.typing as npt


class RootResult:

    @property
    def root(self)->npt.NDArray[np.float64]:...

    @property
    def iters(self)->npt.NDArray[np.int64]:...

    @property
    def success(self)->npt.NDArray[np.bool]:...


class _LowLevelSolver:

    def __init__(self, f, jac, nsys, nargs):...

    def _newton_raphson(self, x0: np.ndarray, args: np.ndarray, dataset_dims: int, ftol: float, xtol: float, max_iter: int)->RootResult:...


class _LowLevelSolver1D(_LowLevelSolver):

    def __init__(self, f, jac, nargs):...

    def bisect(self, a: np.ndarray, b: np.ndarray, args: np.ndarray, ftol=1e-8, xtol=1e-8, max_iter=100)->RootResult:...

    def brent(self, a: np.ndarray, b: np.ndarray, args: np.ndarray, ftol=1e-8, xtol=1e-8, max_iter=100)->RootResult:...