from numiphy.lowlevelsupport import *
from typing import Any
from ._solvers import _LowLevelSolver, _LowLevelSolver1D, RootResult #type: ignore


class SymbolicEquationSystem(CompileTemplate):
    
    def __init__(self, f: Iterable[Expr], x: Iterable[Symbol], args: Iterable[Symbol] = ()):
        '''
        f: system of equations
        x: list of variables to solve
        args: extra parameters in the system
        '''
        self.f = tuple(f)
        self.x = tuple(x)
        self.args = tuple(args)
        assert (len(f) == len(x))
    
    @property
    def Nsys(self):
        return len(self.x)
    
    @property
    def Nargs(self):
        return len(self.args)
    
    @cached_property
    def jacobian(self):
        return [[self.f[i].diff(self.x[j]) for j in range(self.Nsys)] for i in range(self.Nsys)]
    
    @cached_property
    def lowlevel_callables(self)->tuple[LowLevelCallable, ...]:
        return [TensorLowLevelCallable(self.f, q = self.x, args = self.args), TensorLowLevelCallable(self.jacobian, q = self.x, args = self.args)]
    
    def dataset_dims(self, x0: np.ndarray, args: np.ndarray)->int:
        '''
        Solve the system of equations using the Newton-Raphson method

        Parameters
        -------------------
        x0 (array)  : Initial guess. If the array is 2D, then for each initial guess, a different set of args can be provided.
        args (array): Extra parameters to be passed in the equations. Must be given in the same order as in the initialization.
            An array of args may also be passed, as long as for each set of args, an extra initial guess has been provided in x0
        ftol, xtol: absolute tolerances
        max_iter: Maximum number of Newton-Raphson iterations

        Returns
        -------------------
        x: Array or array of arrays (depending on x0), each containing the solution of the corresponding set of initial guess and args
        success (Array): whether the algorithm converged for each dataset
        iters: Number of iterations for each dataset
        
        '''
        x0 = np.ascontiguousarray(np.array(x0))
        args = np.ascontiguousarray(np.array(args))
        i = 0
        if (self.Nargs==0 and args.size == 0):
            pass
        elif (self.Nargs==0 and args.size > 0) or (self.Nargs>0 and args.size == 0):
            raise ValueError('Invalid args size')
        elif args.size % len(self.args) != 0:
            raise ValueError('Invalid args shape')
        while i<x0.ndim:
            if np.prod(x0.shape[i:], dtype=int) == self.Nsys:
                break
            elif self.Nargs > 0:
                if x0.shape[i] != args.shape[i]:
                    raise ValueError('Incompatible args and x0 shapes')
            i += 1
        if i==x0.ndim:
            raise ValueError('Invalid x0 shape')
        elif args.size > 0:
            if args.ndim < i+1:
                raise ValueError('Invalid args shape')
            elif np.prod(args.shape[i:]) != self.Nargs:
                raise ValueError('Invalid args shape')
        return i


class EquationSystem(SymbolicEquationSystem, _LowLevelSolver):

    def __init__(self, f: Iterable[Expr], x: Iterable[Symbol], args: Iterable[Symbol] = ()):
        SymbolicEquationSystem.__init__(self, f, x, args)
        _LowLevelSolver.__init__(self, *self.pointers, self.Nsys, self.Nargs)

    def newton_raphson(self, x0, args, ftol=1e-8, xtol=1e-8, max_iter=100)->RootResult:
        dataset_dims = self.dataset_dims(x0, args)
        return self._newton_raphson(x0, args, dataset_dims, ftol, xtol, max_iter)


class Equation(SymbolicEquationSystem, _LowLevelSolver1D):

    def __init__(self, f: Expr, x: Expr, args: Iterable[Symbol] = ()):
        SymbolicEquationSystem.__init__(self, [f], [x], args)
        _LowLevelSolver1D.__init__(self, *self.pointers, self.Nargs)

    def newton_raphson(self, x0, args, ftol=1e-8, xtol=1e-8, max_iter=100)->RootResult:
        x0 = np.array(x0)[..., np.newaxis]
        dataset_dims = self.dataset_dims(x0, args)
        res = self._newton_raphson(x0, args, dataset_dims, ftol, xtol, max_iter)
        return RootResult(res.root.squeeze(axis=-1), res.iters, res.success)
