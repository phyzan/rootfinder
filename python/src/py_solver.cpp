#include "pysolvers/py_solver.hpp"

PYBIND11_MODULE(_solvers, m){
    define_solver_module<double>(m);
}