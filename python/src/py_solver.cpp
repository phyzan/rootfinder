#include "../include/pysolvers/py_solver.hpp"

PYBIND11_MODULE(_solvers, m){
    define_solver_module<double>(m);
}

//g++ -O3 -fno-math-errno -Wall -march=x86-64 -shared -std=c++20 -fopenmp -I/usr/include/python3.12 -I/usr/include/pybind11 -fPIC $(python3 -m pybind11 --includes) solvers/py_solver.cpp -o solvers$(python3-config --extension-suffix)