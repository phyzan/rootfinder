#include <cstddef>
#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <vector>
#include "../solvers/solvers.hpp"

namespace py = pybind11;

template<typename T>
py::array_t<T> array(T* data, const std::vector<py::ssize_t>& shape){
    py::capsule capsule = py::capsule(data, [](void* r){T* d = reinterpret_cast<T*>(r); delete[] d;});
    return py::array_t<T>(shape, data, capsule);
}

template<typename Scalar>
using py_func_t = void(*)(Scalar* result, const Scalar* x, const Scalar* args, const void*);

template<typename T>
struct PySolveData{
    py_func_t<T> py_func;
    py_func_t<T> py_jacobian;
    const T* args;
};

template<typename T>
void py_f(T* result, const T* x, const void* obj){
    const auto* data = reinterpret_cast<const PySolveData<T>*>(obj);
    return data->py_func(result, x, data->args, nullptr);
}

template<typename T>
void py_jac(T* result, const T* x, const void* obj){
    const auto* data = reinterpret_cast<const PySolveData<T>*>(obj);
    return data->py_jacobian(result, x, data->args, nullptr);
}


template<typename T>
struct PySolveResult{

    PySolveResult(py::array_t<T> x_, py::array_t<py::ssize_t> iters_, py::array_t<bool> success_) : x(std::move(x_)), iters(std::move(iters_)), success(std::move(success_)) {}

    py::array_t<T> x;
    py::array_t<py::ssize_t> iters;
    py::array_t<bool> success;
};


template<typename T>
struct PyEqSolver{

    PyEqSolver(const py::capsule& py_func, const py::capsule& py_jacobian, size_t nsys, size_t nargs) : solver(py_f, py_jac, nsys), Nsys(nsys), Nargs(nargs){
        f = reinterpret_cast<py_func_t<T>>(py_func.get_pointer());
        jac = reinterpret_cast<py_func_t<T>>(py_jacobian.get_pointer());
    }

    
    PySolveResult<T> py_solve_newt(py::array_t<T> x0, py::array_t<T> args, size_t dataset_dims, T ftol, T xtol, size_t max_iter){
        //if py_f and py_jac have accept some shape for the input "x", then, for every extra axis that args has, x0 must have one more axis and vice versa.
        //the data must arranged in a C-order, such that args_pointer + n_args is the next set of args. Similarly, x0_pointer+nsys
        const size_t n_datasets = static_cast<size_t>(x0.size()) / Nsys;
        auto ndim = static_cast<size_t>(x0.ndim());
        T* x = new T[x0.size()];
        auto* py_iters = new size_t[n_datasets];
        bool* py_success = new bool[n_datasets];
        const T* py_data = static_cast<const T*>(x0.request().ptr);
        copy_array(x, py_data, static_cast<size_t>(x0.size()));

        const T* py_args = static_cast<const T*>(args.request().ptr);
        PySolveData<T> data = {f, jac, py_args};
        
        for (size_t i=0; i<n_datasets; i++){
            data.args = py_args + i*Nargs;
            auto res = solver.newton_raphson(x+i*Nsys, ftol, xtol, max_iter, &data);
            py_iters[i] = res.iters;
            py_success[i] = res.converged;
        }
        

        std::vector<py::ssize_t> shape(ndim);
        const py::ssize_t* shape_ptr = x0.shape();
        for (size_t i=0; i<ndim; i++){
            shape[i] = shape_ptr[i];
        }

        std::vector<py::ssize_t> dataset_shape(dataset_dims);
        for (py::ssize_t i=0; i<static_cast<py::ssize_t>(dataset_dims); i++){
            dataset_shape[i] = x0.shape(i);
        }

        py::array_t<T> x_res = array(x, shape);

        py::array_t<py::size_t> iters_array = array(py_iters, dataset_shape);

        py::array_t<bool> succ_array = array(py_success, dataset_shape);
        return PySolveResult<T>(x_res, iters_array, succ_array);
    }

    Solver<T> solver;
    size_t Nsys;
    size_t Nargs;
    py_func_t<T> f;
    py_func_t<T> jac;
};


template<typename T>
struct PyEqSolver1D : PyEqSolver<T>{

    PyEqSolver1D(const py::capsule& py_func, const py::capsule& py_jacobian, size_t nargs) : PyEqSolver<T>(py_func, py_jacobian, 1, nargs), solver1D(py_f, py_jac) {}

    PySolveResult<T> py_solve_bisect(py::array_t<T> a, py::array_t<T> b, py::array_t<T> args, T ftol, T xtol, size_t max_iter){
        // Validate shapes: a and b must have the same shape
        auto a_ndim = static_cast<size_t>(a.ndim());
        auto b_ndim = static_cast<size_t>(b.ndim());
        auto args_ndim = static_cast<size_t>(args.ndim());

        if (a_ndim != b_ndim) {
            throw std::runtime_error("a and b must have the same number of dimensions");
        }

        for (size_t i = 0; i < a_ndim; i++) {
            if (a.shape(i) != b.shape(i)) {
                throw std::runtime_error("a and b must have the same shape");
            }
        }

        // Validate args shape based on Nargs
        if (this->Nargs > 0) {
            // args must have one more dimension than a and b
            if (args_ndim != a_ndim + 1) {
                throw std::runtime_error("args must have one more dimension than a and b");
            }

            // Check that leading dimensions of args match a and b
            for (size_t i = 0; i < a_ndim; i++) {
                if (args.shape(i) != a.shape(i)) {
                    throw std::runtime_error("Leading dimensions of args must match a and b");
                }
            }

            // Check that last dimension of args matches Nargs
            if (static_cast<size_t>(args.shape(args_ndim - 1)) != this->Nargs) {
                throw std::runtime_error("Last dimension of args must match nargs");
            }
        } else {
            // When Nargs == 0, args must be empty
            if (args.size() != 0) {
                throw std::runtime_error("When nargs is 0, args must be an empty array");
            }
        }

        const size_t n_datasets = static_cast<size_t>(a.size());
        T* x = new T[n_datasets];
        auto* py_iters = new size_t[n_datasets];
        bool* py_success = new bool[n_datasets];

        const T* py_a = static_cast<const T*>(a.request().ptr);
        const T* py_b = static_cast<const T*>(b.request().ptr);
        const T* py_args = static_cast<const T*>(args.request().ptr);
        PySolveData<T> data = {this->f, this->jac, py_args};

        for (size_t i=0; i<n_datasets; i++){
            data.args = py_args + i*this->Nargs;
            auto res = solver1D.bisect(py_a[i], py_b[i], ftol, xtol, max_iter, &data);
            x[i] = res.x;
            py_iters[i] = res.iters;
            py_success[i] = res.converged;
        }

        std::vector<py::ssize_t> dataset_shape(a.ndim());
        for (py::ssize_t i=0; i<static_cast<py::ssize_t>(a.ndim()); i++){
            dataset_shape[i] = a.shape(i);
        }

        py::array_t<T> x_res = array(x, dataset_shape);
        py::array_t<py::size_t> iters_array = array(py_iters, dataset_shape);
        py::array_t<bool> succ_array = array(py_success, dataset_shape);
        return PySolveResult<T>(x_res, iters_array, succ_array);
    }

    PySolveResult<T> py_solve_brent(py::array_t<T> a, py::array_t<T> b, py::array_t<T> args, T ftol, T xtol, size_t max_iter){
        // Validate shapes: a and b must have the same shape
        auto a_ndim = static_cast<size_t>(a.ndim());
        auto b_ndim = static_cast<size_t>(b.ndim());
        auto args_ndim = static_cast<size_t>(args.ndim());

        if (a_ndim != b_ndim) {
            throw std::runtime_error("a and b must have the same number of dimensions");
        }

        for (size_t i = 0; i < a_ndim; i++) {
            if (a.shape(i) != b.shape(i)) {
                throw std::runtime_error("a and b must have the same shape");
            }
        }

        // Validate args shape based on Nargs
        if (this->Nargs > 0) {
            // args must have one more dimension than a and b
            if (args_ndim != a_ndim + 1) {
                throw std::runtime_error("args must have one more dimension than a and b");
            }

            // Check that leading dimensions of args match a and b
            for (size_t i = 0; i < a_ndim; i++) {
                if (args.shape(i) != a.shape(i)) {
                    throw std::runtime_error("Leading dimensions of args must match a and b");
                }
            }

            // Check that last dimension of args matches Nargs
            if (static_cast<size_t>(args.shape(args_ndim - 1)) != this->Nargs) {
                throw std::runtime_error("Last dimension of args must match nargs");
            }
        } else {
            // When Nargs == 0, args must be empty
            if (args.size() != 0) {
                throw std::runtime_error("When nargs is 0, args must be an empty array");
            }
        }

        const size_t n_datasets = static_cast<size_t>(a.size());
        T* x = new T[n_datasets];
        auto* py_iters = new size_t[n_datasets];
        bool* py_success = new bool[n_datasets];

        const T* py_a = static_cast<const T*>(a.request().ptr);
        const T* py_b = static_cast<const T*>(b.request().ptr);
        const T* py_args = static_cast<const T*>(args.request().ptr);
        PySolveData<T> data = {this->f, this->jac, py_args};

        for (size_t i=0; i<n_datasets; i++){
            data.args = py_args + i*this->Nargs;
            auto res = solver1D.brent(py_a[i], py_b[i], ftol, xtol, max_iter, &data);
            x[i] = res.x;
            py_iters[i] = res.iters;
            py_success[i] = res.converged;
        }

        std::vector<py::ssize_t> dataset_shape(a.ndim());
        for (py::ssize_t i=0; i<static_cast<py::ssize_t>(a.ndim()); i++){
            dataset_shape[i] = a.shape(i);
        }

        py::array_t<T> x_res = array(x, dataset_shape);
        py::array_t<py::size_t> iters_array = array(py_iters, dataset_shape);
        py::array_t<bool> succ_array = array(py_success, dataset_shape);
        return PySolveResult<T>(x_res, iters_array, succ_array);
    }

    Solver1D<T> solver1D;
};


template<typename T>
void define_solver_module(py::module& m){

    py::class_<PyEqSolver<T>, std::unique_ptr<PyEqSolver<T>>>(m, "_LowLevelSolver")
        .def(py::init<py::capsule, py::capsule, size_t, size_t>(),
        py::arg("f"),
        py::arg("jac"),
        py::arg("nsys"),
        py::arg("nargs"))

        .def("_newton_raphson", &PyEqSolver<T>::py_solve_newt,
            py::arg("x0"), py::arg("args"), py::arg("dataset_dims"), py::arg("ftol"), py::arg("xtol"), py::arg("max_iter"));

    py::class_<PyEqSolver1D<T>, PyEqSolver<T>>(m, "_LowLevelSolver1D")
        .def(py::init<py::capsule, py::capsule, size_t>(),
        py::arg("f"),
        py::arg("jac"),
        py::arg("nargs"))

        .def("bisect", &PyEqSolver1D<T>::py_solve_bisect,
            py::arg("a"), py::arg("b"), py::arg("args"), py::arg("ftol")=1e-8, py::arg("xtol")=1e-8, py::arg("max_iter")=100)

        .def("brent", &PyEqSolver1D<T>::py_solve_brent,
            py::arg("a"), py::arg("b"), py::arg("args"), py::arg("ftol")=1e-8, py::arg("xtol")=1e-8, py::arg("max_iter")=100);

    py::class_<PySolveResult<T>, std::unique_ptr<PySolveResult<T>>>(m, "RootResult")
    .def(py::init<py::array_t<T>, py::array_t<py::ssize_t>, py::array_t<bool>>())
    .def_property_readonly("root", [](const PySolveResult<T>& self){return self.x;})
    .def_property_readonly("iters", [](const PySolveResult<T>& self){return self.iters;})
    .def_property_readonly("success", [](const PySolveResult<T>& self){return self.success;});
}
