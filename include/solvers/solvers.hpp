#include "omp.h"
#include <vector>
#include <cstring>
#include <iostream>

struct RootResult{
    size_t iters;
    bool converged;
};

template<typename T>
inline void copy_array(T* dest, const T* src, size_t size){
    if (size==0) {return;}
    if constexpr (std::is_trivially_copyable_v<T>){
        std::memcpy(dest, src, size*sizeof(T));
    }
    else{
        std::copy(src, src+size, dest);
    }
}

template<typename T>
using func_t = void(*)(T* result, const T* x, const T* args, const void*);

template<typename T, int rhs_factor = 1>
void lin_solve(T* X, const T* A, const T* B, T* A_tmp, T* B_tmp, size_t n){
    size_t j, k, max_row;
    T tmp, factor;
    copy_array(A_tmp, A, n*n);
    copy_array(B_tmp, B, n);

    for (size_t i=0; i<n; i++){
        
        max_row = i;
        for (k=i+1; k<n; k++){
            if (abs(A_tmp[k*n+i]) > abs(A_tmp[max_row*n+i])){
                max_row = k;
            }
        }
        if (max_row != i){
            for (j=0; j<n; j++){
                tmp = A_tmp[i*n+j];
                A_tmp[i*n+j] = A_tmp[max_row*n+j];
                A_tmp[max_row*n+j] = tmp;
            }
            tmp = B_tmp[i];
            B_tmp[i] = B_tmp[max_row];
            B_tmp[max_row] = tmp;
        }
        for (k=i+1;k<n;k++){
            factor = A_tmp[k*n+i]/A_tmp[i*n+i];
            for (j=i; j<n; j++){
                A_tmp[k*n+j] -= factor * A_tmp[i*n+j];
            }
            B_tmp[k] -= factor * B_tmp[i];
        }
    }

    
    for (size_t t = 0; t < n; t++) {
        size_t i = n - 1 - t;  // counts down from n-1 to 0

        if constexpr (rhs_factor != 1) {
            X[i] = B_tmp[i] * rhs_factor;
        } else {
            X[i] = B_tmp[i];
        }

        for (j = i + 1; j < n; j++) {
            X[i] -= A_tmp[i * n + j] * X[j];
        }

        X[i] /= A_tmp[i * n + i];
    }
}


template<typename Scalar>
RootResult newton_raphson(Scalar* x, Scalar* y, Scalar* dx, Scalar* x_like_tmp, Scalar* jac_like_tmp, Scalar* jac, func_t<Scalar> f, func_t<Scalar> f_jac, Scalar ftol, Scalar xtol, size_t max_iter, size_t n, const Scalar* args, const void* obj = nullptr){
    //only the "x" array will contain the result and is important.
    //It initially contains the starting values
    //the rest of the arrays passed are auxiliary
    //
    bool converged = false;
    size_t iter;
    Scalar norm_dx_sq, norm_f_sq;
    for (iter=0; iter<max_iter; iter++){
        f(y, x, args, obj);
        f_jac(jac, x, args, obj);

        lin_solve<Scalar, -1>(dx, jac, y, jac_like_tmp, x_like_tmp, n);

        #pragma omp simd
        for (size_t i=0; i<n; i++){
            x[i] += dx[i];
        }

        norm_dx_sq = 0;
        norm_f_sq = 0;
        #pragma omp simd reduction(+:norm_dx_sq, norm_f_sq)
        for (size_t i=0; i<n; i++){
            norm_dx_sq += dx[i]*dx[i];
            norm_f_sq += y[i]*y[i];
        }

        if (sqrt(norm_dx_sq) < xtol && sqrt(norm_f_sq) < ftol){
            converged = true;
            iter++;
            break;
        }
    }

    return {.iters=iter, .converged=converged};

}



template<typename T>
class SolverND{

public:

    SolverND(func_t<T> f, func_t<T> jac, size_t Nsys) : _f(f), _jac(jac), _Nsys(Nsys), _y(Nsys), _dx(Nsys), _x_tmp(Nsys), _jac_tmp(Nsys*Nsys), _jac_arr(Nsys*Nsys) {}

    RootResult newton_raphson(T* result, T xtol, T ftol, size_t max_iter, const T* args, const void* obj) const {
        return ::newton_raphson(result, _y.data(), _dx.data(), _x_tmp.data(), _jac_tmp.data(), _jac_arr.data(), _f, _jac, xtol, ftol, max_iter, _Nsys, args, obj);
    }

protected:

    // Auxiliary arrays used for the Newton Raphson method
    func_t<T> _f, _jac;
    size_t _Nsys;
    mutable std::vector<T> _y, _dx, _x_tmp, _jac_tmp, _jac_arr;

};
