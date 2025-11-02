#include "omp.h"
#include <vector>
#include <cstring>
#include <iostream>

struct RootResult{
    size_t iters;
    bool converged;
};

template<typename Scalar>
struct RootResult1D{
    Scalar x;
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

template<typename Scalar>
using func_t = void(*)(Scalar* result, const Scalar* x, const void*);

template<typename T>
inline T abs(T x){ return x > 0 ? x : -x;}

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
RootResult newton_raphson(Scalar* x, Scalar* y, Scalar* dx, Scalar* x_like_tmp, Scalar* jac_like_tmp, Scalar* jac, func_t<Scalar> f, func_t<Scalar> f_jac, Scalar ftol, Scalar xtol, size_t max_iter, size_t n, const void* obj = nullptr){
    //only the "x" array will contain the result and is important.
    //It initially contains the starting values
    //the rest of the arrays passed are auxiliary
    //
    bool converged = false;
    size_t iter;
    Scalar norm_dx, norm_f;
    for (iter=0; iter<max_iter; iter++){
        f(y, x, obj);
        f_jac(jac, x, obj);

        lin_solve<Scalar, -1>(dx, jac, y, jac_like_tmp, x_like_tmp, n);

        #pragma omp simd
        for (size_t i=0; i<n; i++){
            x[i] += dx[i];
        }

        norm_dx = 0;
        norm_f = 0;
        #pragma omp simd reduction(+:norm_dx, norm_f)
        for (size_t i=0; i<n; i++){
            norm_dx += dx[i]*dx[i];
            norm_f += y[i]*y[i];
        }

        if (sqrt(norm_dx) < xtol && sqrt(norm_f) < ftol){
            converged = true;
            iter++;
            break;
        }
    }

    return {.iters=iter, .converged=converged};

}


template<typename Scalar>
RootResult1D<Scalar> newton_raphson1D(func_t<Scalar> f, func_t<Scalar> f_jac, Scalar x0, Scalar ftol, Scalar xtol, size_t max_iter, const void* obj = nullptr){

    bool converged = false;
    size_t iter;
    Scalar y, norm_dx, norm_f, jac, dx;
    Scalar x = x0;
    for (iter=0; iter<max_iter; iter++){
        f(&y, &x, obj);
        f_jac(&jac, &x, obj);
        x += y/jac;

        if (sqrt(abs(dx)) < xtol && sqrt(abs(y)) < ftol){
            converged = true;
            iter++;
            break;
        }
    }

    return {.x=x, .iters=iter, .converged=converged};
}



template<typename T>
class Solver{

public:

    Solver(func_t<T> f, func_t<T> jac, size_t Nsys) : _f(f), _jac(jac), _Nsys(Nsys), _y(Nsys), _dx(Nsys), _x_tmp(Nsys), _jac_tmp(Nsys*Nsys), _jac_arr(Nsys*Nsys) {}

    RootResult newton_raphson(T* result, T ftol, T xtol, size_t max_iter, const void* obj) const {
        return ::newton_raphson(result, _y.data(), _dx.data(), _x_tmp.data(), _jac_tmp.data(), _jac_arr.data(), _f, _jac, ftol, xtol, max_iter, _Nsys, obj);
    }

protected:

    // Auxiliary arrays used for the Newton Raphson method
    func_t<T> _f, _jac;
    size_t _Nsys;
    mutable std::vector<T> _y, _dx, _x_tmp, _jac_tmp, _jac_arr;

};


template<typename T>
class Solver1D : public Solver<T>{

public:

    Solver1D(func_t<T> f, func_t<T> jac) : Solver<T>(f, jac, 1) {}

    RootResult1D<T> newton_raphson(T x0, T ftol, T xtol, size_t max_iter, const void* obj) const {
        return newton_raphson1D(this->_f, this->_jac, x0, ftol, xtol, max_iter, obj);
    }

    RootResult1D<T> bisect(T a, T b, T ftol, T xtol, size_t max_iter, const void* obj) const{
        //obj is passed inside _f
        T fa, fb, fc, c;
        this->_f(&fa, &a, obj);
        this->_f(&fb, &b, obj);

        // Check if root is bracketed
        if (fa * fb > 0) {
            return {.x = (a + b) / 2, .iters = 0, .converged = false};
        }

        bool converged = false;
        size_t iter;
        for (iter = 0; iter < max_iter; iter++) {
            c = (a + b) / 2;
            this->_f(&fc, &c, obj);

            // Check convergence
            if (abs(fc) < ftol || (b - a) / 2 < xtol) {
                converged = true;
                iter++;
                break;
            }

            // Update interval
            if (fa * fc < 0) {
                b = c;
                fb = fc;
            } else {
                a = c;
                fa = fc;
            }
        }
        return {.x = c, .iters = iter, .converged = converged};
    }

    RootResult1D<T> brent(T a, T b, T ftol, T xtol, size_t max_iter, const void* obj) const{
        T fa, fb, fc, fs;
        this->_f(&fa, &a, obj);
        this->_f(&fb, &b, obj);

        // Check if root is bracketed
        if (fa * fb > 0) {
            return {.x = (a + b) / 2, .iters = 0, .converged = false};
        }

        // Ensure |f(a)| >= |f(b)|
        if (abs(fa) < abs(fb)) {
            T tmp = a; a = b; b = tmp;
            tmp = fa; fa = fb; fb = tmp;
        }

        T c = a;
        fc = fa;
        bool mflag = true;
        T s = b;
        T d = 0;

        bool converged = false;
        size_t iter;

        for (iter = 0; iter < max_iter; iter++) {
            // Check convergence
            if (abs(fb) < ftol || abs(b - a) < xtol) {
                converged = true;
                iter++;
                break;
            }

            if (fa != fc && fb != fc) {
                // Inverse quadratic interpolation
                s = a * fb * fc / ((fa - fb) * (fa - fc))
                  + b * fa * fc / ((fb - fa) * (fb - fc))
                  + c * fa * fb / ((fc - fa) * (fc - fb));
            } else {
                // Secant method
                s = b - fb * (b - a) / (fb - fa);
            }

            // Check if bisection is needed
            T tmp2 = (3 * a + b) / 4;
            bool cond1 = !((s > tmp2 && s < b) || (s < tmp2 && s > b));
            bool cond2 = mflag && abs(s - b) >= abs(b - c) / 2;
            bool cond3 = !mflag && abs(s - b) >= abs(c - d) / 2;
            bool cond4 = mflag && abs(b - c) < xtol;
            bool cond5 = !mflag && abs(c - d) < xtol;

            if (cond1 || cond2 || cond3 || cond4 || cond5) {
                s = (a + b) / 2;
                mflag = true;
            } else {
                mflag = false;
            }

            this->_f(&fs, &s, obj);
            d = c;
            c = b;
            fc = fb;

            if (fa * fs < 0) {
                b = s;
                fb = fs;
            } else {
                a = s;
                fa = fs;
            }

            // Ensure |f(a)| >= |f(b)|
            if (abs(fa) < abs(fb)) {
                T tmp = a; a = b; b = tmp;
                tmp = fa; fa = fb; fb = tmp;
            }
        }

        return {.x = b, .iters = iter, .converged = converged};
    }

};