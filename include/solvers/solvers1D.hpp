#pragma once


#include "tools.hpp"
#include <vector>
#include <iostream>

template<typename T>
using ScalarFunc = T(*)(const T&, const T* args, const void*);


template<typename T>
struct RootResult1D{
    T x;
    size_t iters;
    bool converged;
};

enum class ConvParamTypes : std::uint8_t {x1_and_x2, x1_and_dx, x2_and_dx};

enum class ConvReq : std::uint8_t {All, Any};

template<typename T, ConvReq requirement>
inline bool _has_converged(T x1, T x2, T dx, T y, const ScalarFunc<T>* g, int n_fun, T xtol, T ftol, T rtol, T atol, const T* args, const void* obj){
    if constexpr (requirement == ConvReq::All){
        if (abs(dx) > xtol || abs(y) > ftol){
            return false;
        }
        for (int i=0; i<n_fun; i++){
            T g1 = g[i](x1, args, obj);
            T g2 = g[i](x2, args, obj);
            T dg = g1 - g2;
            if ((abs(dg) > atol) || (abs(dg) > rtol*abs(g1))){
                return false;
            }
        }
        return true;
    }
    else{
        if (abs(dx) <= xtol || abs(y) <= ftol){
            return true;
        }
        for (int i=0; i<n_fun; i++){
            T g1 = g[i](x1, args, obj);
            T g2 = g[i](x2, args, obj);
            T dg = g1 - g2;
            if ((abs(dg) <= atol) || (abs(dg) <= rtol*abs(g1))){
                return true;
            }
        }
        return false;
    }
}

template<typename T, ConvReq requirement, ConvParamTypes types>
inline bool has_converged(T x_like_1, T x_like_2, T y, const ScalarFunc<T>* obj_fun, int n_fun, T xtol, T ftol, T rtol, T atol, const T* args, const void* obj){
    if constexpr (types == ConvParamTypes::x1_and_x2){
        return _has_converged<T, requirement>(x_like_1, x_like_2, x_like_2-x_like_1, y, obj_fun, n_fun, xtol, ftol, rtol, atol, args, obj);
    }
    else if constexpr (types == ConvParamTypes::x1_and_dx){
        return _has_converged<T, requirement>(x_like_1, x_like_1+x_like_2, x_like_2, y, obj_fun, n_fun, xtol, ftol, rtol, atol, args, obj);
    }
    else{
        return _has_converged<T, requirement>(x_like_1-x_like_2, x_like_1, x_like_2, y, obj_fun, n_fun, xtol, ftol, rtol, atol, args, obj);
    }
}

template<typename T, ConvReq REQ>
RootResult1D<T> newton_raphson1D(ScalarFunc<T> f, ScalarFunc<T> f_jac, const ScalarFunc<T>* obj_fun, int n_fun, T x0, T xtol, T ftol, T rtol, T atol, size_t max_iter, const T* args, const void* obj = nullptr){
    bool converged = false;
    size_t iter;
    T y, jac, dx;
    T x = x0;
    for (iter=0; iter<max_iter; iter++){
        y = f(x, args, obj);
        jac = f_jac(x, args, obj);
        dx = y/jac;
        x += y/jac;

        if (has_converged<T, REQ, ConvParamTypes::x2_and_dx>(x, dx, y, obj_fun, n_fun, xtol, ftol, rtol, atol, args, obj)){
            converged = true;
            iter++;
            break;
        }
    }

    return {.x=x, .iters=iter, .converged=converged};
}


template<typename T, ConvReq REQ>
RootResult1D<T> brent(ScalarFunc<T> f, T a, T b, const ScalarFunc<T>* obj_fun, int n_fun, T xtol, T ftol, T rtol, T atol, size_t max_iter, const T* args, const void* obj = nullptr){
    T fa, fb, fc, fs;
    fa = f(a, args, obj);
    fb = f(b, args, obj);

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
        if (has_converged<T, REQ, ConvParamTypes::x1_and_x2>(a, b, fb, obj_fun, n_fun, xtol, ftol, rtol, atol, args, obj)){
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

        fs = f(s, args, obj);
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

template<typename T, ConvReq REQ>
RootResult1D<T> bisect(ScalarFunc<T> f, T a, T b, const ScalarFunc<T>* obj_fun, int n_fun, T xtol, T ftol, T rtol, T atol, size_t max_iter, const T* args, const void* obj = nullptr){
    //obj is passed inside _f
    T fa, fb, fc;
    fa = f(a, args, obj);
    fb = f(b, args, obj);

    // Check if root is bracketed
    if (fa * fb > 0) {
        return {.x = (a + b) / 2, .iters = 0, .converged = false};
    }

    bool converged = false;
    T c = (a + b)/2;
    size_t iter;
    for (iter = 0; iter < max_iter; iter++) {
        c = (a + b) / 2;
        fc = f(c, args, obj);

        // Check convergence
        if (has_converged<T, REQ, ConvParamTypes::x1_and_x2>(c, b, fc, obj_fun, n_fun, xtol, ftol, rtol, atol, args, obj)) {
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


template<typename T>
class Solver1D{

public:

    Solver1D() = default;

    Solver1D(ScalarFunc<T> f, ScalarFunc<T> jac, const ScalarFunc<T>* obj_fun, int n_obj_fun) : _f(f), _jac(jac), _obj_funcs(obj_fun, obj_fun+n_obj_fun){}

    Solver1D(ScalarFunc<T> f, ScalarFunc<T> jac, const std::vector<ScalarFunc<T>>& obj_funcs) : _f(f), _jac(jac), _obj_funcs(obj_funcs){}

    template<ConvReq CONV>
    RootResult1D<T> newton_raphson(T x0, T xtol, T ftol, T rtol, T atol, size_t max_iter, const T* args, const void* obj = nullptr) const {
        return ::newton_raphson1D<T, CONV>(this->_f, this->_jac, this->_obj_funcs.data(), static_cast<int>(this->_obj_funcs.size()), x0, xtol, ftol, rtol, atol, max_iter, args, obj);
    }

    template<ConvReq CONV>
    RootResult1D<T> bisect(T a, T b, T xtol, T ftol, T rtol, T atol, size_t max_iter, const T* args, const void* obj = nullptr) const{
        return ::bisect<T, CONV>(this->_f, a, b, this->_obj_funcs.data(), static_cast<int>(this->_obj_funcs.size()), xtol, ftol, rtol, atol, max_iter, args, obj);
    }

    template<ConvReq CONV>
    RootResult1D<T> brent(T a, T b, T xtol, T ftol, T rtol, T atol, size_t max_iter, const T* args, const void* obj = nullptr) const{
        return ::brent<T, CONV>(this->_f, a, b, this->_obj_funcs.data(), static_cast<int>(this->_obj_funcs.size()), xtol, ftol, rtol, atol, max_iter, args, obj);
    }

private:

    ScalarFunc<T> _f, _jac;
    std::vector<ScalarFunc<T>> _obj_funcs;

};