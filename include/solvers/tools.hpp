#pragma once


#include <cstdlib>
#include <cstdint>

template<typename T>
inline T abs(T x){ return x > 0 ? x : -x;}