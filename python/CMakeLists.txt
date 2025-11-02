cmake_minimum_required(VERSION 3.20)

set(CMAKE_CXX_COMPILER "g++")

project(solvers LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

# Compiler flags
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -fopenmp -O3 -Wall -Wextra -Wpedantic -Wconversion -Wshadow -Werror -fno-math-errno")

# Find Python + Pybind11
find_package(Python3 COMPONENTS Interpreter Development REQUIRED)
find_package(pybind11 CONFIG REQUIRED)

# Include directories - use parent directory for headers
include_directories(${CMAKE_SOURCE_DIR}/../include)

# Build Python module
pybind11_add_module(_solvers src/py_solver.cpp)

# Install the module into the solvers package
install(TARGETS _solvers LIBRARY DESTINATION solvers)
