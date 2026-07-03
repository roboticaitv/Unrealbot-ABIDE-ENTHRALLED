#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <pybind11/stl.h>

namespace py = pybind11;

// Dummy function to test compilation
int add(int i, int j) {
    return i + j;
}

PYBIND11_MODULE(vision_cpp, m) {
    m.doc() = "C++ Vision Engine for Unrealbot";
    m.def("add", &add, "A function that adds two numbers");
}
