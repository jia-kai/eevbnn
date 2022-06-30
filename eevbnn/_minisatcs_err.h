#include <Python.h>

#include <exception>

namespace {

PyObject* g_error_pyobj;

void raise_py_error_setup(PyObject* pyobj) {
    g_error_pyobj = pyobj;
}

void raise_py_error() {
    try {
        throw;
    } catch (const std::exception& e) {
        PyErr_SetString(g_error_pyobj, e.what());
    }
}

}  // anonymous namespace
