/**
 * @file interrupt.cpp
 * @author Room 6.030
 * @copyright 2025 lucid
 * @licence BSD 3-Clause License
 */
#ifndef LUCID_PYTHON_BUILD
#error LUCID_PYTHON_BUILD is not defined. Ensure you are building with the option '--config=py'
#endif

#include <Python.h>
#include <pybind11/pybind11.h>

namespace py = pybind11;

namespace lucid {

void py_check_signals() {
  if (PyErr_CheckSignals() == -1) {
    throw py::error_already_set();
  }
}

void py_interrupt_flag(volatile bool *interrupt) {
  if (PyErr_CheckSignals() == -1) *interrupt = true;
}

}  // namespace lucid
