/**
 * @file pylucid.cpp
 * @author Room 6.030
 * @copyright 2025 lucid
 * @licence BSD 3-Clause License
 */
#ifndef LUCID_PYTHON_BUILD
#error LUCID_PYTHON_BUILD is not defined. Ensure you are building with the option '--config=python'
#endif

#include <pybind11/pybind11.h>

#include "lucid/util/calculator.h"
#include "lucid/version.h"

namespace py = pybind11;

using Calculator = lucid::Calculator;

PYBIND11_MODULE(_pylucid, m) {
  auto CalculatorClass = py::class_<Calculator>(m, "Calculator");

  CalculatorClass.def(py::init<>())
      .def(py::init<const int>(), py::arg("verbose"))
      .def("add", &Calculator::add<int>)
      .def("subtract", &Calculator::subtract<int>)
      .def("multiply", &Calculator::multiply<int>)
      .def("divide", &Calculator::divide<int>)
      .def_property_readonly("verbose", &Calculator::getVerbose);

  m.doc() = LUCID_DESCRIPTION;
#ifdef LUCID_VERSION_STRING
  m.attr("__version__") = LUCID_VERSION_STRING;
#else
#error "LUCID_VERSION_STRING is not defined"
#endif
}
