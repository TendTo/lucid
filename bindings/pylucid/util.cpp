/**
 * @author c3054737
 * @copyright 2025 lucid
 * @licence BSD 3-Clause License
 * @file
 * util class.
 */
#ifndef LUCID_PYTHON_BUILD
#error LUCID_PYTHON_BUILD is not defined. Ensure you are building with the option '--config=python'
#endif

#include "lucid/util/util.h"

#include "bindings/pylucid/pylucid.h"

namespace py = pybind11;
using namespace lucid;

void init_util(py::module_ &m) {
  m.attr("LOG_NONE") = -1;
  m.attr("LOG_CRITICAL") = 0;
  m.attr("LOG_ERROR") = 1;
  m.attr("LOG_WARN") = 2;
  m.attr("LOG_INFO") = 3;
  m.attr("LOG_DEBUG") = 4;
  m.attr("LOG_TRACE") = 5;
  m.def("set_verbosity", [](const int value) { LUCID_LOG_INIT_VERBOSITY(value); });
}