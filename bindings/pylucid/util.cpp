/**
 * @author c3054737
 * @copyright 2025 lucid
 * @licence BSD 3-Clause License
 * @file
 * util class.
 */
#ifndef LUCID_PYTHON_BUILD
#error LUCID_PYTHON_BUILD is not defined. Ensure you are building with the option '--config=py'
#endif

#include "lucid/util/util.h"

#include "bindings/pylucid/pylucid.h"
#include "lucid/util/logging.h"

namespace py = pybind11;
using namespace lucid;

void init_util(py::module_& m) {
  m.attr("LOG_NONE") = -1;
  m.attr("LOG_CRITICAL") = 0;
  m.attr("LOG_ERROR") = 1;
  m.attr("LOG_WARN") = 2;
  m.attr("LOG_INFO") = 3;
  m.attr("LOG_DEBUG") = 4;
  m.attr("LOG_TRACE") = 5;
  m.def("set_verbosity", [](const int value) { LUCID_LOG_INIT_VERBOSITY(value); });

  m.def("log_trace", [](const std::string& message) { LUCID_TRACE_FMT("{}", message); });
  m.def("log_debug", [](const std::string& message) { LUCID_DEBUG_FMT("{}", message); });
  m.def("log_info", [](const std::string& message) { LUCID_INFO_FMT("{}", message); });
  m.def("log_warn", [](const std::string& message) { LUCID_WARN_FMT("{}", message); });
  m.def("log_error", [](const std::string& message) { LUCID_ERROR_FMT("{}", message); });
  m.def("log_critical", [](const std::string& message) { LUCID_CRITICAL_FMT("{}", message); });

  py::register_exception<exception::LucidException>(m, "LucidException", PyExc_RuntimeError);
  py::register_exception<exception::LucidInvalidArgumentException>(m, "LucidInvalidArgumentException",
                                                                   PyExc_ValueError);
  py::register_exception<exception::LucidAssertionException>(m, "LucidAssertionException", PyExc_AssertionError);
  py::register_exception<exception::LucidParserException>(m, "LucidParserException", PyExc_RuntimeError);
  py::register_exception<exception::LucidNotImplementedException>(m, "LucidNotImplementedException",
                                                                  PyExc_NotImplementedError);
  py::register_exception<exception::LucidNotSupportedException>(m, "LucidNotSupportedException",
                                                                PyExc_NotImplementedError);
  py::register_exception<exception::LucidOutOfRangeException>(m, "LucidOutOfRangeException", PyExc_IndexError);
  py::register_exception<exception::LucidUnreachableException>(m, "LucidUnreachableException", PyExc_RuntimeError);
  py::register_exception<exception::LucidPyException>(m, "LucidPyException", PyExc_RuntimeError);
  py::register_exception<exception::LucidLpSolverException>(m, "LucidLpSolverException", PyExc_RuntimeError);
}
