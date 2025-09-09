/**
 * @author Room 6.030
 * @copyright 2025 lucid
 * @licence BSD 3-Clause License
 * @file
 * Util module.
 */
#ifndef LUCID_PYTHON_BUILD
#error LUCID_PYTHON_BUILD is not defined. Ensure you are building with the option '--config=py'
#endif

#include "lucid/util/util.h"

#include <pybind11/functional.h>

#include "bindings/pylucid/pylucid.h"
#include "lucid/util/logging.h"

namespace py = pybind11;
using namespace lucid;

#define STATS_PROPERTY(name)                                                             \
  [](const ScopedStats& self) {                                                          \
    if (self.stats.empty()) {                                                            \
      throw std::runtime_error("No stats available. Make sure to use the 'with' sytax"); \
    }                                                                                    \
    return self.stats.at(0)->name;                                                       \
  }

struct ScopedStats {
  std::vector<Stats::Scoped> stats;
};

std::ostream& operator<<(std::ostream& os, const ScopedStats& stats) {
  return os << (stats.stats.empty() ? "No stats available." : fmt::format("{}", *stats.stats.at(0)));
}

void init_util(py::module_& m) {
  py::module_ r = m.def_submodule("random");
  r.def("seed", &random::seed, py::arg("s") = -1);

  py::module_ log = m.def_submodule("log");
  log.attr("LOG_NONE") = -1;
  log.attr("LOG_CRITICAL") = 0;
  log.attr("LOG_ERROR") = 1;
  log.attr("LOG_WARN") = 2;
  log.attr("LOG_INFO") = 3;
  log.attr("LOG_DEBUG") = 4;
  log.attr("LOG_TRACE") = 5;

  log.def("set_verbosity", py::overload_cast<int>(log::set_verbosity_level), py::arg("level") = 3);
  log.def("set_sink", py::overload_cast<std::function<void(std::string)>>(log::set_logger_sink), py::arg("cb"));
  log.def("set_pattern", &log::set_pattern, py::arg("pattern"));
  log.def("clear", log::clear_logger);

  log.def("trace", [](const std::string& message) { LUCID_TRACE_FMT("{}", message); }, py::arg("message"));
  log.def("debug", [](const std::string& message) { LUCID_DEBUG_FMT("{}", message); }, py::arg("message"));
  log.def("info", [](const std::string& message) { LUCID_INFO_FMT("{}", message); }, py::arg("message"));
  log.def("warn", [](const std::string& message) { LUCID_WARN_FMT("{}", message); }, py::arg("message"));
  log.def("error", [](const std::string& message) { LUCID_ERROR_FMT("{}", message); }, py::arg("message"));
  log.def("critical", [](const std::string& message) { LUCID_CRITICAL_FMT("{}", message); }, py::arg("message"));

  const py::module_ e = m.def_submodule("exception");
  py::register_exception<exception::LucidException>(e, "LucidException", PyExc_RuntimeError);
  py::register_exception<exception::LucidInvalidArgumentException>(e, "LucidInvalidArgumentException",
                                                                   PyExc_ValueError);
  py::register_exception<exception::LucidAssertionException>(e, "LucidAssertionException", PyExc_AssertionError);
  py::register_exception<exception::LucidParserException>(e, "LucidParserException", PyExc_RuntimeError);
  py::register_exception<exception::LucidNotImplementedException>(e, "LucidNotImplementedException",
                                                                  PyExc_NotImplementedError);
  py::register_exception<exception::LucidNotSupportedException>(e, "LucidNotSupportedException",
                                                                PyExc_NotImplementedError);
  py::register_exception<exception::LucidOutOfRangeException>(e, "LucidOutOfRangeException", PyExc_IndexError);
  py::register_exception<exception::LucidUnreachableException>(e, "LucidUnreachableException", PyExc_RuntimeError);
  py::register_exception<exception::LucidPyException>(e, "LucidPyException", PyExc_RuntimeError);
  py::register_exception<exception::LucidLpSolverException>(e, "LucidLpSolverException", PyExc_RuntimeError);

  py::class_<ScopedStats>(m, "Stats")
      .def(py::init<>())
      .def_property_readonly("estimator_time", STATS_PROPERTY(estimator_timer.seconds()))
      .def_property_readonly("feature_map_time", STATS_PROPERTY(feature_map_timer.seconds()))
      .def_property_readonly("optimiser_time", STATS_PROPERTY(optimiser_timer.seconds()))
      .def_property_readonly("tuning_time", STATS_PROPERTY(tuning_timer.seconds()))
      .def_property_readonly("num_constraints", STATS_PROPERTY(num_constraints))
      .def_property_readonly("num_variables", STATS_PROPERTY(num_variables))
      .def_property_readonly("peak_memory_usage_kb", STATS_PROPERTY(peak_memory_usage_kb))
      .def("__enter__",
           [](ScopedStats& self) -> ScopedStats& {
             self.stats.emplace_back();
             return self;
           })
      .def("__exit__",
           [](ScopedStats& self, const py::object&, const py::object&, const py::object&) { self.stats.pop_back(); })
      .def("__str__", STRING_LAMBDA(ScopedStats));
}
