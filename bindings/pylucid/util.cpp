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
#include <pybind11/stl.h>

#include <optional>

#include "bindings/pylucid/doxygen_docstrings.h"
#include "bindings/pylucid/pylucid.h"
#include "lucid/util/error.h"
#include "lucid/util/logging.h"

namespace py = pybind11;
using namespace lucid;

#define THROW_NOT_STATS_AVAILABLE_ERROR() \
  throw exception::LucidPyException(      \
      "No stats available. Make sure to check the property the 'with' block it was defined in")

#define STATS_PROPERTY(name) [](const ScopedStats& self) { return self.stats().name; }

/**
 * Scoped stats class to be used in Python bindings.
 * This class provides a context manager interface to collect and access statistics
 * related to various operations within a defined scope.
 * @todo Replace the vector with an optional when the issue with pybind11's __enter__ and optional is resolved.
 */
class ScopedStats {
 public:
  /** Create a new ScopedStats instance. */
  ScopedStats() { stats_.reserve(1); }

  /**
   * Emplace a new Stats instance onto the stack if none exists.
   * This method ensures that there is always a Stats instance available when entering a new scope.
   * @return reference to the top Stats instance
   */
  ScopedStats& enter() {
    if (stats_.empty()) stats_.emplace_back();
    stats_.front()->total_timer.start();
    return *this;
  }

  /** Clear the Stats instance from the stack. */
  void exit() { stats_.clear(); }

  /**
   * Get a read-only reference to the top Stats instance.
   * @return const reference to the top Stats instance
   * @throw lucid::exception::LucidException if no Stats instance is available
   */
  [[nodiscard]] const Stats& stats() const {
    if (stats_.empty()) THROW_NOT_STATS_AVAILABLE_ERROR();
    return *stats_.front();
  }

  /** @checker{has_stats, whether stats are available} */
  [[nodiscard]] bool has_stats() const { return !stats_.empty(); }

  void collect_peak_rss_memory_usage() {
    if (stats_.empty()) THROW_NOT_STATS_AVAILABLE_ERROR();
    stats_.front()->peak_rss_memory_usage = metrics::get_peak_rss();
  }

 private:
  std::vector<Stats::Scoped> stats_;  ///< Stack of Stats instances. Can contain at most one element.
};

std::ostream& operator<<(std::ostream& os, const ScopedStats& stats) {
  return os << (stats.has_stats()
                    ? fmt::format("{}", stats.stats())
                    : "No stats available. Make sure the object is within the 'with' block it was defined in");
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

  py::class_<ScopedStats>(m, "Stats", Stats_)
      .def(py::init<>())
      .def("collect_peak_rss_memory_usage", &ScopedStats::collect_peak_rss_memory_usage)
      .def_property_readonly("estimator_time", STATS_PROPERTY(estimator_timer.seconds()), Stats_estimator_timer)
      .def_property_readonly("feature_map_time", STATS_PROPERTY(feature_map_timer.seconds()), Stats_feature_map_timer)
      .def_property_readonly("barrier_time", STATS_PROPERTY(barrier_timer.seconds()), Stats_barrier_timer)
      .def_property_readonly("optimiser_time", STATS_PROPERTY(optimiser_timer.seconds()), Stats_optimiser_timer)
      .def_property_readonly("tuning_time", STATS_PROPERTY(tuning_timer.seconds()), Stats_tuning_timer)
      .def_property_readonly("kernel_time", STATS_PROPERTY(kernel_timer.seconds()), Stats_kernel_timer)
      .def_property_readonly("total_time", STATS_PROPERTY(total_timer.seconds()), Stats_total_timer)
      .def_property_readonly("num_constraints", STATS_PROPERTY(num_constraints), Stats_num_constraints)
      .def_property_readonly("num_variables", STATS_PROPERTY(num_variables), Stats_num_variables)
      .def_property_readonly("peak_rss_memory_usage", STATS_PROPERTY(peak_rss_memory_usage),
                             Stats_peak_rss_memory_usage)
      .def_property_readonly("num_estimator_consolidations", STATS_PROPERTY(num_estimator_consolidations),
                             Stats_num_estimator_consolidations)
      .def_property_readonly("num_feature_map_applications", STATS_PROPERTY(num_feature_map_applications),
                             Stats_num_feature_map_applications)
      .def_property_readonly("num_kernel_applications", STATS_PROPERTY(num_kernel_applications),
                             Stats_num_kernel_applications)
      .def_property_readonly("num_tuning", STATS_PROPERTY(num_tuning), Stats_num_tuning)
      .def("__enter__", &ScopedStats::enter)
      .def("__exit__", [](ScopedStats& self, const py::object&, const py::object&, const py::object&) { self.exit(); })
      .def("__str__", STRING_LAMBDA(ScopedStats));
}
