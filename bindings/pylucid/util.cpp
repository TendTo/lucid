/**
 * @author Ernesto Casablanca
 * @author Oliver Schön
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
  /**
   * Emplace a new Stats instance onto the stack if none exists.
   * This method ensures that there is always a Stats instance available when entering a new scope.
   * @return reference to the top Stats instance
   */
  ScopedStats& enter() {
    if (!stats_.has_value()) stats_.emplace();
    stats_.value()->total_timer.start();
    return *this;
  }

  /** Clear the Stats instance from the stack. */
  void exit() { stats_.reset(); }

  /**
   * Get a read-only reference to the top Stats instance.
   * @return const reference to the top Stats instance
   * @throw lucid::exception::LucidException if no Stats instance is available
   */
  [[nodiscard]] const Stats& stats() const {
    if (!stats_.has_value()) THROW_NOT_STATS_AVAILABLE_ERROR();
    return *stats_.value();
  }

  /** @checker{has_stats, whether stats are available} */
  [[nodiscard]] bool has_stats() const { return stats_.has_value(); }

  void collect_peak_rss_memory_usage() {
    if (!stats_.has_value()) THROW_NOT_STATS_AVAILABLE_ERROR();
    stats_.value()->peak_rss_memory_usage = metrics::get_peak_rss();
  }

  /** @to_string */
  [[nodiscard]] std::string to_string() const {
    if (!stats_.has_value())
      return "No stats available. Make sure the object is within the 'with' block it was defined in";
    return fmt::format("{}", *stats_.value());
  }

 private:
  std::optional<Stats::Scoped> stats_;  ///< Stack of Stats instances. Can contain at most one element.
};

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
      .def_property_readonly("lattice_resolution", STATS_PROPERTY(lattice_resolution), Stats_lattice_resolution)
      .def_property_readonly("dimension", STATS_PROPERTY(dimension), Stats_dimension)
      .def_property_readonly("lattice_size_active", STATS_PROPERTY(lattice_size_active), Stats_lattice_size_active)
      .def_property_readonly("C", STATS_PROPERTY(C), Stats_C)
      .def_property_readonly("A_xn_wo_x", STATS_PROPERTY(A_xn_wo_x), Stats_A_xn_wo_x)
      .def_property_readonly("A_xn_wo_x0", STATS_PROPERTY(A_xn_wo_x0), Stats_A_xn_wo_x0)
      .def_property_readonly("A_xn_wo_xu", STATS_PROPERTY(A_xn_wo_xu), Stats_A_xn_wo_xu)
      .def_property_readonly("min_x0", STATS_PROPERTY(min_x0), Stats_min_x0)
      .def_property_readonly("max_sx0", STATS_PROPERTY(max_sx0), Stats_max_sx0)
      .def_property_readonly("max_xu", STATS_PROPERTY(max_xu), Stats_max_xu)
      .def_property_readonly("min_sxu", STATS_PROPERTY(min_sxu), Stats_min_sxu)
      .def_property_readonly("max_x", STATS_PROPERTY(max_x), Stats_max_x)
      .def_property_readonly("min_sx", STATS_PROPERTY(min_sx), Stats_min_sx)
      .def_property_readonly("min_d", STATS_PROPERTY(min_d), Stats_min_d)
      .def_property_readonly("max_d_sx", STATS_PROPERTY(max_d_sx), Stats_max_d_sx)
      .def_property_readonly("peak_rss_memory_usage", STATS_PROPERTY(peak_rss_memory_usage),
                             Stats_peak_rss_memory_usage)
      .def_property_readonly("num_estimator_consolidations", STATS_PROPERTY(num_estimator_consolidations),
                             Stats_num_estimator_consolidations)
      .def_property_readonly("num_feature_map_applications", STATS_PROPERTY(num_feature_map_applications),
                             Stats_num_feature_map_applications)
      .def_property_readonly("num_kernel_applications", STATS_PROPERTY(num_kernel_applications),
                             Stats_num_kernel_applications)
      .def_property_readonly("num_tuning", STATS_PROPERTY(num_tuning), Stats_num_tuning)
      .def("to_dict",
           [](const ScopedStats& self) {
             if (!self.has_stats()) THROW_NOT_STATS_AVAILABLE_ERROR();
             py::dict d;
             const Stats& stats = self.stats();
             d["estimator_time"] = stats.estimator_timer.seconds();
             d["feature_map_time"] = stats.feature_map_timer.seconds();
             d["barrier_time"] = stats.barrier_timer.seconds();
             d["optimiser_time"] = stats.optimiser_timer.seconds();
             d["tuning_time"] = stats.tuning_timer.seconds();
             d["kernel_time"] = stats.kernel_timer.seconds();
             d["total_time"] = stats.total_timer.seconds();
             d["num_constraints"] = stats.num_constraints;
             d["num_variables"] = stats.num_variables;
             d["lattice_resolution"] = fmt::format("{}^{}", stats.lattice_resolution, stats.dimension);
             d["lattice_size_active"] = stats.lattice_size_active;
             d["dimension"] = stats.dimension;
             d["C"] = stats.C;
             d["A_xn_wo_x"] = stats.A_xn_wo_x;
             d["A_xn_wo_x0"] = stats.A_xn_wo_x;
             d["A_xn_wo_xu"] = stats.A_xn_wo_xu;
             d["min_x0"] = stats.min_x0;
             d["max_sx0"] = stats.max_sx0;
             d["max_xu"] = stats.max_xu;
             d["min_sxu"] = stats.min_sxu;
             d["max_x"] = stats.max_x;
             d["min_sx"] = stats.min_sx;
             d["min_d"] = stats.min_d;
             d["max_d_sx"] = stats.max_d_sx;
             d["peak_rss_memory_usage"] = stats.peak_rss_memory_usage;
             d["num_estimator_consolidations"] = stats.num_estimator_consolidations;
             d["num_feature_map_applications"] = stats.num_feature_map_applications;
             d["num_kernel_applications"] = stats.num_kernel_applications;
             d["num_tuning"] = stats.num_tuning;
             return d;
           })
      .def("__enter__", &ScopedStats::enter)
      .def("__exit__", [](ScopedStats& self, const py::object&, const py::object&, const py::object&) { self.exit(); })
      .def("__str__", STRING_LAMBDA(ScopedStats));
}
