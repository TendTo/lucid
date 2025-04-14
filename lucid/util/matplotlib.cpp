/**
 * @author Room 6.030
 * @author Benno Evers
 * @copyright 2014 Benno Evers
 * @copyright 2025 lucid
 * @licence BSD 3-Clause License
 * @file
 */
#ifndef LUCID_MATPLOTLIB_BUILD
#error "This file should not be included without LUCID_LUCID_MATPLOTLIB_BUILD"
#endif
#include "lucid/util/matplotlib.h"

namespace lucid::plt {

bool xlim(double left, double right) {
  return static_cast<bool>(internal::Interpreter::get().xlim()(py::make_tuple(left, right)));
}
bool ylim(double bottom, double top) {
  return static_cast<bool>(internal::Interpreter::get().ylim()(py::make_tuple(bottom, top)));
}
std::array<double, 2> xlim() {
  const py::tuple lims = internal::Interpreter::get().xlim()();
  if (!lims || lims.is_none() || lims.size() != 2) throw std::runtime_error("Failed to get xlim");
  return {lims[0].cast<double>(), lims[1].cast<double>()};
}
std::array<double, 2> ylim() {
  const py::tuple lims = internal::Interpreter::get().ylim()();
  if (!lims || lims.is_none() || lims.size() != 2) throw std::runtime_error("Failed to get ylim");
  return {lims[0].cast<double>(), lims[1].cast<double>()};
}

}  // namespace lucid::plt
