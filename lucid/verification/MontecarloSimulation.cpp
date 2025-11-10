/**
 * @author c3054737
 * @copyright 2025 lucid
 * @licence BSD 3-Clause License
 * @file
 */
#include "lucid/verification/MontecarloSimulation.h"

#include <algorithm>
#include <execution>
#include <ostream>
#include <utility>

#include "lucid/model/MultiSet.h"
#include "lucid/util/error.h"

namespace lucid {

std::pair<double, double> MontecarloSimulation::safety_probability(
    const Set& X_bounds, const Set& X_init, const Set& X_unsafe,
    const std::function<Vector(ConstVectorRef)>& system_dynamics, const std::size_t time_horizon,
    const double confidence_level, const Dimension num_samples) const {
  LUCID_TRACE_FMT("({}, {}, {}, <func>, {}, {}, {})", X_bounds, X_init, X_unsafe, time_horizon, confidence_level,
                  num_samples);
  LUCID_CHECK_ARGUMENT_CMP(confidence_level, >=, 0.0);
  LUCID_CHECK_ARGUMENT_CMP(confidence_level, <, 1.0);
  LUCID_CHECK_ARGUMENT_CMP(num_samples, >, 0);
  LUCID_CHECK_ARGUMENT_CMP(time_horizon, >, 0);
  LUCID_CHECK_ARGUMENT_CMP(X_bounds.dimension(), ==, X_init.dimension());
  LUCID_CHECK_ARGUMENT_CMP(X_bounds.dimension(), ==, X_unsafe.dimension());

#ifndef NCHECK
  if (dynamic_cast<const MultiSet*>(&X_init) != nullptr) {
    LUCID_WARN(
        "Initial set is a MultiSet. "
        "Sampling will be performed across all subsets regardless of their size or intersections. "
        "Consider using a single set for more accurate results.");
  }
#endif

  const Matrix samples{X_init.sample(num_samples)};
  // Count the number of safe trajectories
  std::size_t satisf_count = 0;
  for (auto& row : samples.rowwise()) {
    Vector state = row;
    bool is_safe = true;
    for (std::size_t j = 0; j < time_horizon; ++j) {
      state = system_dynamics(state);
      if (X_unsafe.contains(state)) {
        is_safe = false;  // Unsafe trajectory detected
        break;
      }
      if (!X_bounds.contains(state)) break;  // Out of bounds is safe
    }
    satisf_count += is_safe ? 1 : 0;
  }
  const double sat_prob_mc = static_cast<double>(satisf_count) / static_cast<double>(num_samples);
  const double err = std::sqrt(1 / (4 * static_cast<double>(num_samples) * (1 - confidence_level)));
  return {std::max(sat_prob_mc - err, 0.0), std::min(sat_prob_mc + err, 1.0)};
}

std::ostream& operator<<(std::ostream& os, const MontecarloSimulation&) { return os << "MontecarloSimulation( )"; }

}  // namespace lucid
