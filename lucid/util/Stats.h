/**
 * @author c3054737
 * @copyright 2025 lucid
 * @licence BSD 3-Clause License
 * @file
 * Stats struct.
 */
#pragma once

#include <iosfwd>

#include "lucid/util/ScopedValue.h"
#include "lucid/util/Timer.h"

namespace lucid {

/** Simple struct to hold various statistics about the execution of different components. */
struct Stats {
  using Scoped = ScopedValue<Stats, struct StatsTag>;

  Timer estimator_timer;                        ///< Timer spent for estimator applications
  Timer feature_map_timer;                      ///< Timer spent for feature map applications
  Timer optimiser_timer;                        ///< Timer spent in optimising
  Timer barrier_timer;                          ///< Timer spent in barrier certificate synthesis
  Timer tuning_timer;                           ///< Timer spent for hyperparameter tuning
  Timer kernel_timer;                           ///< Timer spent for kernel evaluations
  Timer total_timer;                            ///< Timer for the whole pipeline
  std::size_t num_estimator_consolidations{0};  ///< Number of times an estimator was consolidated
  std::size_t num_kernel_applications{0};       ///< Number of times a kernel was applied
  std::size_t num_feature_map_applications{0};  ///< Number of times a feature map was applied
  std::size_t num_tuning{0};                    ///< Number of hyperparameter tuning runs
  std::size_t num_constraints{0};               ///< Number of constraints in the last optimisation problem
  std::size_t num_variables{0};                 ///< Number of variables in the last optimisation problem
  std::size_t peak_rss_memory_usage{0};         ///< Peak Resident Set Size (RSS) memory usage in bytes
};

std::ostream& operator<<(std::ostream& os, const Stats& stats);

}  // namespace lucid

#ifdef LUCID_INCLUDE_FMT

#include "lucid/util/logging.h"

OSTREAM_FORMATTER(lucid::Stats)

#endif
