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

struct Stats {
  using Scoped = ScopedValue<Stats, struct StatsTag>;

  Timer estimator_timer;
  Timer feature_map_timer;
  Timer optimiser_timer;
  Timer tuning_timer;
  Timer kernel_timer;
  std::size_t num_estimator_consolidations{0};
  std::size_t num_kernel_applications{0};
  std::size_t number_of_constraints{0};
  std::size_t number_of_variables{0};
  std::size_t peak_memory_usage_kb{0};
};

std::ostream& operator<<(std::ostream& os, const Stats& stats);

}  // namespace lucid

#ifdef LUCID_INCLUDE_FMT

#include "lucid/util/logging.h"

OSTREAM_FORMATTER(lucid::Stats)

#endif
