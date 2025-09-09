/**
 * @author c3054737
 * @copyright 2025 lucid
 * @licence BSD 3-Clause License
 * @file
 * Stats class.
 */
#include "lucid/util/Stats.h"

#include "lucid/util/logging.h"

namespace lucid {

std::ostream& operator<<(std::ostream& os, const Stats& stats) {
  return os << fmt::format(
             "Stats:\n"
             "  Estimator time (s):        {:.6f}\n"
             "  Feature map time (s):      {:.6f}\n"
             "  Optimiser time (s):        {:.6f}\n"
             "  Tuning time (s):           {:.6f}\n"
             "  Number of constraints:     {}\n"
             "  Number of variables:       {}\n"
             "  Peak memory usage (kB):    {}\n",
             stats.estimator_timer.seconds(), stats.feature_map_timer.seconds(), stats.optimiser_timer.seconds(),
             stats.tuning_timer.seconds(), stats.number_of_constraints, stats.number_of_variables,
             stats.peak_memory_usage_kb);
}

}  // namespace lucid
