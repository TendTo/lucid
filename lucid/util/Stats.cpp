/**
 * @author c3054737
 * @copyright 2025 lucid
 * @licence BSD 3-Clause License
 * @file
 * Stats class.
 */
#include "lucid/util/Stats.h"

#include "lucid/util/logging.h"
#include "lucid/util/metrics.h"

namespace lucid {

std::ostream& operator<<(std::ostream& os, const Stats& stats) {
  metrics::MemoryUnit unit = metrics::get_suggested_memory_unit(stats.peak_rss_memory_usage);
  return os << fmt::format(
             "Stats:\n"
             "  Kernel time (s):                   {:.3f}\n"
             "  Feature map time (s):              {:.3f}\n"
             "  Estimator time (s):                {:.3f}\n"
             "  Tuning time (s):                   {:.3f}\n"
             "  Optimiser time (s):                {:.3f}\n"
             "  Total time (s):                    {:.3f}\n"
             "  No. of estimator consolidations:   {}\n"
             "  No. of kernel applications:        {}\n"
             "  No. of feature map applications:   {}\n"
             "  No. of hyperparameter tuning:      {}\n"
             "  No. of constraints:                {}\n"
             "  No. of variables:                  {}\n"
             "  Peak memory usage ({}):            {:.3f}\n",
             stats.kernel_timer.seconds(),        //
             stats.feature_map_timer.seconds(),   //
             stats.estimator_timer.seconds(),     //
             stats.tuning_timer.seconds(),        //
             stats.optimiser_timer.seconds(),     //
             stats.total_timer.seconds(),         //
             stats.num_estimator_consolidations,  //
             stats.num_kernel_applications,       //
             stats.num_feature_map_applications,  //
             stats.num_tuning,                    //
             stats.num_constraints,               //
             stats.num_variables,                 //
             unit,                                //
             metrics::bytes_to(stats.peak_rss_memory_usage, unit));
}

}  // namespace lucid
