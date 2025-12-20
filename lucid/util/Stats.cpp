/**
 * @author Ernesto Casablanca
 * @author Oliver Schön
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
             "  Barrier time (s):                  {:.3f}\n"
             "  Optimiser time (s):                {:.3f}\n"
             "  Total time (s):                    {:.3f}\n"
             "  No. of estimator consolidations:   {}\n"
             "  No. of kernel applications:        {}\n"
             "  No. of feature map applications:   {}\n"
             "  No. of hyperparameter tuning:      {}\n"
             "  No. of constraints:                {}\n"
             "  No. of variables:                  {}\n"
             "  Lattice size (periodic):           {}\n"
             "  Lattice size (active):             {}\n"
             "  eta:                               {:.3f}\n"
             "  gamma:                             {:.3f}\n"
             "  c:                                 {:.3f}\n"
             "  Safety percentage:                 {:.3f}%\n"
             "  Barrier norm:                      {:.3f}\n"
             "  C:                                 {:.3f}\n"
             "  A xn/x:                            {:.3f}\n"
             "  A xn/x0:                           {:.3f}\n"
             "  A xn/xu:                           {:.3f}\n"
             "  Min x0:                            {:.3f}\n"
             "  Max xn/x0:                         {:.3f}\n"
             "  Max xu:                            {:.3f}\n"
             "  Min xn/xu:                         {:.3f}\n"
             "  Max x:                             {:.3f}\n"
             "  Min xn/x:                          {:.3f}\n"
             "  Min d:                             {:.3f}\n"
             "  Max d xn/x:                        {:.3f}\n"
             "  Peak memory usage ({}):            {:.3f}\n",
             stats.kernel_timer.seconds(),        //
             stats.feature_map_timer.seconds(),   //
             stats.estimator_timer.seconds(),     //
             stats.tuning_timer.seconds(),        //
             stats.barrier_timer.seconds(),       //
             stats.optimiser_timer.seconds(),     //
             stats.total_timer.seconds(),         //
             stats.num_estimator_consolidations,  //
             stats.num_kernel_applications,       //
             stats.num_feature_map_applications,  //
             stats.num_tuning,                    //
             stats.num_constraints,               //
             stats.num_variables,                 //
             fmt::format("{}^{}", stats.lattice_resolution, stats.dimension),  //
             stats.lattice_size_active,     //
             stats.eta,                           //
             stats.gamma,                         //
             stats.c,                             //
             stats.safety * 100.0,                //
             stats.b_norm,                        //
             stats.C,                             //
             stats.A_xn_wo_x,                     //
             stats.A_xn_wo_x0,                    //
             stats.A_xn_wo_xu,                    //
             stats.min_x0,                        //
             stats.max_sx0,                       //
             stats.max_xu,                        //
             stats.min_sxu,                       //
             stats.max_x,                         //
             stats.min_sx,                        //
             stats.min_d,                         //
             stats.max_d_sx,                      //
             unit,                                //
             metrics::bytes_to(stats.peak_rss_memory_usage, unit));
}

}  // namespace lucid
