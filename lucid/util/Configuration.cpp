/**
 * @author Room 6.030
 * @copyright 2025 lucid
 * @licence BSD 3-Clause License
 * @file
 */
#include "lucid/util/Configuration.h"

#include <ostream>

#include "lucid/util/error.h"
#include "lucid/util/logging.h"

namespace lucid {

const Matrix emptyMatrix{Matrix::Zero(0, 0)};

std::ostream &operator<<(std::ostream &os, const Configuration &config) {
  return os << fmt::format(
             "Config( "
             "verbose( {} ) "
             "seed( {} ) "
             "plot( {} ) "
             "verify( {} ) "
             "problem_log_file( {} ) "
             "iis_log_file( {} ) "
             "system_dynamics( {} ) "
             "X_bounds( {} ) "
             "X_init( {} ) "
             "X_unsafe( {} ) "
             "x_samples( {} ) "
             "xp_samples( {} ) "
             "f_xp_samples( {} ) "
             "num_samples( {} ) "
             "noise_scale( {} ) "
             "lambda( {} ) "
             "sigma_f( {} ) "
             "sigma_l( {} ) "
             "num_frequencies( {} ) "
             "oversample_factor( {} ) "
             "num_oversample( {} ) "
             "gamma( {} ) "
             "C_coeff( {} ) "
             "time_horizon( {} ) "
             "epsilon( {} ) "
             "b_norm( {} ) "
             "b_kappa( {} ) "
             "estimator( {} ) "
             "kernel( {} ) "
             "feature_map( {} ) "
             "optimiser( {} ) "
             ")",
             config.verbose(), config.seed(), config.plot(), config.verify(), config.problem_log_file(),
             config.iis_log_file(), config.system_dynamics() ? "provided" : "not provided",
             config.X_bounds() == nullptr ? "-" : fmt::format("{}", *config.X_bounds()),
             config.X_init() == nullptr ? "-" : fmt::format("{}", *config.X_init()),
             config.X_unsafe() == nullptr ? "-" : fmt::format("{}", *config.X_unsafe()), config.x_samples(),
             config.xp_samples(), config.f_xp_samples(), config.num_samples(), config.noise_scale(), config.lambda(),
             config.sigma_f(), config.sigma_l(), config.num_frequencies(), config.oversample_factor(),
             config.num_oversample(), config.gamma(), config.C_coeff(), config.time_horizon(), config.epsilon(),
             config.b_norm(), config.b_kappa(), config.estimator(), config.kernel(), config.feature_map(),
             config.optimiser());
}
std::ostream &operator<<(std::ostream &os, const Configuration::Optimiser &optimiser) {
  switch (optimiser) {
    case Configuration::Optimiser::GUROBI:
      return os << "GUROBI";
    case Configuration::Optimiser::ALGLIB:
      return os << "ALGLIB";
    case Configuration::Optimiser::HIGHS:
      return os << "HIGHS";
    default:
      LUCID_UNREACHABLE();
  }
}
std::ostream &operator<<(std::ostream &os, const Configuration::Estimator &estimator) {
  switch (estimator) {
    case Configuration::Estimator::KERNEL_RIDGE_REGRESSOR:
      return os << "KERNEL_RIDGE_REGRESSOR";
    default:
      LUCID_UNREACHABLE();
  }
}
std::ostream &operator<<(std::ostream &os, const Configuration::Kernel &kernel) {
  switch (kernel) {
    case Configuration::Kernel::GAUSSIAN_KERNEL:
      return os << "GAUSSIAN_KERNEL";
    default:
      LUCID_UNREACHABLE();
  }
}
std::ostream &operator<<(std::ostream &os, const Configuration::FeatureMap &feature_map) {
  switch (feature_map) {
    case Configuration::FeatureMap::LINEAR_TRUNCATED_FOURIER_FEATURE_MAP:
      return os << "LINEAR_TRUNCATED_FOURIER_FEATURE_MAP";
    case Configuration::FeatureMap::LOG_TRUNCATED_FOURIER_FEATURE_MAP:
      return os << "LOG_TRUNCATED_FOURIER_FEATURE_MAP";
    case Configuration::FeatureMap::CONSTANT_TRUNCATED_FOURIER_FEATURE_MAP:
      return os << "CONSTANT_TRUNCATED_FOURIER_FEATURE_MAP";
    default:
      LUCID_UNREACHABLE();
  }
}

}  // namespace lucid
