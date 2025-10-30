/**
 * @author Room 6.030
 * @copyright 2025 lucid
 * @licence BSD 3-Clause License
 * @file
 * Configuration class.
 */
#pragma once

#include <iosfwd>
#include <memory>
#include <string>

#include "lucid/lib/eigen.h"
#include "lucid/model/Set.h"

namespace lucid {

extern const Matrix emptyMatrix;

#define LUCID_CONSTEXPR_PARAMETER(param_name, type, default_value, help)                   \
 public:                                                                                   \
  /** @getter{`##param_name##` parameter, configuration, Default to default_value##} */    \
  type &m_##param_name() { return param_name##_; }                                         \
  /** @getsetter{`##param_name##` parameter, configuration, Default to default_value##} */ \
  [[nodiscard]] const type &param_name() const { return param_name##_; }                   \
  static constexpr type default_##param_name{default_value};                               \
  static constexpr const char *const help_##param_name{help};                              \
                                                                                           \
 private:                                                                                  \
  type param_name##_{default_value}  // NOLINT(whitespace/braces): false positive

#define LUCID_CONST_PARAMETER(param_name, type, default_value, help)                       \
 public:                                                                                   \
  /** @getter{`##param_name##` parameter, configuration, Default to default_value##} */    \
  type &m_##param_name() { return param_name##_; }                                         \
  /** @getsetter{`##param_name##` parameter, configuration, Default to default_value##} */ \
  [[nodiscard]] const type &param_name() const { return param_name##_; }                   \
  static constexpr const char *const help_##param_name{help};                              \
                                                                                           \
 private:                                                                                  \
  type param_name##_{default_value}  // NOLINT(whitespace/braces): false positive

/**
 * Simple dataclass used to store the configuration of the program.
 */
class Configuration {
 public:
  /** Underlying optimiser to use. */
  enum class Optimiser {
    GUROBI,  ///< Gurobi optimiser. Default option
    ALGLIB,  ///< Alglib optimiser
    HIGHS,   ///< HiGHS optimiser
  };

  /** Estimator class to use for regression. */
  enum class Estimator {
    KERNEL_RIDGE_REGRESSOR,  ///< Kernel Ridge Regressor. Default option
  };

  /** Kernel class to use for the estimator. */
  enum class Kernel {
    GAUSSIAN_KERNEL,  ///< Gaussian Kernel. Default option
  };

  /** Feature map class to use for transformation. */
  enum class FeatureMap {
    LINEAR_TRUNCATED_FOURIER_FEATURE_MAP,    ///< Linear Truncated Fourier Feature Map. Default option
    CONSTANT_TRUNCATED_FOURIER_FEATURE_MAP,  ///< Constant Truncated Fourier Feature Map
    LOG_TRUNCATED_FOURIER_FEATURE_MAP,       ///< Log Truncated Fourier Feature Map
  };

  /** @constructor{Configuration} */
  Configuration() = default;

 private:
  // Global generic configuration
  LUCID_CONSTEXPR_PARAMETER(verbose, int, 3, "Verbosity level for logging. It goes from -1 (no logging) to 5 (TRACE)");
  LUCID_CONSTEXPR_PARAMETER(seed, int, -1, "Random seed for reproducibility. If < 0, no seeding is done");
  LUCID_CONSTEXPR_PARAMETER(plot, bool, false, "Whether to plot the solution using plotly");
  LUCID_CONSTEXPR_PARAMETER(verify, bool, false, "Whether to verify the barrier certificate using dReal");
  LUCID_CONST_PARAMETER(problem_log_file, std::string, "",
                        "File to save the optimization problem in LP format. "
                        "If empty, the problem will not be saved");
  LUCID_CONST_PARAMETER(iis_log_file, std::string, "",
                        "File to save the irreducible infeasible set (IIS) in ILP format. "
                        "If empty, the IIS will not be saved");

  // System dynamics and specification
  LUCID_CONST_PARAMETER(system_dynamics, std::function<Matrix(Matrix)>, nullptr,
                        "Function that takes a state vector and returns the next state vector. "
                        "If None, the system dynamics are not used and the samples must be provided");
  LUCID_CONST_PARAMETER(X_bounds, std::unique_ptr<Set>, nullptr,
                        "Set of bounds for the state space. If None, no bounds are used. "
                        "If provided, the system dynamics must be provided as well");
  LUCID_CONST_PARAMETER(X_init, std::unique_ptr<Set>, nullptr,
                        "Set of initial states. If None, the initial state is not constrained");
  LUCID_CONST_PARAMETER(X_unsafe, std::unique_ptr<Set>, nullptr,
                        "Set of unsafe states. If None, the unsafe set is not constrained");

  // Transition samples
  LUCID_CONST_PARAMETER(x_samples, Matrix, emptyMatrix,
                        "Matrix of row-vector input samples from the state space. "
                        "If not provided, it will be generated from the X_bounds");
  LUCID_CONST_PARAMETER(xp_samples, Matrix, emptyMatrix,
                        "Matrix of row-vector samples for the next state x+. "
                        "If not provided, it will be generated by applying the system dynamics to the x_samples");
  LUCID_CONST_PARAMETER(f_xp_samples, Matrix, emptyMatrix,
                        "Precomputed feature map application to the next state variable x+. "
                        "If not provided, it will be generated by applying the appropriate feature map");
  LUCID_CONSTEXPR_PARAMETER(num_samples, int, 1000,
                            "Number of samples to use for training the estimator. "
                            "Only used if the samples are not provided");
  LUCID_CONSTEXPR_PARAMETER(noise_scale, double, 0.01,
                            "Scale of the gaussian noise added to the generated xp_samples. "
                            "If 0, no noise is added. Only used if xp_samples are not provided");

  // Kernel regression parameters
  LUCID_CONSTEXPR_PARAMETER(lambda, double, 1e-6, "Regularization constant for the estimator");
  LUCID_CONSTEXPR_PARAMETER(sigma_f, double, 1.0, "Amplitude parameter for the kernel");
  LUCID_CONST_PARAMETER(
      sigma_l, Vector, Vector::Constant(1, 1.0),
      "Variance parameter for the kernel, can be a single float (isotropic) or an array of floats (anisotropic)");

  // Barrier certificate parameters
  LUCID_CONSTEXPR_PARAMETER(num_frequencies, int, 10,
                            "Number of frequencies per dimension for the feature map. "
                            "Includes the constant frequency (0)");
  LUCID_CONSTEXPR_PARAMETER(oversample_factor, double, 2.0,
                            "Factor by which to oversample the frequency space with respect to the nyquist frequency "
                            "(i.e., if set to 1 is the nyquist frequency). "
                            "It is ignored if num_oversample is a positive number");
  LUCID_CONSTEXPR_PARAMETER(num_oversample, int, -1,
                            "Number of lattice points for each dimension. "
                            "Must be greater than the nyquist frequency. "
                            "If negative, it is computed based on the oversample_factor");
  LUCID_CONSTEXPR_PARAMETER(gamma, double, 1.0,
                            "Constant such that the barrier value over the unsafe set is at least gamma");
  LUCID_CONSTEXPR_PARAMETER(c_coefficient, double, 1.0,
                            "Coefficient that makes the optimization more (> 1) or less (< 1) conservative");
  LUCID_CONSTEXPR_PARAMETER(time_horizon, int, 5,
                            "The number of time steps for which the barrier certificate is computed");
  LUCID_CONSTEXPR_PARAMETER(epsilon, double, 0.0, "Robustifying radius");
  LUCID_CONSTEXPR_PARAMETER(b_norm, double, 1.0, "Expected value of the barrier norm");
  LUCID_CONSTEXPR_PARAMETER(b_kappa, double, 1.0, "Coefficient");

  // Classes to use for the pipeline
  LUCID_CONSTEXPR_PARAMETER(estimator, Estimator, Estimator::KERNEL_RIDGE_REGRESSOR,
                            "Estimator class to use for regression");
  LUCID_CONSTEXPR_PARAMETER(kernel, Kernel, Kernel::GAUSSIAN_KERNEL, "Kernel class to use for the estimator");
  LUCID_CONSTEXPR_PARAMETER(feature_map, FeatureMap, FeatureMap::LINEAR_TRUNCATED_FOURIER_FEATURE_MAP,
                            "Feature map class to use for transformation or a callable that returns a feature map");
  LUCID_CONSTEXPR_PARAMETER(optimiser, Optimiser, Optimiser::GUROBI, "Optimiser class to use for the optimization");

  // Deprecated
  LUCID_CONSTEXPR_PARAMETER(constant_lattice_points, bool, false,
                            "Flag to indicate whether to use a constant number of lattice points. Deprecated");
};

std::ostream &operator<<(std::ostream &os, const Configuration &config);
std::ostream &operator<<(std::ostream &os, const Configuration::Optimiser &optimiser);
std::ostream &operator<<(std::ostream &os, const Configuration::Estimator &estimator);
std::ostream &operator<<(std::ostream &os, const Configuration::Kernel &kernel);
std::ostream &operator<<(std::ostream &os, const Configuration::FeatureMap &feature_map);

}  // namespace lucid

#ifdef LUCID_INCLUDE_FMT

#include "lucid/util/logging.h"

OSTREAM_FORMATTER(lucid::Configuration::Optimiser)
OSTREAM_FORMATTER(lucid::Configuration::Estimator)
OSTREAM_FORMATTER(lucid::Configuration::Kernel)
OSTREAM_FORMATTER(lucid::Configuration::FeatureMap)
OSTREAM_FORMATTER(lucid::Configuration)

#endif
