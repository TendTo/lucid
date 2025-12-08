/**
 * @file main.cpp
 * @author Room 6.030
 * @copyright 2025 lucid
 * @licence Apache-2.0 license
 * @file
 */
#include <chrono>
#include <iostream>
#include <memory>
#include <string>
#include <utility>
#include <vector>

#include "lucid/lucid.h"
#include "lucid/util/Configuration.h"
#include "lucid/util/error.h"
#include "lucid/util/logging.h"

using namespace lucid;  // NOLINT

#pragma GCC diagnostic ignored "-Wunused-variable"
#pragma GCC diagnostic ignored "-Wunused-parameter"
#pragma GCC diagnostic ignored "-Wunused-but-set-variable"
#pragma GCC diagnostic ignored "-Wunused-function"

namespace {

std::unique_ptr<Optimiser> get_optimiser(const Configuration::Optimiser solver, const Configuration& args) {
  switch (solver) {
#ifdef LUCID_GUROBI_BUILD
    case Configuration::Optimiser::GUROBI:
      return std::make_unique<GurobiOptimiser>(args.problem_log_file, args.iis_log_file);
#endif
#ifdef LUCID_ALGLIB_BUILD
    case Configuration::Optimiser::ALGLIB:
      return std::make_unique<AlglibOptimiser>(args.problem_log_file, args.iis_log_file);
#endif
#ifdef LUCID_HIGHS_BUILD
    case Configuration::Optimiser::HIGHS:
      return std::make_unique<HighsOptimiser>(args.problem_log_file, args.iis_log_file);
#endif
#ifdef LUCID_SOPLEX_BUILD
    case Configuration::Optimiser::SOPLEX:
      return std::make_unique<SoplexOptimiser>(args.problem_log_file, args.iis_log_file);
#endif
    default:
      throw std::invalid_argument("Solver not supported or not built");
  }
}

std::unique_ptr<Estimator> get_estimator(const Configuration::Estimator estimator_type, const Configuration& args) {
  switch (estimator_type) {
    case Configuration::Estimator::MODEL_ESTIMATOR:
      return std::make_unique<ModelEstimator>(args.system_dynamics);
    case Configuration::Estimator::KERNEL_RIDGE_REGRESSOR:
      return std::make_unique<KernelRidgeRegressor>(std::make_unique<GaussianKernel>(args.sigma_l, args.sigma_f),
                                                    args.lambda);
    default:
      throw std::invalid_argument("Estimator not supported");
  }
}

std::unique_ptr<TruncatedFourierFeatureMap> get_feature_map(const Configuration::FeatureMap feature_map_type,
                                                            const Configuration& args, const RectSet& X_bounds) {
  switch (feature_map_type) {
    case Configuration::FeatureMap::LINEAR_TRUNCATED_FOURIER_FEATURE_MAP:
      return std::make_unique<LinearTruncatedFourierFeatureMap>(args.num_frequencies, args.feature_sigma_l,
                                                                args.sigma_f, X_bounds);
    case Configuration::FeatureMap::CONSTANT_TRUNCATED_FOURIER_FEATURE_MAP:
      return std::make_unique<ConstantTruncatedFourierFeatureMap>(args.num_frequencies, args.feature_sigma_l,
                                                                  args.sigma_f, X_bounds);
    case Configuration::FeatureMap::LOG_TRUNCATED_FOURIER_FEATURE_MAP:
      return std::make_unique<LogTruncatedFourierFeatureMap>(args.num_frequencies, args.feature_sigma_l, args.sigma_f,
                                                             X_bounds);
    default:
      throw std::invalid_argument("Feature map not supported");
  }
}

bool pipeline(const Configuration& args) {
  LUCID_LOG_INIT_VERBOSITY(args.verbose);
  random::seed(args.seed);

  std::cout << args << std::endl;
  const RectSet& X_bounds = *dynamic_cast<RectSet*>(args.X_bounds.get());

  auto f = [&args](const Matrix& x) -> Matrix {
    std::normal_distribution d{0.0, args.noise_scale};
    // Add noise to the linear function
    const Matrix y{args.system_dynamics(x)};
    return args.system_dynamics(x) +
           Matrix::NullaryExpr(y.rows(), y.cols(), [&d](Index, Index) { return d(random::gen); });
  };

  const Matrix x_samples{args.X_bounds->sample(args.num_samples)};
  const Matrix xp_samples{f(x_samples)};

  const std::unique_ptr<Estimator> estimator{get_estimator(args.estimator, args)};
  const std::unique_ptr<TruncatedFourierFeatureMap> feature_map{get_feature_map(args.feature_map, args, X_bounds)};
  estimator->fit(x_samples, (*feature_map)(xp_samples));

  FourierBarrierCertificate barrier{args.time_horizon, args.gamma};
  const bool res = barrier.synthesize(*get_optimiser(args.optimiser, args), args.lattice_resolution, *estimator,
                                      *feature_map, X_bounds, *args.X_init, *args.X_unsafe,
                                      {
                                          .set_scaling = args.set_scaling,
                                          .C_coeff = args.C_coeff,
                                          .epsilon = args.epsilon,
                                          .b_norm = args.b_norm,
                                          .kappa = args.b_kappa,
                                      });
  LUCID_INFO_FMT("Synthesized Fourier barrier certificate:\n{}", barrier);
  return res;
}

}  // namespace

Configuration linear{
    .seed = 42,
    .problem_log_file = "problem.lp",
    .iis_log_file = "iis.ilp",
    .system_dynamics = [](const Matrix& x) -> Matrix { return x * 0.5; },
    .X_bounds = std::make_unique<RectSet>(std::vector<std::pair<Scalar, Scalar>>{{-1, 1}}),
    .X_init = std::make_unique<RectSet>(std::vector<std::pair<Scalar, Scalar>>{{-0.5, 0.5}}),
    .X_unsafe = std::make_unique<MultiSet>(RectSet{{-1, -0.9}}, RectSet{{0.9, 1}}),
    .num_samples = 100,
    .noise_scale = 0.01,
    .lambda = 1e-5,
    .sigma_f = 1.0,
    .sigma_l = Vector::Constant(1, 0.29196529402181604),
    .feature_sigma_l = Vector::Constant(1, 0.06),
    .num_frequencies = 8,
    .lattice_resolution = 700,
    .gamma = 1.0,
    .time_horizon = 15,
    .b_norm = 2,
    .set_scaling = 0.02,
};
Configuration barrier2{
    .seed = 42,
    .problem_log_file = "problem.lp",
    .iis_log_file = "iis.ilp",
    .system_dynamics = [](const Matrix& x) -> Matrix {
      // x1 = "x1 + 0.1 * (x2 - 1 + exp(-x1))"
      // x2 = "x2 + 0.1 * (-sin(x1)**2)"
      // out.col(0) = x.col(0).array() + 0.1 * (x.col(1).array() - 1 + (-x.col(0)).array().exp());
      // out.col(1) = x.col(1).array() + 0.1 * -x.col(0).array().sin().square();
      return Matrix::NullaryExpr(x.rows(), x.cols(), [&x](const Index row, const Index col) {
        return col == 0 ? x(row, 0) + 0.1 * (x(row, 1) - 1 + std::exp(-x(row, 0)))
                        : x(row, 1) + 0.1 * std::sin(x(row, 0)) * std::sin(x(row, 0));
      });
    },
    .X_bounds = std::make_unique<RectSet>(std::vector<std::pair<Scalar, Scalar>>{{-2, 2}, {-2, 2}}),
    .X_init = std::make_unique<SphereSet>(Vector2{-0.5, -0.5}, 0.4),
    .X_unsafe = std::make_unique<SphereSet>(Vector2{0.7, -0.7}, 0.3),
    .num_samples = 500,
    .noise_scale = 0.01,
    .lambda = 1.0e-06,
    .sigma_f = 15.0,
    .sigma_l = Vector2{2.50304, 3.77779},
    .feature_sigma_l = Vector2{2.50304, 3.77779},
    .num_frequencies = 6,
    .oversample_factor = 2.0,
    .lattice_resolution = 150,
    .gamma = 1.0,
    .time_horizon = 5,
};
Configuration barrier3{
    .seed = 42,
    .problem_log_file = "problem.lp",
    .iis_log_file = "iis.ilp",
    .system_dynamics = [](const Matrix& x) -> Matrix {
      // x1 = x1 + 0.1 * x2
      // x2 = x2 + 0.1 * (-x1 - x2 + 1 / 3 * x1 ** 3)
      return Matrix::NullaryExpr(x.rows(), x.cols(), [&x](const Index row, const Index col) {
        const double x1 = x(row, 0);
        const double x2 = x(row, 1);
        return col == 0 ? x1 + 0.1 * x2  //
                        : x2 + 0.1 * (-x1 - x2 + 1.0 / 3.0 * std::pow(x1, 3.0));
      });
    },
    .X_bounds = std::make_unique<RectSet>(Vector2{-3, -2}, Vector2{2.5, 1}),
    .X_init = std::make_unique<RectSet>(Vector2{-1.8, -0.1}, Vector2{-1.4, 0.1}),
    .X_unsafe = std::make_unique<RectSet>(Vector2{0.6, 0.2}, Vector2{0.7, 0.4}),
    .num_samples = 1000,
    .noise_scale = 0.01,
    .lambda = 1.e-8,
    .sigma_f = 1.0,
    .sigma_l = Vector2{0.1, 0.1},
    .feature_sigma_l = Vector2{0.1, 0.1},
    .num_frequencies = 10,
    .oversample_factor = 2.0,
    .lattice_resolution = 150,
    .gamma = 1.0,
    .time_horizon = 5,
    .set_scaling = 0.03,
};

/**
 * Main function.
 * @param argc Number of arguments.
 * @param argv Arguments.
 * @return Execution status.
 */
int main(const int argc, char* argv[]) {
  if (argc < 2) {
    fmt::println("Usage: {} <linear|barrier2|barrier3> [gurobi|alglib|highs|soplex]", argv[0]);
    return 1;
  }

  Configuration file_args;
  Configuration* args = nullptr;
  if (std::string_view{argv[1]} == "linear") {  // NOLINT(whitespace/braces): standard initialisation
    args = &linear;
  } else if (std::string_view{argv[1]} == "barrier2") {  // NOLINT(whitespace/braces): standard initialisation
    args = &barrier2;
  } else if (std::string_view{argv[1]} == "barrier3") {  // NOLINT(whitespace/braces): standard initialisation
    args = &barrier3;
  } else {
    file_args = Configuration::from_yaml(argv[1]);
    args = &file_args;
  }
  if (argc >= 3) {
    if (std::string_view{argv[2]} == "alglib") {  // NOLINT(whitespace/braces): standard initialisation
      args->optimiser = Configuration::Optimiser::ALGLIB;
    } else if (std::string_view{argv[2]} == "gurobi") {  // NOLINT(whitespace/braces): standard initialisation
      args->optimiser = Configuration::Optimiser::GUROBI;
    } else if (std::string_view{argv[2]} == "highs") {  // NOLINT(whitespace/braces): standard initialisation
      args->optimiser = Configuration::Optimiser::HIGHS;
    } else if (std::string_view{argv[2]} == "soplex") {  // NOLINT(whitespace/braces): standard initialisation
      args->optimiser = Configuration::Optimiser::SOPLEX;
    } else {
      fmt::println("Usage: {} <linear|barrier2|barrier3> [gurobi|alglib|highs|soplex]", argv[0]);
      return 1;
    }
  }

  pipeline(*args);
  return 0;
}

// #pragma GCC diagnostic pop
