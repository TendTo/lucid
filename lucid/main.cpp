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
#include <span>
#include <string>
#include <utility>
#include <vector>

#include "lucid/lucid.h"
#include "lucid/util/error.h"
#include "lucid/util/logging.h"

using namespace lucid;  // NOLINT

#pragma GCC diagnostic ignored "-Wunused-variable"
#pragma GCC diagnostic ignored "-Wunused-parameter"
#pragma GCC diagnostic ignored "-Wunused-but-set-variable"
#pragma GCC diagnostic ignored "-Wunused-function"

namespace {

enum class Solver { Gurobi, Alglib, HiGHS, SOPLEX };

struct CliArgs {
  int seed{-1};
  double increase{0.1};
  double gamma{1.0};
  int time_horizon{5};
  int num_samples{1000};
  double lambda{1e-6};
  double sigma_f{1.0};
  std::vector<double> sigma_l{1.0};
  int num_frequencies{4};
  double C_coeff{1.0};
  bool plot{false};
  bool verify{false};
  std::string problem_log_file{""};
  std::string iis_log_file{""};
  double oversample_factor{2.0};
  int lattice_resolution{-1};
  double noise_scale{0.01};
  Solver solver{Solver::Gurobi};
  std::unique_ptr<RectSet> X_bounds;
  std::unique_ptr<Set> X_init;
  std::unique_ptr<Set> X_unsafe;
  std::function<Matrix(const Matrix&)> f_det;
};

Vector to_eigen(const std::vector<double>& vectors) {
  return Vector::NullaryExpr(vectors.size(), [&vectors](const Index i) { return vectors[i]; });
}

std::unique_ptr<Optimiser> get_optimiser(const Solver solver, const CliArgs& args) {
  switch (solver) {
#ifdef LUCID_GUROBI_BUILD
    case Solver::Gurobi:
      return std::make_unique<GurobiOptimiser>(args.problem_log_file, args.iis_log_file);
#endif
#ifdef LUCID_ALGLIB_BUILD
    case Solver::Alglib:
      return std::make_unique<AlglibOptimiser>(args.problem_log_file, args.iis_log_file);
#endif
#ifdef LUCID_HIGHS_BUILD
    case Solver::HiGHS:
      return std::make_unique<HighsOptimiser>(args.problem_log_file, args.iis_log_file);
#endif
#ifdef LUCID_SOPLEX_BUILD
    case Solver::SOPLEX:
      return std::make_unique<SoplexOptimiser>(args.problem_log_file, args.iis_log_file);
#endif
    default:
      throw std::invalid_argument("Solver not supported or not built");
  }
}

bool pipeline(const CliArgs& args) {
  random::seed(args.seed);

  auto f = [&args](const Matrix& x) -> Matrix {
    std::normal_distribution d{0.0, args.noise_scale};
    // Add noise to the linear function
    const Matrix y{args.f_det(x)};
    return args.f_det(x) + Matrix::NullaryExpr(y.rows(), y.cols(), [&d](Index, Index) { return d(random::gen); });
  };

  const Matrix x_samples{args.X_bounds->sample(args.num_samples)};
  const Matrix xp_samples{f(x_samples)};

  Vector sigma_l{to_eigen(args.sigma_l)};
  KernelRidgeRegressor estimator{std::make_unique<GaussianKernel>(sigma_l, args.sigma_f), args.lambda};
  LinearTruncatedFourierFeatureMap feature_map{args.num_frequencies, sigma_l, args.sigma_f, *args.X_bounds};
  ModelEstimator model_estimator{[&args, &feature_map](const Matrix& x) { return feature_map(args.f_det(x)); }};

  FourierBarrierCertificate barrier{args.time_horizon, args.gamma};
  const bool res = barrier.synthesize(*get_optimiser(args.solver, args), args.lattice_resolution, model_estimator,
                                      feature_map, *args.X_bounds, *args.X_init, *args.X_unsafe,
                                      {
                                          .increase = args.increase,
                                          .ftol = 1e-10,
                                          .xtol = 1e-10,
                                          .C_coeff = args.C_coeff,
                                          .b_norm = 0.0,
                                      });
  LUCID_INFO_FMT("Synthesized Fourier barrier certificate:\n{}", barrier);
  return res;
}

}  // namespace

CliArgs linear{.seed = 42,
               .gamma = 1.0,
               .time_horizon = 15,
               .num_samples = 1000,
               .lambda = 1e-6,
               .sigma_f = 15.0,
               .sigma_l = {1.2},
               .num_frequencies = 5,
               .plot = true,
               .verify = true,
               .problem_log_file = "problem.lp",
               .iis_log_file = "iis.ilp",
               .lattice_resolution = 704,
               .noise_scale = 0.01,
               .X_bounds = std::make_unique<RectSet>(std::vector<std::pair<Scalar, Scalar>>{{-1, 1}}),
               .X_init = std::make_unique<RectSet>(std::vector<std::pair<Scalar, Scalar>>{{-0.5, 0.5}}),
               .X_unsafe = std::make_unique<MultiSet>(RectSet{{-1, -0.9}}, RectSet{{0.9, 1}}),
               .f_det = [](const Matrix& x) -> Matrix { return x * 0.5; }};
CliArgs barrier2{.seed = 42,
                 .gamma = 2.0,
                 .time_horizon = 5,
                 .num_samples = 500,
                 .lambda = 1.0e-06,
                 .sigma_f = 15.0,
                 .sigma_l = {2.50304, 3.77779},
                 .num_frequencies = 6,
                 .plot = true,
                 .verify = true,
                 .problem_log_file = "problem.lp",
                 .iis_log_file = "iis.ilp",
                 .oversample_factor = 2.0,
                 .lattice_resolution = 150,
                 .noise_scale = 0.01,
                 .X_bounds = std::make_unique<RectSet>(std::vector<std::pair<Scalar, Scalar>>{{-2, 2}, {-2, 2}}),
                 .X_init = std::make_unique<SphereSet>(Vector2{-0.5, -0.5}, 0.4),
                 .X_unsafe = std::make_unique<SphereSet>(Vector2{0.7, -0.7}, 0.3),
                 .f_det = [](const Matrix& x) -> Matrix {
                   // x1 = "x1 + 0.1 * (x2 - 1 + exp(-x1))"
                   // x2 = "x2 + 0.1 * (-sin(x1)**2)"
                   // out.col(0) = x.col(0).array() + 0.1 * (x.col(1).array() - 1 + (-x.col(0)).array().exp());
                   // out.col(1) = x.col(1).array() + 0.1 * -x.col(0).array().sin().square();
                   return Matrix::NullaryExpr(x.rows(), x.cols(), [&x](const Index row, const Index col) {
                     return col == 0 ? x(row, 0) + 0.1 * (x(row, 1) - 1 + std::exp(-x(row, 0)))
                                     : x(row, 1) + 0.1 * std::sin(x(row, 0)) * std::sin(x(row, 0));
                   });
                 }};
CliArgs barrier3{.seed = 42,
                 .increase = 0.03,
                 .gamma = 2.0,
                 .time_horizon = 5,
                 .num_samples = 1000,
                 .lambda = 1.e-8,
                 .sigma_f = 1.0,
                 .sigma_l = {0.1, 0.1},
                 .num_frequencies = 10,
                 .plot = true,
                 .verify = true,
                 .problem_log_file = "problem.lp",
                 .iis_log_file = "iis.ilp",
                 .oversample_factor = 2.0,
                 .lattice_resolution = 150,
                 .noise_scale = 0.01,
                 .X_bounds = std::make_unique<RectSet>(Vector2{-3, -2}, Vector2{2.5, 1}),
                 .X_init = std::make_unique<RectSet>(Vector2{-1.8, -0.1}, Vector2{-1.4, 0.1}),
                 .X_unsafe = std::make_unique<RectSet>(Vector2{0.6, 0.2}, Vector2{0.7, 0.4}),
                 .f_det = [](const Matrix& x) -> Matrix {
                   // x1 = x1 + 0.1 * x2
                   // x2 = x2 + 0.1 * (-x1 - x2 + 1 / 3 * x1 ** 3)
                   return Matrix::NullaryExpr(x.rows(), x.cols(), [&x](const Index row, const Index col) {
                     const double x1 = x(row, 0);
                     const double x2 = x(row, 1);
                     return col == 0 ? x1 + 0.1 * x2  //
                                     : x2 + 0.1 * (-x1 - x2 + 1.0 / 3.0 * std::pow(x1, 3.0));
                   });
                 }};

/**
 * Main function.
 * @param argc Number of arguments.
 * @param argv Arguments.
 * @return Execution status.
 */
int main(const int argc, char* argv[]) {
  CliArgs* args = nullptr;
  if (argc >= 2) {
    if (std::string_view{argv[1]} == "linear") {  // NOLINT(whitespace/braces): standard initialisation
      args = &linear;
    } else if (std::string_view{argv[1]} == "barrier2") {  // NOLINT(whitespace/braces): standard initialisation
      args = &barrier2;
    } else if (std::string_view{argv[1]} == "barrier3") {  // NOLINT(whitespace/braces): standard initialisation
      args = &barrier3;
    } else {
      fmt::println("Usage: {} <linear|barrier2|barrier3> [gurobi|alglib|highs|soplex]", argv[0]);
      return 1;
    }
  }
  if (argc >= 3) {
    if (std::string_view{argv[2]} == "alglib") {  // NOLINT(whitespace/braces): standard initialisation
      args->solver = Solver::Alglib;
    } else if (std::string_view{argv[2]} == "gurobi") {  // NOLINT(whitespace/braces): standard initialisation
      args->solver = Solver::Gurobi;
    } else if (std::string_view{argv[2]} == "highs") {  // NOLINT(whitespace/braces): standard initialisation
      args->solver = Solver::HiGHS;
    } else if (std::string_view{argv[2]} == "soplex") {  // NOLINT(whitespace/braces): standard initialisation
      args->solver = Solver::SOPLEX;
    } else {
      fmt::println("Usage: {} <linear|barrier2|barrier3> [gurobi|alglib|highs|soplex]", argv[0]);
      return 1;
    }
  }
  if (argc < 2) {
    fmt::println("Usage: {} <linear|barrier2|barrier3> [gurobi|alglib|highs|soplex]", argv[0]);
    return 1;
  }

  LUCID_LOG_INIT_VERBOSITY(4);
  args->problem_log_file = fmt::format("/home/campus.ncl.ac.uk/c3054737/Programming/phd/keid/{}.problem.lp",
                                       args->solver == Solver::Gurobi   ? "gurobi"
                                       : args->solver == Solver::Alglib ? "alglib"
                                       : args->solver == Solver::HiGHS  ? "highs"
                                                                        : "soplex");

  pipeline(*args);
  return 0;
}

// #pragma GCC diagnostic pop
