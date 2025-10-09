/**
 * @file main.cpp
 * @author Room 6.030
 * @copyright 2025 lucid
 * @licence Apache-2.0 license
 * @file
 */
#include <chrono>
#include <iostream>
#include <numbers>
#include <span>
#include <string>
#include <utility>
#include <vector>

#include "lucid/lucid.h"
#include "lucid/util/error.h"
#include "lucid/util/logging.h"

#ifdef LUCID_MATPLOTLIB_BUILD
#include "lucid/util/matplotlib.h"
#endif

using namespace lucid;  // NOLINT

#pragma GCC diagnostic ignored "-Wunused-variable"
#pragma GCC diagnostic ignored "-Wunused-parameter"
#pragma GCC diagnostic ignored "-Wunused-but-set-variable"
#pragma GCC diagnostic ignored "-Wunused-function"

namespace {

#if 0
void plot_rect_set(const RectSet& set, const std::string& color = "blue", const double alpha = 0.9) {
  LUCID_CHECK_ARGUMENT_EQ(set.dimension(), 2);
  Vector x(set.dimension());
  x << set.lower_bound()(0), set.upper_bound()(0);
  Vector y1(1);
  y1 << set.lower_bound()(1);
  Vector y2(1);
  y2 << set.upper_bound()(1);
  plt::fill_between(x, y1, y2, {.alpha = alpha, .edgecolor = color});
}
void plot_multi_set(const std::vector<RectSet>& sets, const std::string& color = "blue", const double alpha = 0.9) {
  for (const auto& set : sets) {
    plot_rect_set(set, color, alpha);
  }
}

void test_surface() {
  Matrix X, Y;
  meshgrid(arange(-5, 5, 0.5), arange(-5, 5, 0.5), X, Y);
  Matrix Z = peaks(X, Y);
  plt::plot_wireframe(X, Y, Z);
}

void plot_points(const Matrix& points, const std::string& color = "blue", const double size = 10) {
  if (points.cols() != 2) throw std::invalid_argument("points must have 2 rows");
  const Vector x = points.col(0);
  const Vector y = points.col(1);
  plt::scatter(x, y, {.s = size, .c = color});
}

void test_barrier_3_old() {
  // Generic test
  const benchmark::InitBarr3Scenario scenario;
  scenario.plot();
  Matrix inputs, outputs;
  scenario.sample_transition(1000, inputs, outputs);
  plot_points(inputs, "blue");
  plot_points(outputs, "magenta");

  constexpr double regularization_constant = 0.0000001;
  const GaussianKernel kernel{1, Vector::Constant(scenario.dimension(), 0.3)};
  const KernelRidgeRegression regression{kernel, inputs, outputs, regularization_constant};

  // const Matrix samples = scenario.sample(10);
  // plot_points(samples, "green");
  Matrix res = regression(inputs);
  std::cout << "RMS: " << rms(res - outputs) << std::endl;
  plot_points(res, "cyan", 2);

  plt::figure(2);
  scenario.sample_transition(1000, inputs, outputs);
  scenario.plot();
  plot_points(inputs, "blue");
  plot_points(outputs, "magenta");

  res = regression(inputs);
  std::cout << res.transpose() << std::endl;
  std::cout << "RMS: " << rms(res - outputs) << std::endl;
  plot_points(res, "cyan", 2);

  plt::figure(3);
  plt::scatter(inputs.col(0), inputs.col(1), outputs.col(0));

  plt::figure(4);
  plt::scatter(static_cast<Vector>(inputs.col(0)), static_cast<Vector>(inputs.col(1)), static_cast<Vector>(res.col(0)));
  plt::show();
}

Matrix wavelengths(const Dimension dimension, const int num_frequencies_per_dimension) {
  if (dimension < 1) throw std::invalid_argument("dimension must be at least 1");
  // Compute the total number of columns of the output matrix
  const Vector frequencies{arange(0, num_frequencies_per_dimension)};

  Matrix frequency_combination = frequencies.transpose();
  for (Dimension i = 1; i < dimension; i++) {
    frequency_combination = combvec(frequency_combination, frequencies.transpose());
  }
  return (frequency_combination.rightCols(frequency_combination.cols() - 1) * 2 * std::numbers::pi).transpose();
}

Basis generate_basis(ConstMatrixRef omega_T, const Dimension dimension, Scalar sigma_f, ConstVectorRef sigma_l,
                     const int num_frequencies_per_dimension) {
  if (sigma_l.size() < 1) throw std::invalid_argument("sigma_l must have at least one element");

  const Vector omega_per_dim_lb =
      (2 * std::numbers::pi * arange(0, num_frequencies_per_dimension)).array() - std::numbers::pi;
  const Vector omega_per_dim_ub = omega_per_dim_lb.array() + 2 * std::numbers::pi;

  Matrix prob_per_dim{dimension, num_frequencies_per_dimension};
  for (Dimension i = 0; i < dimension; i++) {
    prob_per_dim.row(i) = normal_cdf(omega_per_dim_ub, 0, sigma_l(i)) - normal_cdf(omega_per_dim_lb, 0, sigma_l(i));
    prob_per_dim.row(i).rightCols(prob_per_dim.cols() - 1) *= 2;
  }

  auto prod_ND = combvec(prob_per_dim);
  auto prod = prod_ND.colwise().prod().transpose();
  if (Scalar sum = prod.sum(); sum > 0.9)
    LUCID_INFO_FMT("Probability captured by Fourier expansion is {:.3f} percent", sum);
  else
    LUCID_WARN_FMT("Probability captured by Fourier expansion is only {:.3f} percent", sum);
  auto temp = prod.transpose().cwiseSqrt();

  Vector weights{temp.size() * 2 - 1};
  // Repeat each column twice, except the first one
  for (Index i = 0; i < temp.size(); i++) {
    weights(2 * i) = temp(i);
    if (i != 0) weights(2 * i - 1) = temp(i);
  }

  return [weights = std::move(weights), omega_T = omega_T.eval(), sigma_f](ConstVectorRef x, ConstMatrixRef x_limits) {
    // TODO(tend): this only works for rectangles sets
    // Project the element onto the unit hypercube
    auto z = (x.transpose() - x_limits.row(0)).cwiseQuotient(x_limits.row(1) - x_limits.row(0));

    Vector z_proj = omega_T * z.transpose();
    Vector trig{2 * z_proj.size() + 1};
    trig(0) = 1;
    for (Index i = 0; i < z_proj.size(); i++) {
      trig(2 * i + 1) = std::cos(z_proj(i));
      trig(2 * i + 2) = std::sin(z_proj(i));
    }

    auto basis = sigma_f * weights.cwiseProduct(trig);
    if (Scalar checksum = (basis.cwiseProduct(basis).colwise().sum().array().sqrt() - sigma_f).abs().maxCoeff();
        checksum > 1e-3) {
      LUCID_WARN_FMT("Checksum failed: Fourier basis frequency bands don't add up: {} > 1e-3", checksum);
    }
    // LUCID_ASSERT((basis.cwiseProduct(basis).colwise().sum().array().sqrt() - sigma_f).abs().maxCoeff() <= 1e-3,
    //              "Checksum failed: Fourier basis frequency bands don't add up");
    return Matrix{basis};
  };
}

Basis generate_basis(ConstMatrixRef omega_T, const Dimension dimension, const Scalar sigma_f, const Scalar sigma_l,
                     const int num_frequencies_per_dimension) {
  return generate_basis(omega_T, dimension, sigma_f, Vector::Constant(dimension, sigma_l),
                        num_frequencies_per_dimension);
}
#endif

enum class Solver { Gurobi, Alglib, HiGHS, SOPLEX };

struct CliArgs {
  int seed{-1};
  double gamma{1.0};
  int time_horizon{5};
  int num_samples{1000};
  double lambda{1e-6};
  double sigma_f{1.0};
  std::vector<double> sigma_l{1.0};
  int num_frequencies{4};
  double c_coefficient{1.0};
  bool plot{false};
  bool verify{false};
  std::string problem_log_file{""};
  std::string iis_log_file{""};
  double oversample_factor{2.0};
  int num_oversample{-1};
  double noise_scale{0.01};
  Solver solver{Solver::Gurobi};
  std::unique_ptr<RectSet> X_bounds;
  std::unique_ptr<Set> X_init;
  std::unique_ptr<Set> X_unsafe;
  std::function<Matrix(const Matrix&)> f_det;
};

#if 0
bool test_overtaking(const CliArgs& args) {
  const auto start_time = std::chrono::steady_clock::now().time_since_epoch().count();
  random::seed(args.seed);

  const RectSet X_bounds{{{1, 90}, {-7, 19}, {-std::numbers::pi, std::numbers::pi}}};
  const RectSet X_init{{{1, 2}, {-0.5, 0.5}, {-0.005, 0.005}}};
  const MultiSet X_unsafe{RectSet{{1, 90}, {-7, -6}, {-std::numbers::pi, std::numbers::pi}},
                          RectSet{{1, 90}, {18, 19}, {-std::numbers::pi, std::numbers::pi}},
                          RectSet{{40, 45}, {-6, 6}, {-std::numbers::pi, std::numbers::pi}}};

  const Matrix x_samples{X_bounds.sample(args.num_samples)};
  const Matrix xp_samples{x_samples + Matrix::Constant(x_samples.rows(), x_samples.cols(), 0.1)};
  KernelRidgeRegressor estimator{std::make_unique<GaussianKernel>(args.sigma_l, args.sigma_f), args.lambda};
  LinearTruncatedFourierFeatureMap feature_map{args.num_frequencies, args.sigma_l, args.sigma_f, X_bounds};

  const Matrix f_xp_samples{feature_map(xp_samples)};

  const int n_per_dim = args.num_oversample < 0
                            ? static_cast<int>(std::ceil((2 * args.num_frequencies + 1) * args.oversample_factor))
                            : args.num_oversample;
  LUCID_DEBUG_FMT("Number of samples per dimension: {}", n_per_dim);
  LUCID_ASSERT(n_per_dim > 2 * args.num_frequencies,
               "n_per_dim must be greater than nyquist (2 * num_freq_per_dim + 1)");

  LUCID_DEBUG_FMT("Estimator pre-fit: {}", estimator);
  estimator.fit(x_samples, f_xp_samples);
  LUCID_INFO_FMT("Estimator post-fit: {}", estimator);

  LUCID_DEBUG_FMT("Feature map: {}", feature_map);

  const Matrix x_lattice = X_bounds.lattice(n_per_dim, true);
  const Matrix u_f_x_lattice = feature_map(x_lattice);
  Matrix u_f_xp_lattice_via_regressor = estimator(x_lattice);
  // We are fixing the zero frequency to the constant value we computed in the feature map
  // If we don't, the regressor has a hard time learning it on the extreme left and right points, because it tends to 0
  u_f_xp_lattice_via_regressor.col(0).array() = feature_map.weights()[0] * args.sigma_f;

  const Matrix x0_lattice = X_init.lattice(n_per_dim, true);
  const Matrix f_x0_lattice = feature_map(x0_lattice);

  const Matrix xu_lattice = X_unsafe.lattice(n_per_dim, true);
  const Matrix f_xu_lattice = feature_map(xu_lattice);

  LUCID_INFO_FMT("End of operation, took {:.3f} seconds",
                 (std::chrono::steady_clock::now().time_since_epoch().count() - start_time) / 1e9);
  return false;
}


switch (args.solver) {
#ifdef LUCID_GUROBI_BUILD
  case Solver::Gurobi:
    return GurobiOptimiser{
      args.time_horizon,     args.gamma,        0, 1, 1, args.sigma_f, args.c_coefficient,
      args.problem_log_file, args.iis_log_file,
  }
    .solve(f_x0_lattice, f_xu_lattice, u_f_x_lattice, u_f_xp_lattice_via_regressor, feature_map.dimension(),
           args.num_frequencies - 1, n_per_dim, args.X_bounds->dimension(), check_cb);
#endif
#ifdef LUCID_ALGLIB_BUILD
  case Solver::Alglib:
    return AlglibOptimiser{
      args.time_horizon,     args.gamma,        0, 1, 1, args.sigma_f, args.c_coefficient,
      args.problem_log_file, args.iis_log_file,
  }
    .solve(f_x0_lattice, f_xu_lattice, u_f_x_lattice, u_f_xp_lattice_via_regressor, feature_map.dimension(),
           args.num_frequencies - 1, n_per_dim, args.X_bounds->dimension(), check_cb);
#endif
#ifdef LUCID_HIGHS_BUILD
  case Solver::HiGHS:
    return HighsOptimiser{
      args.time_horizon,     args.gamma,        0, 1, 1, args.sigma_f, args.c_coefficient,
      args.problem_log_file, args.iis_log_file,
  }
    .solve(f_x0_lattice, f_xu_lattice, u_f_x_lattice, u_f_xp_lattice_via_regressor, feature_map.dimension(),
           args.num_frequencies - 1, n_per_dim, args.X_bounds->dimension(), check_cb);
#endif
  default:
    LUCID_NOT_SUPPORTED("The chosen solver");
}
#endif

Vector to_eigen(const std::vector<double>& vectors) {
  return Vector::NullaryExpr(vectors.size(), [&vectors](const Index i) { return vectors[i]; });
}

std::unique_ptr<Optimiser> get_optimiser(const Solver solver, const CliArgs& args) {
  switch (solver) {
#ifdef LUCID_GUROBI_BUILD
    case Solver::Gurobi:
      return std::make_unique<GurobiOptimiser>(args.time_horizon, args.gamma, 0, 1, 1, args.sigma_f, args.c_coefficient,
                                               args.problem_log_file, args.iis_log_file);
#endif
#ifdef LUCID_ALGLIB_BUILD
    case Solver::Alglib:
      return std::make_unique<AlglibOptimiser>(args.time_horizon, args.gamma, 0, 1, 1, args.sigma_f, args.c_coefficient,
                                               args.problem_log_file, args.iis_log_file);
#endif
#ifdef LUCID_HIGHS_BUILD
    case Solver::HiGHS:
      return std::make_unique<HighsOptimiser>(args.time_horizon, args.gamma, 0, 1, 1, args.sigma_f, args.c_coefficient,
                                              args.problem_log_file, args.iis_log_file);
#endif
#ifdef LUCID_SOPLEX_BUILD
    case Solver::SOPLEX:
      return std::make_unique<SoplexOptimiser>(args.time_horizon, args.gamma, 0, 1, 1, args.sigma_f, args.c_coefficient,
                                               args.problem_log_file, args.iis_log_file);
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

  const Matrix f_xp_samples{feature_map(xp_samples)};

  const int n_per_dim = args.num_oversample < 0
                            ? static_cast<int>(std::ceil((2 * args.num_frequencies + 1) * args.oversample_factor))
                            : args.num_oversample;
  LUCID_DEBUG_FMT("Number of samples per dimension: {}", n_per_dim);
  LUCID_ASSERT(n_per_dim > 2 * args.num_frequencies,
               "n_per_dim must be greater than nyquist (2 * num_freq_per_dim + 1)");

  LUCID_DEBUG_FMT("Estimator pre-fit: {}", estimator);
  estimator.fit(x_samples, f_xp_samples);
  LUCID_INFO_FMT("Estimator post-fit: {}", estimator);

#ifdef COMPUTE_TRAINING_ERROR
  {
    LUCID_DEBUG_FMT("RMSE on f_xp_samples {}", scorer::rmse_score(estimator, x_samples, f_xp_samples));
    LUCID_DEBUG_FMT("Score on f_xp_samples {}", estimator.score(x_samples, f_xp_samples));
    Matrix x_evaluation = args.X_bounds->sample(x_samples.rows() / 2);
    Matrix f_xp_evaluation = feature_map(args.f_det(x_evaluation));
    LUCID_DEBUG_FMT("RMSE on f_det_evaluated {}", scorer::rmse_score(estimator, x_evaluation, f_xp_evaluation));
    LUCID_DEBUG_FMT("Score on f_det_evaluated {}", estimator.score(x_evaluation, f_xp_evaluation));
  }
#endif

  LUCID_DEBUG_FMT("Feature map: {}", feature_map);

  const Matrix x_lattice = args.X_bounds->lattice(n_per_dim, true);
  const Matrix u_f_x_lattice = feature_map(x_lattice);
  Matrix u_f_xp_lattice_via_regressor = estimator(x_lattice);
  // We are fixing the zero frequency to the constant value we computed in the feature map
  // If we don't, the regressor has a hard time learning it on the extreme left and right points, because it tends to 0
  u_f_xp_lattice_via_regressor.col(0).array() = feature_map.weights()[0] * args.sigma_f;

  const Matrix x0_lattice = args.X_init->lattice(n_per_dim, true);
  const Matrix f_x0_lattice = feature_map(x0_lattice);

  const Matrix xu_lattice = args.X_unsafe->lattice(n_per_dim, true);
  const Matrix f_xu_lattice = feature_map(xu_lattice);

  FourierBarrierCertificate barrier{args.time_horizon, args.gamma};
  barrier.synthesize(*get_optimiser(args.solver, args), u_f_x_lattice, u_f_xp_lattice_via_regressor, f_x0_lattice,
                     f_xu_lattice, feature_map, n_per_dim, args.c_coefficient);
  std::cout << barrier << std::endl;
  return true;
}

void validator(const CliArgs& args) {
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

  const LeaveOneOut loo;
  const KFold kfold5{5};
  const LbfgsTuner tuner{{{0.1, 1}, {0.1, 15.0}, {0.1, 15.0}}, LbfgsParameters{.linesearch = 15, .min_step = 0}};
  // const MedianHeuristicTuner tuner;
  LUCID_INFO_FMT("Initial estimator: {}", estimator);

  kfold5.fit(estimator, x_samples, xp_samples, tuner, static_cast<scorer::ScorerType>(scorer::mse_score));
  LUCID_INFO_FMT("Estimator after LOO: {}", estimator);
}

}  // namespace

/**
 * Main function.
 * @param argc Number of arguments.
 * @param argv Arguments.
 * @return Execution status.
 */
int main(const int argc, char* argv[]) {
  Solver solver = Solver::Gurobi;
  if (argc > 1) {
    if (std::string_view{argv[1]} == "alglib") {  // NOLINT(whitespace/braces): standard initialisation
      solver = Solver::Alglib;
    } else if (std::string_view{argv[1]} == "gurobi") {  // NOLINT(whitespace/braces): standard initialisation
      solver = Solver::Gurobi;
    } else if (std::string_view{argv[1]} == "highs") {
      // NOLINT(whitespace/braces): standard initialisation
      solver = Solver::HiGHS;
    } else if (std::string_view{argv[1]} == "soplex") {
      solver = Solver::SOPLEX;
    } else {
      fmt::print("Usage: {} [gurobi|alglib|highs|soplex]\n", argv[0]);
      return 1;
    }
  }
  LUCID_LOG_INIT_VERBOSITY(4);
  const std::string log_file = fmt::format("/home/campus.ncl.ac.uk/c3054737/Programming/phd/keid/{}.problem.lp",
                                           solver == Solver::Gurobi   ? "gurobi"
                                           : solver == Solver::Alglib ? "alglib"
                                           : solver == Solver::HiGHS  ? "highs"
                                                                      : "soplex");

#if 1
  Vector centre_0{2}, centre_u{2};
  centre_0 << -0.5, -0.5;
  centre_u << 0.7, -0.7;
  validator(CliArgs{.seed = 42,
                    .gamma = 2.0,
                    .time_horizon = 5,
                    .num_samples = 100,
                    .lambda = 1.0e-06,
                    .sigma_f = 15.0,
                    .sigma_l = {1.18, 1.23},
                    .num_frequencies = 6,
                    .plot = true,
                    .verify = true,
                    .problem_log_file = log_file,
                    .iis_log_file = "iis.ilp",
                    .oversample_factor = 2.0,
                    .num_oversample = 150,
                    .noise_scale = 0.01,
                    .solver = solver,
                    .X_bounds = std::make_unique<RectSet>(std::vector<std::pair<Scalar, Scalar>>{{-2, 2}, {-2, 2}}),
                    .X_init = std::make_unique<SphereSet>(centre_0, 0.4),
                    .X_unsafe = std::make_unique<SphereSet>(centre_u, 0.3),
                    .f_det = [](const Matrix& x) -> Matrix {
                      // x1 = "x1 + 0.1 * (x2 - 1 + exp(-x1))"
                      // x2 = "x2 + 0.1 * (-sin(x1)**2)"
                      Matrix out{x.rows(), x.cols()};
                      // out.col(0) = x.col(0).array() + 0.1 * (x.col(1).array() - 1 + (-x.col(0)).array().exp());
                      // out.col(1) = x.col(1).array() + 0.1 * -x.col(0).array().sin().square();
                      out = Matrix::NullaryExpr(x.rows(), x.cols(), [&x](const Index row, Index col) {
                        return col == 0 ? x(row, 0) + 0.1 * (x(row, 1) - 1 + std::exp(-x(row, 0)))
                                        : x(row, 1) + 0.1 * std::sin(x(row, 0)) * std::sin(x(row, 0));
                      });
                      return out;
                    }});
#endif
#if 0
  pipeline({.seed = 42,
            .gamma = 1.0,
            .time_horizon = 15,
            .num_samples = 1000,
            .lambda = 1e-6,
            .sigma_f = 15.0,
            .sigma_l = {1.2},
            .num_frequencies = 5,
            .plot = true,
            .verify = true,
            .problem_log_file = log_file,
            .iis_log_file = "iis.ilp",
            .num_oversample = 704,
            .noise_scale = 0.01,
            .solver = solver,
            .X_bounds = std::make_unique<RectSet>(std::vector<std::pair<Scalar, Scalar>>{{-1, 1}}),
            .X_init = std::make_unique<RectSet>(std::vector<std::pair<Scalar, Scalar>>{{-0.5, 0.5}}),
            .X_unsafe = std::make_unique<MultiSet>(RectSet{{-1, -0.9}}, RectSet{{0.9, 1}}),
            .f_det = [](const Matrix& x) -> Matrix { return x * 0.5; }});
// #else
  Vector centre_0{2}, centre_u{2};
  centre_0 << -0.5, -0.5;
  centre_u << 0.7, -0.7;
  pipeline(CliArgs{.seed = 42,
                   .gamma = 2.0,
                   .time_horizon = 5,
                   .num_samples = 500,
                   .lambda = 1.0e-06,
                   .sigma_f = 15.0,
                   .sigma_l = {2.50304, 3.77779},
                   .num_frequencies = 6,
                   .plot = true,
                   .verify = true,
                   .problem_log_file = log_file,
                   .iis_log_file = "iis.ilp",
                   .oversample_factor = 2.0,
                   .num_oversample = 150,
                   .noise_scale = 0.01,
                   .solver = solver,
                   .X_bounds = std::make_unique<RectSet>(std::vector<std::pair<Scalar, Scalar>>{{-2, 2}, {-2, 2}}),
                   .X_init = std::make_unique<SphereSet>(centre_0, 0.4),
                   .X_unsafe = std::make_unique<SphereSet>(centre_u, 0.3),
                   .f_det = [](const Matrix& x) -> Matrix {
                     // x1 = "x1 + 0.1 * (x2 - 1 + exp(-x1))"
                     // x2 = "x2 + 0.1 * (-sin(x1)**2)"
                     Matrix out{x.rows(), x.cols()};
                     // out.col(0) = x.col(0).array() + 0.1 * (x.col(1).array() - 1 + (-x.col(0)).array().exp());
                     // out.col(1) = x.col(1).array() + 0.1 * -x.col(0).array().sin().square();
                     out = Matrix::NullaryExpr(x.rows(), x.cols(), [&x](const Index row, Index col) {
                       return col == 0 ? x(row, 0) + 0.1 * (x(row, 1) - 1 + std::exp(-x(row, 0)))
                                       : x(row, 1) + 0.1 * std::sin(x(row, 0)) * std::sin(x(row, 0));
                     });
                     return out;
                   }});
#endif

  return 0;
}

// #pragma GCC diagnostic pop
