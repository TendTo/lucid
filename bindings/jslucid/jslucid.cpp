/**
 * @author c3054737
 * @copyright 2025 lucid
 * @licence BSD 3-Clause License
 * @file
 * main class.
 */
#ifndef LUCID_JS_BUILD
#error LUCID_JS_BUILD is not defined. Ensure you are building with the option '--config=js'
#endif

#include <emscripten/bind.h>

#include <iostream>

#include "lucid/lib/eigen.h"
#include "lucid/model/model.h"
#include "lucid/util/util.h"
#include "lucid/verification/verification.h"
#include "lucid/version.h"

using namespace emscripten;
using namespace lucid;

#pragma GCC diagnostic ignored "-Wunused-variable"
#pragma GCC diagnostic ignored "-Wunused-parameter"
#pragma GCC diagnostic ignored "-Wunused-but-set-variable"
#pragma GCC diagnostic ignored "-Wunused-function"

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
  std::vector<std::pair<double, double>> X_bounds;
  std::vector<std::pair<double, double>> X_init;
  std::vector<std::pair<double, double>> X_unsafe;
  // std::function<Matrix(const Matrix&)> f_det;
};

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

Vector to_eigen(const std::vector<double>& vectors) {
  return Vector::NullaryExpr(vectors.size(), [&vectors](const Index i) { return vectors[i]; });
}

#if 1
bool pipeline(const CliArgs& args) {
  random::seed(args.seed);

  std::unique_ptr<RectSet> X_bounds = std::make_unique<RectSet>(args.X_bounds);
  std::unique_ptr<RectSet> X_init = std::make_unique<RectSet>(args.X_init);
  std::unique_ptr<RectSet> X_unsafe = std::make_unique<RectSet>(args.X_unsafe);

  LUCID_DEBUG_FMT("X_bounds: {}", *X_bounds);
  LUCID_DEBUG_FMT("X_init: {}", *X_init);
  LUCID_DEBUG_FMT("X_unsafe: {}", *X_unsafe);

  auto f = [&args](const Matrix& x) -> Matrix {
    std::normal_distribution d{0.0, args.noise_scale};
    // Add noise to the linear function
    auto f_det = [](const Matrix& x) -> Matrix {
      // Example deterministic function: identity
      return x.array() / 2.0;
    };
    const Matrix y{f_det(x)};
    return f_det(x) + Matrix::NullaryExpr(y.rows(), y.cols(), [&d](Index, Index) { return d(random::gen); });
  };

  const Matrix x_samples{X_bounds->sample(args.num_samples)};
  const Matrix xp_samples{f(x_samples)};

  Vector sigma_l{to_eigen(args.sigma_l)};
  KernelRidgeRegressor estimator{std::make_unique<GaussianKernel>(sigma_l, args.sigma_f), args.lambda};
  LinearTruncatedFourierFeatureMap feature_map{args.num_frequencies, sigma_l, args.sigma_f, *X_bounds};

  const Matrix f_xp_samples{feature_map(xp_samples)};

  const int n_per_dim = args.num_oversample < 0
                            ? static_cast<int>(std::ceil((2 * args.num_frequencies + 1) * args.oversample_factor))
                            : args.num_oversample;
  LUCID_DEBUG_FMT("Number of samples per dimension: {}", n_per_dim);

  LUCID_DEBUG_FMT("Estimator pre-fit: {}", estimator);
  estimator.fit(x_samples, f_xp_samples);
  LUCID_INFO_FMT("Estimator post-fit: {}", estimator);

#ifdef COMPUTE_TRAINING_ERROR
  {
    LUCID_DEBUG_FMT("RMSE on f_xp_samples {}", scorer::rmse_score(estimator, x_samples, f_xp_samples));
    LUCID_DEBUG_FMT("Score on f_xp_samples {}", estimator.score(x_samples, f_xp_samples));
    Matrix x_evaluation = X_bounds->sample(x_samples.rows() / 2);
    Matrix f_xp_evaluation = feature_map(args.f_det(x_evaluation));
    LUCID_DEBUG_FMT("RMSE on f_det_evaluated {}", scorer::rmse_score(estimator, x_evaluation, f_xp_evaluation));
    LUCID_DEBUG_FMT("Score on f_det_evaluated {}", estimator.score(x_evaluation, f_xp_evaluation));
  }
#endif

  LUCID_DEBUG_FMT("Feature map: {}", feature_map);

  const Matrix x_lattice = X_bounds->lattice(n_per_dim, true);
  const Matrix u_f_x_lattice = feature_map(x_lattice);
  Matrix u_f_xp_lattice_via_regressor = estimator(x_lattice);
  // We are fixing the zero frequency to the constant value we computed in the feature map
  // If we don't, the regressor has a hard time learning it on the extreme left and right points, because it tends to 0
  u_f_xp_lattice_via_regressor.col(0).array() = feature_map.weights()[0] * args.sigma_f;

  const Matrix x0_lattice = X_init->lattice(n_per_dim, true);
  const Matrix f_x0_lattice = feature_map(x0_lattice);

  const Matrix xu_lattice = X_unsafe->lattice(n_per_dim, true);
  const Matrix f_xu_lattice = feature_map(xu_lattice);

  FourierBarrierCertificate barrier{args.time_horizon, args.gamma};
  barrier.synthesize(*get_optimiser(args.solver, args), u_f_x_lattice, u_f_xp_lattice_via_regressor, f_x0_lattice,
                     f_xu_lattice, feature_map, n_per_dim, args.c_coefficient);
  LUCID_INFO_FMT("End of operation: {}", barrier);
  return true;
}

double vector_norm(const std::vector<double>& v) {
  return Eigen::Map<const Eigen::ArrayXd>{v.data(), static_cast<long int>(v.size())}.matrix().norm();
}
#endif

class Log {
 public:
  static void set_sink(emscripten::val cb) {
    lucid::log::set_logger_sink([cb](std::string msg) { cb(msg); });
  }
  static void trace(const std::string& message) { LUCID_TRACE_FMT("{}", message); }
  static void debug(const std::string& message) { LUCID_DEBUG_FMT("{}", message); }
  static void info(const std::string& message) { LUCID_INFO_FMT("{}", message); }
  static void warn(const std::string& message) { LUCID_WARN_FMT("{}", message); }
  static void error(const std::string& message) { LUCID_ERROR_FMT("{}", message); }
  static void critical(const std::string& message) { LUCID_CRITICAL_FMT("{}", message); }
};

class Random {};

EMSCRIPTEN_BINDINGS(jslucid) {
  register_vector<double>("VectorDouble");
  register_vector<std::pair<double, double>>("VectorPairDouble");
  value_array<std::pair<double, double>>("PairDouble")
      .element(&std::pair<double, double>::first)
      .element(&std::pair<double, double>::second);

  constant("name", std::string{LUCID_PROGRAM_NAME});
#ifdef LUCID_DESCRIPTION
  constant("description", std::string{LUCID_DESCRIPTION});
#else
#error "LUCID_DESCRIPTION is not defined"
#endif
#ifdef LUCID_VERSION_STRING
  constant("version", std::string{LUCID_VERSION_STRING});
#else
#error "LUCID_VERSION_STRING is not defined"
#endif
  constant("MATPLOTLIB_BUILD", lucid::constants::MATPLOTLIB_BUILD);
  constant("GUROBI_BUILD", lucid::constants::GUROBI_BUILD);
  constant("ALGLIB_BUILD", lucid::constants::ALGLIB_BUILD);
  constant("HIGHS_BUILD", lucid::constants::HIGHS_BUILD);
  constant("SOPLEX_BUILD", lucid::constants::SOPLEX_BUILD);
  constant("OMP_BUILD", lucid::constants::OMP_BUILD);
  constant("CUDA_BUILD", lucid::constants::CUDA_BUILD);

  constant("LOG_NONE", LUCID_LOG_OFF_LEVEL);
  constant("LOG_CRITICAL", LUCID_LOG_CRITICAL_LEVEL);
  constant("LOG_ERROR", LUCID_LOG_ERROR_LEVEL);
  constant("LOG_WARN", LUCID_LOG_WARN_LEVEL);
  constant("LOG_INFO", LUCID_LOG_INFO_LEVEL);
  constant("LOG_DEBUG", LUCID_LOG_DEBUG_LEVEL);
  constant("LOG_TRACE", LUCID_LOG_TRACE_LEVEL);

  class_<Log>("log")
      .class_function("set_verbosity", select_overload<void(int)>(&lucid::log::set_verbosity_level))
      .class_function("set_sink", &Log::set_sink)
      .class_function("set_pattern", &lucid::log::set_pattern)
      .class_function("clear", &lucid::log::clear_logger)
      .class_function("trace", &Log::trace)
      .class_function("debug", &Log::debug)
      .class_function("info", &Log::info)
      .class_function("warn", &Log::warn)
      .class_function("error", &Log::error)
      .class_function("critical", &Log::critical);

  class_<Random>("random").class_function("seed", &lucid::random::seed);

  enum_<Solver>("Solver")
      .value("Gurobi", Solver::Gurobi)
      .value("Alglib", Solver::Alglib)
      .value("HiGHS", Solver::HiGHS)
      .value("SOPLEX", Solver::SOPLEX);

  value_object<CliArgs>("CliArgs")
      .field("seed", &CliArgs::seed)
      .field("gamma", &CliArgs::gamma)
      .field("time_horizon", &CliArgs::time_horizon)
      .field("num_samples", &CliArgs::num_samples)
      .field("lambda", &CliArgs::lambda)
      .field("sigma_f", &CliArgs::sigma_f)
      .field("sigma_l", &CliArgs::sigma_l)
      .field("num_frequencies", &CliArgs::num_frequencies)
      .field("c_coefficient", &CliArgs::c_coefficient)
      .field("plot", &CliArgs::plot)
      .field("verify", &CliArgs::verify)
      .field("problem_log_file", &CliArgs::problem_log_file)
      .field("iis_log_file", &CliArgs::iis_log_file)
      .field("oversample_factor", &CliArgs::oversample_factor)
      .field("num_oversample", &CliArgs::num_oversample)
      .field("noise_scale", &CliArgs::noise_scale)
      .field("solver", &CliArgs::solver)
      .field("X_bounds", &CliArgs::X_bounds)
      .field("X_init", &CliArgs::X_init)
      .field("X_unsafe", &CliArgs::X_unsafe);
  //.field("f_det", &CliArgs::f_det);

  function("pipeline", &pipeline);
}
