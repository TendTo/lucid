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

enum class Solver { Gurobi, Alglib, HiGHS, SOPLEX };

struct BarrierCertificateResult {
  bool success{false};
  double eta{0.0};
  double gamma{0.0};
  int T{0};
  double c{0.0};
  double safety{0.0};
  std::vector<double> coefficients{};
  double b_norm{0.0};
};

Vector to_eigen(const std::vector<double>& vectors) {
  return Vector::NullaryExpr(vectors.size(), [&vectors](const Index i) { return vectors[i]; });
}

Vector to_eigen(const emscripten::val& vector) {
  return Vector::NullaryExpr(vector["length"].as<std::size_t>(),
                             [&vector](const Index i) { return vector[i].as<double>(); });
}

class JsSet {
 public:
  virtual ~JsSet() = default;
  virtual std::unique_ptr<Set> to_set() const = 0;
};

class JsRectSet final : public JsSet {
 public:
  explicit JsRectSet(const emscripten::val& bounds) : bounds_(bounds["length"].as<std::size_t>()) {
    for (std::size_t i = 0; i < bounds_.size(); ++i) {
      bounds_[i] = std::make_pair(bounds[i][0].as<double>(), bounds[i][1].as<double>());
    }
  }
  std::unique_ptr<Set> to_set() const override { return std::make_unique<RectSet>(bounds_); }

 private:
  std::vector<std::pair<double, double>> bounds_;
};

class JsSphereSet final : public JsSet {
 public:
  explicit JsSphereSet(const emscripten::val& center, const double radius)
      : center_(center["length"].as<std::size_t>()), radius_{radius} {
    for (std::size_t i = 0; i < center_.size(); ++i) {
      center_[i] = center[i].as<double>();
    }
  }
  std::unique_ptr<Set> to_set() const override { return std::make_unique<SphereSet>(to_eigen(center_), radius_); }

 private:
  std::vector<double> center_;
  double radius_;
};

class JsEllipseSet final : public JsSet {
 public:
  JsEllipseSet(const emscripten::val& center, const emscripten::val& semi_axes)
      : center_(center["length"].as<std::size_t>()), semi_axes_(semi_axes["length"].as<std::size_t>()) {
    for (std::size_t i = 0; i < center_.size(); ++i) {
      center_[i] = center[i].as<double>();
    }
    for (std::size_t i = 0; i < semi_axes_.size(); ++i) {
      semi_axes_[i] = semi_axes[i].as<double>();
    }
  }
  std::unique_ptr<Set> to_set() const override {
    return std::make_unique<EllipseSet>(to_eigen(center_), to_eigen(semi_axes_));
  }

 private:
  std::vector<double> center_;
  std::vector<double> semi_axes_;
};

class JsMultiSet final : public JsSet {
 public:
  explicit JsMultiSet(const emscripten::val& sets) : sets_{} {
    const std::size_t num_sets = sets["length"].as<std::size_t>();
    sets_.reserve(num_sets);
    for (std::size_t i = 0; i < num_sets; ++i) {
      sets_.emplace_back(sets[i].as<std::shared_ptr<JsSet>>());
    }
  }
  void add_set(const std::shared_ptr<JsSet>& set) { sets_.emplace_back(set); }
  std::unique_ptr<Set> to_set() const override {
    std::vector<std::unique_ptr<Set>> unique_sets;
    unique_sets.reserve(sets_.size());
    for (const auto& set : sets_) {
      unique_sets.emplace_back(set->to_set());
    }
    return std::make_unique<MultiSet>(std::move(unique_sets));
  }

 private:
  std::vector<std::shared_ptr<JsSet>> sets_;
};

class JsMatrix {
 public:
  static std::shared_ptr<JsMatrix> empty() { return std::make_shared<JsMatrix>(0, 0); }
  explicit JsMatrix(const emscripten::val& matrix)
      : matrix_{matrix["length"].as<int>(), matrix[0]["length"].as<int>()} {
    set_matrix(matrix);
  }
  JsMatrix(const int rows, const int cols) : matrix_{rows, cols} {}

  void set_coeff(const int row, const int col, const double value) { matrix_(row, col) = value; }
  void set_row(const int row, const emscripten::val& values) {
    for (int col = 0; col < matrix_.cols(); ++col) {
      matrix_(row, col) = values[col].as<double>();
    }
  }
  void set_col(const int col, const emscripten::val& values) {
    for (int row = 0; row < matrix_.rows(); ++row) {
      matrix_(row, col) = values[row].as<double>();
    }
  }
  void set_matrix(const emscripten::val& matrix) {
    for (int row = 0; row < matrix_.rows(); ++row) {
      for (int col = 0; col < matrix_.cols(); ++col) {
        matrix_(row, col) = matrix[row][col].as<double>();
      }
    }
  }
  double get_coeff(const int row, const int col) const { return matrix_(row, col); }

  const Matrix& matrix() const { return matrix_; }

 private:
  Matrix matrix_;
};

struct CliArgs {
  int verbose{LUCID_LOG_INFO_LEVEL};
  int seed{-1};
  bool plot{false};
  bool verify{false};
  std::string problem_log_file{""};
  std::string iis_log_file{""};
  emscripten::val system_dynamics;
  std::shared_ptr<JsSet> X_bounds;
  std::shared_ptr<JsSet> X_init;
  std::shared_ptr<JsSet> X_unsafe;
  std::shared_ptr<JsMatrix> x_samples{};
  std::shared_ptr<JsMatrix> xp_samples{};
  std::shared_ptr<JsMatrix> f_xp_samples{};
  int num_samples{1000};
  double noise_scale{0.01};
  double lambda{1e-6};
  double sigma_f{1.0};
  emscripten::val sigma_l;
  emscripten::val feature_sigma_l;
  int num_frequencies{4};
  double oversample_factor{2.0};
  int lattice_resolution{-1};
  double gamma{1.0};
  double C_coeff{1.0};
  int time_horizon{5};
  double epsilon{0.0};
  double b_norm{0.0};
  double b_kappa{1.0};
  double set_scaling{0.1};
  Solver solver{Solver::Gurobi};
};

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

BarrierCertificateResult pipeline(const CliArgs& args) {
  // log::set_verbosity_level(args.verbose);
  random::seed(args.seed);

  const std::unique_ptr<Set> X_bounds_ptr{args.X_bounds->to_set()};
  const RectSet* const X_bounds = dynamic_cast<const RectSet*>(X_bounds_ptr.get());
  const std::unique_ptr<Set> X_init{args.X_init->to_set()};
  const std::unique_ptr<Set> X_unsafe{args.X_unsafe->to_set()};

  LUCID_DEBUG_FMT("X_bounds: {}", *X_bounds);
  LUCID_DEBUG_FMT("X_init: {}", *X_init);
  LUCID_DEBUG_FMT("X_unsafe: {}", *X_unsafe);

  const Matrix x_samples = args.x_samples ? args.x_samples->matrix() : X_bounds->sample(args.num_samples);
  LUCID_DEBUG_FMT("x_samples: {}", LUCID_FORMAT_MATRIX(x_samples));

  auto f_det = [&args](const Matrix& x) -> Matrix {
    Matrix y{x};
    emscripten::val view = emscripten::val(emscripten::typed_memory_view(y.size(), y.data()));
    args.system_dynamics(view, y.rows(), y.cols());
    return y;
  };
  std::normal_distribution d{0.0, args.noise_scale};
  auto f = [&f_det, &d](const Matrix& x) -> Matrix {
    Matrix y{f_det(x)};
    return y + Matrix::NullaryExpr(y.rows(), y.cols(), [&d](Index, Index) { return d(random::gen); });
  };

  const Matrix xp_samples = args.xp_samples ? args.xp_samples->matrix() : f(x_samples);
  LUCID_DEBUG_FMT("xp_samples: {}", LUCID_FORMAT_MATRIX(xp_samples));

  Vector sigma_l{to_eigen(args.sigma_l)};
  Vector feature_sigma_l{to_eigen(args.feature_sigma_l)};
  KernelRidgeRegressor estimator{std::make_unique<GaussianKernel>(sigma_l, args.sigma_f), args.lambda};
  LinearTruncatedFourierFeatureMap feature_map{args.num_frequencies, feature_sigma_l, args.sigma_f, *X_bounds};

  const Matrix f_xp_samples = args.f_xp_samples ? args.f_xp_samples->matrix() : feature_map(xp_samples);
  LUCID_DEBUG_FMT("f_xp_samples: {}", LUCID_FORMAT_MATRIX(f_xp_samples));

  const int lattice_resolution =
      args.lattice_resolution < 0 ? static_cast<int>(std::ceil((2 * args.num_frequencies + 1) * args.oversample_factor))
                                  : args.lattice_resolution;
  LUCID_DEBUG_FMT("Number of samples per dimension: {}", lattice_resolution);

  estimator.fit(x_samples, feature_map(f(x_samples)));
  LUCID_INFO_FMT("Estimator: {}", estimator);

  LUCID_DEBUG_FMT("Feature map: {}", feature_map);

  FourierBarrierCertificate barrier{args.time_horizon, args.gamma};
  barrier.synthesize(*get_optimiser(args.solver, args), lattice_resolution,  //
                     estimator,
                     //  ModelEstimator{[&f_det, &feature_map](const Matrix& x) { return feature_map(f_det(x)); }},
                     feature_map, *X_bounds, *X_init, *X_unsafe,
                     FourierBarrierCertificateParameters{
                         .set_scaling = args.set_scaling,
                         .C_coeff = args.C_coeff,
                         .epsilon = args.epsilon,
                         .b_norm = args.b_norm,
                         .kappa = args.b_kappa,
                     });
  LUCID_INFO_FMT("End of operation: {}", barrier);
  return {
      .success = barrier.is_synthesized(),
      .eta = barrier.eta(),
      .gamma = barrier.gamma(),
      .T = barrier.T(),
      .c = barrier.c(),
      .safety = barrier.safety(),
      .coefficients = std::vector<double>{barrier.coefficients().data(),
                                          barrier.coefficients().data() + barrier.coefficients().size()},
      .b_norm = barrier.norm(),
  };
}

double vector_norm(const std::vector<double>& v) {
  return Eigen::Map<const Eigen::ArrayXd>{v.data(), static_cast<long int>(v.size())}.matrix().norm();
}

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
  constant("PSOCPP_BUILD", lucid::constants::PSOCPP_BUILD);
  constant("OMP_BUILD", lucid::constants::OMP_BUILD);
  constant("CUDA_BUILD", lucid::constants::CUDA_BUILD);
  constant("DEBUG_BUILD", lucid::constants::DEBUG_BUILD);
  constant("RELEASE_BUILD", lucid::constants::RELEASE_BUILD);
  constant("ASSERT_CHECKS_ENABLED", lucid::constants::ASSERT_CHECKS_ENABLED);
  constant("RUNTIME_CHECKS_ENABLED", lucid::constants::RUNTIME_CHECKS_ENABLED);
  constant("LOG_ENABLED", lucid::constants::LOG_ENABLED);

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

  class_<JsMatrix>("Matrix")
      .smart_ptr_constructor<std::shared_ptr<JsMatrix>>("Matrix", &std::make_shared<JsMatrix, const emscripten::val&>)
      .class_function("empty", &JsMatrix::empty)
      .function("set_coeff", &JsMatrix::set_coeff)
      .function("set_row", &JsMatrix::set_row)
      .function("set_col", &JsMatrix::set_col)
      .function("set_matrix", &JsMatrix::set_matrix)
      .function("get_coeff", &JsMatrix::get_coeff);

  class_<JsSet>("Set").smart_ptr<std::shared_ptr<JsSet>>("Set");
  class_<JsRectSet, base<JsSet>>("RectSet").smart_ptr_constructor<std::shared_ptr<JsRectSet>, const emscripten::val&>(
      "RectSet", &std::make_shared<JsRectSet>);
  class_<JsSphereSet, base<JsSet>>("SphereSet")
      .smart_ptr_constructor<std::shared_ptr<JsSphereSet>, const emscripten::val&, const double&>(
          "SphereSet", &std::make_shared<JsSphereSet>);
  class_<JsEllipseSet, base<JsSet>>("EllipseSet")
      .smart_ptr_constructor<std::shared_ptr<JsEllipseSet>, const emscripten::val&, const emscripten::val&>(
          "EllipseSet", &std::make_shared<JsEllipseSet>);
  class_<JsMultiSet, base<JsSet>>("MultiSet")
      .smart_ptr_constructor<std::shared_ptr<JsMultiSet>, const emscripten::val&>("MultiSet",
                                                                                  &std::make_shared<JsMultiSet>);

  enum_<Solver>("Solver")
      .value("Gurobi", Solver::Gurobi)
      .value("Alglib", Solver::Alglib)
      .value("HiGHS", Solver::HiGHS)
      .value("SOPLEX", Solver::SOPLEX);

  value_object<BarrierCertificateResult>("BarrierCertificateResult")
      .field("success", &BarrierCertificateResult::success)
      .field("eta", &BarrierCertificateResult::eta)
      .field("gamma", &BarrierCertificateResult::gamma)
      .field("T", &BarrierCertificateResult::T)
      .field("c", &BarrierCertificateResult::c)
      .field("safety", &BarrierCertificateResult::safety)
      .field("coefficients", &BarrierCertificateResult::coefficients)
      .field("b_norm", &BarrierCertificateResult::b_norm);

  value_object<CliArgs>("CliArgs")
      .field("verbose", &CliArgs::verbose)
      .field("seed", &CliArgs::seed)
      .field("plot", &CliArgs::plot)
      .field("verify", &CliArgs::verify)
      .field("problem_log_file", &CliArgs::problem_log_file)
      .field("iis_log_file", &CliArgs::iis_log_file)
      .field("system_dynamics", &CliArgs::system_dynamics)
      .field("X_bounds", &CliArgs::X_bounds)
      .field("X_init", &CliArgs::X_init)
      .field("X_unsafe", &CliArgs::X_unsafe)
      .field("x_samples", &CliArgs::x_samples)
      .field("xp_samples", &CliArgs::xp_samples)
      .field("f_xp_samples", &CliArgs::f_xp_samples)
      .field("num_samples", &CliArgs::num_samples)
      .field("noise_scale", &CliArgs::noise_scale)
      .field("lambda", &CliArgs::lambda)
      .field("sigma_f", &CliArgs::sigma_f)
      .field("sigma_l", &CliArgs::sigma_l)
      .field("feature_sigma_l", &CliArgs::feature_sigma_l)
      .field("num_frequencies", &CliArgs::num_frequencies)
      .field("oversample_factor", &CliArgs::oversample_factor)
      .field("lattice_resolution", &CliArgs::lattice_resolution)
      .field("gamma", &CliArgs::gamma)
      .field("C_coeff", &CliArgs::C_coeff)
      .field("time_horizon", &CliArgs::time_horizon)
      .field("epsilon", &CliArgs::epsilon)
      .field("b_norm", &CliArgs::b_norm)
      .field("b_kappa", &CliArgs::b_kappa)
      .field("set_scaling", &CliArgs::set_scaling)
      .field("solver", &CliArgs::solver);

  function("pipeline", &pipeline);
}
