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

struct PlotPreviewData {
  emscripten::val x_lattice;
  emscripten::val xp_lattice;
};

struct PlotSolutionData {
  emscripten::val x_lattice;
  emscripten::val B_lattice;
  emscripten::val Bp_lattice;
  emscripten::val Bp_lattice_est;
};

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

bool is_empty_array(const emscripten::val& array) { return !array.isArray() || array["length"].as<std::size_t>() == 0; }

Matrix to_eigen_matrix(const emscripten::val& js_matrix) {
  const int rows = js_matrix["length"].as<int>();
  const int cols = js_matrix[0]["length"].as<int>();
  Matrix matrix{rows, cols};
  for (int row = 0; row < rows; ++row) {
    for (int col = 0; col < cols; ++col) {
      matrix(row, col) = js_matrix[row][col].as<double>();
    }
  }
  return matrix;
}

template <class Derived>
emscripten::val to_array_matrix(const Eigen::MatrixBase<Derived>& matrix) {
  emscripten::val js_matrix = emscripten::val::array();
  for (Index row = 0; row < matrix.rows(); ++row) {
    emscripten::val js_row = emscripten::val::array();
    for (Index col = 0; col < matrix.cols(); ++col) {
      js_row.call<void>("push", matrix(row, col));
    }
    js_matrix.call<void>("push", js_row);
  }
  return js_matrix;
}

template <class Derived>
emscripten::val to_array_vector(const Eigen::MatrixBase<Derived>& vector) {
  emscripten::val js_vector = emscripten::val::array();
  for (Index i = 0; i < vector.size(); ++i) {
    js_vector.call<void>("push", vector(i));
  }
  return js_vector;
}

Vector to_eigen_vector(const emscripten::val& vector) {
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
  explicit JsSphereSet(const emscripten::val& center, const double radius) : center_(center), radius_{radius} {}
  std::unique_ptr<Set> to_set() const override {
    return std::make_unique<SphereSet>(to_eigen_vector(center_), radius_);
  }

 private:
  emscripten::val center_;
  double radius_;
};

class JsEllipseSet final : public JsSet {
 public:
  JsEllipseSet(const emscripten::val& center, const emscripten::val& radii) : center_(center), radii_(radii) {}
  std::unique_ptr<Set> to_set() const override {
    return std::make_unique<EllipseSet>(to_eigen_vector(center_), to_eigen_vector(radii_));
  }

 private:
  emscripten::val center_;
  emscripten::val radii_;
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

struct Configuration {
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
  emscripten::val x_samples;
  emscripten::val xp_samples;
  emscripten::val f_xp_samples;
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

std::unique_ptr<Optimiser> get_optimiser(const Solver solver, const Configuration& conf) {
  switch (solver) {
#ifdef LUCID_GUROBI_BUILD
    case Solver::Gurobi:
      return std::make_unique<GurobiOptimiser>(conf.problem_log_file, conf.iis_log_file);
#endif
#ifdef LUCID_ALGLIB_BUILD
    case Solver::Alglib:
      return std::make_unique<AlglibOptimiser>(conf.problem_log_file, conf.iis_log_file);
#endif
#ifdef LUCID_HIGHS_BUILD
    case Solver::HiGHS:
      return std::make_unique<HighsOptimiser>(conf.problem_log_file, conf.iis_log_file);
#endif
#ifdef LUCID_SOPLEX_BUILD
    case Solver::SOPLEX:
      return std::make_unique<SoplexOptimiser>(conf.problem_log_file, conf.iis_log_file);
#endif
    default:
      throw std::invalid_argument("Solver not supported or not built");
  }
}

std::function<Matrix(const Matrix&)> get_f_det(const Configuration& conf) {
  return [&conf](const Matrix& x) -> Matrix {
    Matrix y{x};
    emscripten::val view = emscripten::val(emscripten::typed_memory_view(y.size(), y.data()));
    conf.system_dynamics(view, y.rows(), y.cols());
    return y;
  };
}

std::function<Matrix(const Matrix&)> get_f(const std::function<Matrix(const Matrix&)>& f_det,
                                           const Configuration& conf) {
  return [&f_det, &conf](const Matrix& x) -> Matrix {
    std::normal_distribution d{0.0, conf.noise_scale};
    Matrix y{f_det(x)};
    return y + Matrix::NullaryExpr(y.rows(), y.cols(), [&d](Index, Index) { return d(random::gen); });
  };
}

PlotPreviewData plot_preview(const Configuration& conf) {
  const std::unique_ptr<Set> X_bounds_ptr{conf.X_bounds->to_set()};
  const RectSet* const X_bounds = dynamic_cast<const RectSet*>(X_bounds_ptr.get());
  const auto f_det = get_f_det(conf);

  const Matrix lattice = X_bounds->lattice(conf.num_samples, true);
  const Matrix xp_lattice = !conf.system_dynamics.isUndefined() ? f_det(lattice) : Matrix{};

  return {
      .x_lattice = to_array_matrix(lattice),
      .xp_lattice = to_array_matrix(xp_lattice),
  };
}

PlotSolutionData plot_solution(const Set& X_bounds, const std::function<Matrix(const Matrix&)>& f_det,
                               const Estimator& estimator, const TruncatedFourierFeatureMap& feature_map,
                               const FourierBarrierCertificate& barrier, const Configuration& conf) {
  const Matrix lattice = X_bounds.lattice(25, true);
  const Matrix f_lattice = feature_map(lattice);
  const Matrix fp_lattice = !conf.system_dynamics.isUndefined() ? feature_map(f_det(lattice)) : Matrix{};
  const Matrix fp_lattice_est = estimator(lattice);

  return {
      .x_lattice = to_array_matrix(lattice),
      .B_lattice = to_array_vector(f_lattice * barrier.coefficients().transpose()),
      .Bp_lattice = fp_lattice.rows() > 0 ? to_array_vector(fp_lattice * barrier.coefficients().transpose())
                                          : emscripten::val::array(),
      .Bp_lattice_est = to_array_vector(fp_lattice_est * barrier.coefficients().transpose()),
  };
}

std::pair<BarrierCertificateResult, PlotSolutionData> pipeline(const Configuration& conf) {
  // log::set_verbosity_level(conf.verbose);
  random::seed(conf.seed);

  const std::unique_ptr<Set> X_bounds_ptr{conf.X_bounds->to_set()};
  const RectSet* const X_bounds = dynamic_cast<const RectSet*>(X_bounds_ptr.get());
  const std::unique_ptr<Set> X_init{conf.X_init->to_set()};
  const std::unique_ptr<Set> X_unsafe{conf.X_unsafe->to_set()};

  LUCID_DEBUG_FMT("X_bounds: {}", *X_bounds);
  LUCID_DEBUG_FMT("X_init: {}", *X_init);
  LUCID_DEBUG_FMT("X_unsafe: {}", *X_unsafe);

  const Matrix x_samples =
      is_empty_array(conf.x_samples) ? X_bounds->sample(conf.num_samples) : to_eigen_matrix(conf.x_samples);
  LUCID_DEBUG_FMT("x_samples: {}", LUCID_FORMAT_MATRIX(x_samples));

  const auto f_det = get_f_det(conf);
  const auto f = get_f(f_det, conf);

  const Matrix xp_samples = is_empty_array(conf.xp_samples) ? f(x_samples) : to_eigen_matrix(conf.xp_samples);
  LUCID_DEBUG_FMT("xp_samples: {}", LUCID_FORMAT_MATRIX(xp_samples));

  Vector sigma_l{to_eigen_vector(conf.sigma_l)};
  Vector feature_sigma_l{to_eigen_vector(conf.feature_sigma_l)};
  KernelRidgeRegressor estimator{std::make_unique<GaussianKernel>(sigma_l, conf.sigma_f), conf.lambda};
  LinearTruncatedFourierFeatureMap feature_map{conf.num_frequencies, feature_sigma_l, conf.sigma_f, *X_bounds};

  const Matrix f_xp_samples =
      is_empty_array(conf.f_xp_samples) ? feature_map(xp_samples) : to_eigen_matrix(conf.f_xp_samples);
  LUCID_DEBUG_FMT("f_xp_samples: {}", LUCID_FORMAT_MATRIX(f_xp_samples));

  const int lattice_resolution =
      conf.lattice_resolution < 0 ? static_cast<int>(std::ceil((2 * conf.num_frequencies + 1) * conf.oversample_factor))
                                  : conf.lattice_resolution;
  LUCID_DEBUG_FMT("Number of samples per dimension: {}", lattice_resolution);

  estimator.fit(x_samples, feature_map(f(x_samples)));
  LUCID_INFO_FMT("Estimator: {}", estimator);

  LUCID_DEBUG_FMT("Feature map: {}", feature_map);

  FourierBarrierCertificate barrier{conf.time_horizon, conf.gamma};
  barrier.synthesize(*get_optimiser(conf.solver, conf), lattice_resolution,  //
                     estimator, feature_map, *X_bounds, *X_init, *X_unsafe,
                     FourierBarrierCertificateParameters{
                         .set_scaling = conf.set_scaling,
                         .C_coeff = conf.C_coeff,
                         .epsilon = conf.epsilon,
                         .b_norm = conf.b_norm,
                         .kappa = conf.b_kappa,
                     });
  LUCID_INFO_FMT("Result: {}", barrier);
  return {{
              .success = barrier.is_synthesized(),
              .eta = barrier.eta(),
              .gamma = barrier.gamma(),
              .T = barrier.T(),
              .c = barrier.c(),
              .safety = barrier.safety(),
              .coefficients = std::vector<double>{barrier.coefficients().data(),
                                                  barrier.coefficients().data() + barrier.coefficients().size()},
              .b_norm = barrier.norm(),
          },
          plot_solution(*X_bounds, f_det, estimator, feature_map, barrier, conf)};
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
  value_array<std::pair<BarrierCertificateResult, PlotSolutionData>>("PairResult")
      .element(&std::pair<BarrierCertificateResult, PlotSolutionData>::first)
      .element(&std::pair<BarrierCertificateResult, PlotSolutionData>::second);

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

  value_object<PlotPreviewData>("PlotPreviewData")
      .field("x_lattice", &PlotPreviewData::x_lattice)
      .field("xp_lattice", &PlotPreviewData::xp_lattice);

  value_object<PlotSolutionData>("PlotSolutionData")
      .field("x_lattice", &PlotSolutionData::x_lattice)
      .field("B_lattice", &PlotSolutionData::B_lattice)
      .field("Bp_lattice", &PlotSolutionData::Bp_lattice)
      .field("Bp_lattice_est", &PlotSolutionData::Bp_lattice_est);

  value_object<BarrierCertificateResult>("BarrierCertificateResult")
      .field("success", &BarrierCertificateResult::success)
      .field("eta", &BarrierCertificateResult::eta)
      .field("gamma", &BarrierCertificateResult::gamma)
      .field("T", &BarrierCertificateResult::T)
      .field("c", &BarrierCertificateResult::c)
      .field("safety", &BarrierCertificateResult::safety)
      .field("coefficients", &BarrierCertificateResult::coefficients)
      .field("b_norm", &BarrierCertificateResult::b_norm);

  value_object<Configuration>("Configuration")
      .field("verbose", &Configuration::verbose)
      .field("seed", &Configuration::seed)
      .field("plot", &Configuration::plot)
      .field("verify", &Configuration::verify)
      .field("problem_log_file", &Configuration::problem_log_file)
      .field("iis_log_file", &Configuration::iis_log_file)
      .field("system_dynamics", &Configuration::system_dynamics)
      .field("X_bounds", &Configuration::X_bounds)
      .field("X_init", &Configuration::X_init)
      .field("X_unsafe", &Configuration::X_unsafe)
      .field("x_samples", &Configuration::x_samples)
      .field("xp_samples", &Configuration::xp_samples)
      .field("f_xp_samples", &Configuration::f_xp_samples)
      .field("num_samples", &Configuration::num_samples)
      .field("noise_scale", &Configuration::noise_scale)
      .field("lambda", &Configuration::lambda)
      .field("sigma_f", &Configuration::sigma_f)
      .field("sigma_l", &Configuration::sigma_l)
      .field("feature_sigma_l", &Configuration::feature_sigma_l)
      .field("num_frequencies", &Configuration::num_frequencies)
      .field("oversample_factor", &Configuration::oversample_factor)
      .field("lattice_resolution", &Configuration::lattice_resolution)
      .field("gamma", &Configuration::gamma)
      .field("C_coeff", &Configuration::C_coeff)
      .field("time_horizon", &Configuration::time_horizon)
      .field("epsilon", &Configuration::epsilon)
      .field("b_norm", &Configuration::b_norm)
      .field("b_kappa", &Configuration::b_kappa)
      .field("set_scaling", &Configuration::set_scaling)
      .field("solver", &Configuration::solver);

  function("plot_preview", &plot_preview);
  function("pipeline", &pipeline);
}
