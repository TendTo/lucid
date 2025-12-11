/**
 * @author Ernesto Casablanca
 * @author Oliver Schön
 * @copyright 2025 lucid
 * @licence BSD 3-Clause License
 * @file
 */
#include "lucid/util/Configuration.h"

#include <yaml-cpp/yaml.h>

#include <memory>
#include <ostream>
#include <string>
#include <utility>
#include <vector>

// The exprtk library is used for parsing mathematical expressions
// It should go after the c++ headers
#include <exprtk/exprtk.hpp>

#include "lucid/model/EllipseSet.h"
#include "lucid/model/MultiSet.h"
#include "lucid/model/RectSet.h"
#include "lucid/model/SphereSet.h"
#include "lucid/util/error.h"
#include "lucid/util/logging.h"

namespace lucid {

const Matrix emptyMatrix{Matrix::Zero(0, 0)};

namespace {

Vector to_vector(const YAML::Node &node) {
  if (node.IsScalar()) return Vector::Constant(1, node.as<double>());
  const std::vector<double> vector = node.as<std::vector<double>>();
  return Eigen::Map<const Vector>{vector.data(), static_cast<Index>(vector.size())};
}

Matrix to_matrix(const YAML::Node &node) {
  const std::vector<std::vector<double>> vecvec = node.as<std::vector<std::vector<double>>>();
  if (vecvec.empty()) return Matrix{0, 0};
  const Index rows = static_cast<Index>(vecvec.size());
  const Index cols = static_cast<Index>(vecvec[0].size());
  Matrix mat{rows, cols};
  for (Index r = 0; r < rows; r++) {
    LUCID_CHECK_ARGUMENT_CMP(static_cast<Index>(vecvec[r].size()), ==, cols);
    for (Index c = 0; c < cols; c++) {
      mat(r, c) = vecvec[r][c];
    }
  }
  return mat;
}

std::unique_ptr<Set> parse_set(const YAML::Node &node) {
  if (node["RectSet"]) {
    const auto rect_node = node["RectSet"];
    if (!rect_node["lower_bounds"] || !rect_node["upper_bounds"]) {
      const auto bounds = rect_node.as<std::vector<std::pair<double, double>>>();
      return std::make_unique<RectSet>(bounds);
    }
    const Matrix lower_bounds = to_vector(rect_node["lower_bounds"]);
    const Matrix upper_bounds = to_vector(rect_node["upper_bounds"]);
    return std::make_unique<RectSet>(lower_bounds, upper_bounds);
  }
  if (node["SphereSet"]) {
    const auto sphere_node = node["SphereSet"];
    const Vector center = to_vector(sphere_node["center"]);
    const double radius = sphere_node["radius"].as<double>();
    return std::make_unique<SphereSet>(center, radius);
  }
  if (node["EllipseSet"]) {
    const auto ellipse_node = node["EllipseSet"];
    const Vector center = to_vector(ellipse_node["center"]);
    const Matrix shape_matrix = to_matrix(ellipse_node["shape_matrix"]);
    return std::make_unique<EllipseSet>(center, shape_matrix);
  }
  if (node.IsSequence()) {
    std::vector<std::unique_ptr<Set>> sets;
    for (const auto &rect : node) {
      sets.emplace_back(parse_set(rect));
    }
    return sets.size() == 1 ? std::move(sets.back()) : std::make_unique<MultiSet>(std::move(sets));
  }
  return nullptr;
}

}  // namespace

Configuration Configuration::from_yaml(const std::string &filename) {
  std::ifstream ifs(filename.c_str());
  if (!ifs.is_open()) LUCID_RUNTIME_ERROR_FMT("Could not open configuration file: {}", filename);
  const std::string yaml_str((std::istreambuf_iterator<char>(ifs)), std::istreambuf_iterator<char>());
  return from_yaml_str(yaml_str);
}
Configuration Configuration::from_yaml_str(const std::string &yaml_str) {
  Configuration config;
  YAML::Node yaml = YAML::Load(yaml_str);
  if (yaml["verbose"]) config.verbose = yaml["verbose"].as<int>();
  if (yaml["seed"]) config.seed = yaml["seed"].as<int>();
  if (yaml["plot"]) config.plot = yaml["plot"].as<bool>();
  if (yaml["verify"]) config.verify = yaml["verify"].as<bool>();
  if (yaml["print_stats"]) config.print_stats = yaml["print_stats"].as<bool>();
  if (yaml["problem_log_file"]) config.problem_log_file = yaml["problem_log_file"].as<std::string>();
  if (yaml["iis_log_file"]) config.iis_log_file = yaml["iis_log_file"].as<std::string>();
  if (yaml["system_dynamics"]) {
    const auto system_dynamics_str = yaml["system_dynamics"].as<std::vector<std::string>>();

    auto row_vector{std::make_shared<std::vector<double>>(system_dynamics_str.size(), 0.0)};
    auto symbol_table{std::make_shared<exprtk::symbol_table<double>>()};
    for (std::size_t col = 0; col < system_dynamics_str.size(); col++) {
      symbol_table->add_variable("x" + std::to_string(col + 1), (*row_vector)[col]);
    }
    symbol_table->add_constants();
    std::vector<exprtk::expression<double>> expressions;

    exprtk::parser<double> parser;
    for (const auto &dynamics_str : system_dynamics_str) {
      expressions.emplace_back();
      expressions.back().register_symbol_table(*symbol_table);
      if (!parser.compile(dynamics_str, expressions.back())) {
        LUCID_RUNTIME_ERROR_FMT("Error compiling system dynamics expression: {}\n", parser.error());
      }
    }

    config.system_dynamics = [expressions, symbol_table, row_vector](const Matrix &x) -> Matrix {
      // Output matrix
      Matrix out{x.rows(), static_cast<Index>(expressions.size())};
      // For each row...
      for (Index row = 0; row < x.rows(); row++) {
        // Copy the elements of the row into the symbol table variables
        for (Index col = 0; col < x.cols(); col++) (*row_vector)[col] = x(row, col);
        for (Index col = 0; col < out.cols(); col++) out(row, col) = expressions[col].value();
      }
      return out;
    };
  }
  if (yaml["X_bounds"]) config.X_bounds = parse_set(yaml["X_bounds"]);
  if (yaml["X_init"]) config.X_init = parse_set(yaml["X_init"]);
  if (yaml["X_unsafe"]) config.X_unsafe = parse_set(yaml["X_unsafe"]);
  if (yaml["x_samples"]) config.x_samples = to_matrix(yaml["x_samples"]);
  if (yaml["xp_samples"]) config.xp_samples = to_matrix(yaml["xp_samples"]);
  if (yaml["f_xp_samples"]) config.f_xp_samples = to_matrix(yaml["f_xp_samples"]);
  if (yaml["num_samples"]) config.num_samples = yaml["num_samples"].as<int>();
  if (yaml["noise_scale"]) config.noise_scale = yaml["noise_scale"].as<double>();
  if (yaml["lambda"]) config.lambda = yaml["lambda"].as<double>();
  if (yaml["sigma_f"]) config.sigma_f = yaml["sigma_f"].as<double>();
  if (yaml["sigma_l"]) config.sigma_l = to_vector(yaml["sigma_l"]);
  if (yaml["feature_sigma_l"]) config.feature_sigma_l = to_vector(yaml["feature_sigma_l"]);
  if (yaml["num_frequencies"]) config.num_frequencies = yaml["num_frequencies"].as<int>();
  if (yaml["oversample_factor"]) config.oversample_factor = yaml["oversample_factor"].as<double>();
  if (yaml["lattice_resolution"]) config.lattice_resolution = yaml["lattice_resolution"].as<int>();
  if (yaml["gamma"]) config.gamma = yaml["gamma"].as<double>();
  if (yaml["C_coeff"]) config.C_coeff = yaml["C_coeff"].as<double>();
  if (yaml["time_horizon"]) config.time_horizon = yaml["time_horizon"].as<int>();
  if (yaml["epsilon"]) config.epsilon = yaml["epsilon"].as<double>();
  if (yaml["b_norm"]) config.b_norm = yaml["b_norm"].as<double>();
  if (yaml["b_kappa"]) config.b_kappa = yaml["b_kappa"].as<double>();
  if (yaml["set_scaling"]) config.set_scaling = yaml["set_scaling"].as<double>();
  if (yaml["estimator"]) {
    const auto estimator_str = yaml["estimator"].as<std::string>();
    if (estimator_str == "KernelRidgeRegressor") {
      config.estimator = Estimator::KERNEL_RIDGE_REGRESSOR;
    } else if (estimator_str == "ModelEstimator") {
      config.estimator = Estimator::MODEL_ESTIMATOR;
    } else {
      LUCID_RUNTIME_ERROR_FMT("Unknown estimator type: {}", estimator_str);
    }
  }
  if (yaml["kernel"]) {
    const auto kernel_str = yaml["kernel"].as<std::string>();
    if (kernel_str == "GaussianKernel") {
      config.kernel = Kernel::GAUSSIAN_KERNEL;
    } else {
      LUCID_RUNTIME_ERROR_FMT("Unknown kernel type: {}", kernel_str);
    }
  }
  if (yaml["feature_map"]) {
    const auto feature_map_str = yaml["feature_map"].as<std::string>();
    if (feature_map_str == "LinearTruncatedFourierFeatureMap") {
      config.feature_map = FeatureMap::LINEAR_TRUNCATED_FOURIER_FEATURE_MAP;
    } else if (feature_map_str == "LogTruncatedFourierFeatureMap") {
      config.feature_map = FeatureMap::LOG_TRUNCATED_FOURIER_FEATURE_MAP;
    } else if (feature_map_str == "ConstantTruncatedFourierFeatureMap") {
      config.feature_map = FeatureMap::CONSTANT_TRUNCATED_FOURIER_FEATURE_MAP;
    } else {
      LUCID_RUNTIME_ERROR_FMT("Unknown feature map type: {}", feature_map_str);
    }
  }

  if (yaml["optimiser"]) {
    const auto optimiser_str = yaml["optimiser"].as<std::string>();
    if (optimiser_str == "GurobiOptimiser") {
      config.optimiser = Optimiser::GUROBI;
    } else if (optimiser_str == "AlglibOptimiser") {
      config.optimiser = Optimiser::ALGLIB;
    } else if (optimiser_str == "HighsOptimiser") {
      config.optimiser = Optimiser::HIGHS;
    } else if (optimiser_str == "SoplexOptimiser") {
      config.optimiser = Optimiser::SOPLEX;
    } else {
      LUCID_RUNTIME_ERROR_FMT("Unknown optimiser type: {}", optimiser_str);
    }
  }
  return config;
}

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
             "feature_sigma_l( {} ) "
             "num_frequencies( {} ) "
             "oversample_factor( {} ) "
             "lattice_resolution( {} ) "
             "gamma( {} ) "
             "C_coeff( {} ) "
             "time_horizon( {} ) "
             "epsilon( {} ) "
             "b_norm( {} ) "
             "b_kappa( {} ) "
             "set_scaling( {} ) "
             "estimator( {} ) "
             "kernel( {} ) "
             "feature_map( {} ) "
             "optimiser( {} ) "
             ")",
             config.verbose, config.seed, config.plot, config.verify, config.problem_log_file, config.iis_log_file,
             config.system_dynamics ? "provided" : "not provided",
             config.X_bounds == nullptr ? "-" : fmt::format("{}", *config.X_bounds),
             config.X_init == nullptr ? "-" : fmt::format("{}", *config.X_init),
             config.X_unsafe == nullptr ? "-" : fmt::format("{}", *config.X_unsafe), config.x_samples,
             config.xp_samples, config.f_xp_samples, config.num_samples, config.noise_scale, config.lambda,
             config.sigma_f, config.sigma_l, config.feature_sigma_l, config.num_frequencies, config.oversample_factor,
             config.lattice_resolution, config.gamma, config.C_coeff, config.time_horizon, config.epsilon,
             config.b_norm, config.b_kappa, config.set_scaling, config.estimator, config.kernel, config.feature_map,
             config.optimiser);
}
std::ostream &operator<<(std::ostream &os, const Configuration::Optimiser &optimiser) {
  switch (optimiser) {
    case Configuration::Optimiser::GUROBI:
      return os << "GUROBI";
    case Configuration::Optimiser::ALGLIB:
      return os << "ALGLIB";
    case Configuration::Optimiser::HIGHS:
      return os << "HIGHS";
    case Configuration::Optimiser::SOPLEX:
      return os << "SOPLEX";
    default:
      LUCID_UNREACHABLE();
  }
}
std::ostream &operator<<(std::ostream &os, const Configuration::Estimator &estimator) {
  switch (estimator) {
    case Configuration::Estimator::KERNEL_RIDGE_REGRESSOR:
      return os << "KERNEL_RIDGE_REGRESSOR";
    case Configuration::Estimator::MODEL_ESTIMATOR:
      return os << "MODEL_ESTIMATOR";
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
