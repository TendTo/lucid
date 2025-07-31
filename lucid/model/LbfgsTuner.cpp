/**
 * @author Room 6.030
 * @copyright 2025 lucid
 * @licence BSD 3-Clause License
 * @file
 */
#include "LbfgsTuner.h"

#include <LBFGS.h>
#include <LBFGSB.h>

#include <memory>
#include <ostream>
#include <vector>

#include "lucid/lib/eigen.h"
#include "lucid/model/GradientOptimizable.h"
#include "lucid/util/error.h"
#include "lucid/util/logging.h"

namespace lucid {

namespace {

LBFGSpp::LBFGSParam<double> to_lbfgs(const LbfgsParameters& parameters) {
  LBFGSpp::LBFGSParam<double> external_parameters{};
  external_parameters.m = parameters.m;
  external_parameters.epsilon = parameters.epsilon;
  external_parameters.epsilon_rel = parameters.epsilon_rel;
  external_parameters.past = parameters.past;
  external_parameters.delta = parameters.delta;
  external_parameters.max_iterations = parameters.max_iterations;
  external_parameters.linesearch = parameters.linesearch;
  external_parameters.max_linesearch = parameters.max_linesearch;
  external_parameters.min_step = parameters.min_step;
  external_parameters.max_step = parameters.max_step;
  external_parameters.ftol = parameters.ftol;
  external_parameters.wolfe = parameters.wolfe;
  return external_parameters;
}

LBFGSpp::LBFGSBParam<double> to_lbfgsb(const LbfgsParameters& parameters) {
  LBFGSpp::LBFGSBParam<double> external_parameters{};
  external_parameters.m = parameters.m;
  external_parameters.epsilon = parameters.epsilon;
  external_parameters.epsilon_rel = parameters.epsilon_rel;
  external_parameters.past = parameters.past;
  external_parameters.delta = parameters.delta;
  external_parameters.max_iterations = parameters.max_iterations;
  external_parameters.max_submin = parameters.max_submin;
  external_parameters.max_linesearch = parameters.max_linesearch;
  external_parameters.min_step = parameters.min_step;
  external_parameters.max_step = parameters.max_step;
  external_parameters.ftol = parameters.ftol;
  external_parameters.wolfe = parameters.wolfe;
  return external_parameters;
}

template <int I, template <class, class...> class T>
Eigen::VectorXd bounds_to_vector(const T<std::pair<Scalar, Scalar>>& bounds) {
  Eigen::VectorXd v(bounds.size());
  Index i = 0;
  for (const auto& [first, second] : bounds) {
    if constexpr (I == 0) {
      v(i++) = first;
    } else {
      v(i++) = second;
    }
  }
  return v;
}

}  // namespace

LbfgsTuner::LbfgsTuner(const LbfgsParameters& parameters) : LbfgsTuner{{}, {}, parameters} {}
LbfgsTuner::LbfgsTuner(const std::vector<std::pair<Scalar, Scalar>>& bounds, const LbfgsParameters& parameters)
    : LbfgsTuner{bounds_to_vector<0, std::vector>(bounds), bounds_to_vector<1, std::vector>(bounds), parameters} {}
LbfgsTuner::LbfgsTuner(const Eigen::VectorXd& lb, const Eigen::VectorXd& ub, const LbfgsParameters& parameters)
    : lb_{lb}, ub_{ub}, parameters_{parameters} {
  LUCID_CHECK_ARGUMENT_EQ(lb_.size(), ub_.size());
#ifndef NCHECK
  if (lb_.size() == 0) {
    to_lbfgs(parameters).check_param();
  } else {
    to_lbfgsb(parameters).check_param();
  }
#endif
}

bool LbfgsTuner::is_bounded() const {
  LUCID_ASSERT(lb_.size() == ub_.size(), "lower and upper bounds must have the same size");
  return lb_.size() > 0;
}

void LbfgsTuner::tune_impl(Estimator& estimator, ConstMatrixRef training_inputs,
                           const OutputComputer& training_outputs) const {
  LUCID_ASSERT(lb_.size() == ub_.size(), "lower and upper bounds must have the same size");
  LUCID_CHECK_ARGUMENT(dynamic_cast<GradientOptimizable*>(&estimator) != nullptr, "estimator",
                       "not an instance of GradientOptimizable");
  GradientOptimizable& gradient_estimator = static_cast<GradientOptimizable&>(estimator);
  ConstMatrixRef training_outputs_ref = training_outputs(gradient_estimator, training_inputs);

  // Function to minimize
  const auto f = [&gradient_estimator, &training_inputs, &training_outputs_ref](const Eigen::VectorXd& x,
                                                                                Eigen::VectorXd& grad) {
    gradient_estimator.set(Parameter::GRADIENT_OPTIMIZABLE, static_cast<Vector>(x));
    gradient_estimator.consolidate(training_inputs, training_outputs_ref, Request::GRADIENT | Request::OBJECTIVE_VALUE);
    // TODO(tend): copy elements instead of allocating a new vector
    grad = -gradient_estimator.gradient();
    return -gradient_estimator.objective_value();
  };
  // Initial guess is the current gradient-optimizable parameters
  Eigen::VectorXd x_out = gradient_estimator.get<Parameter::GRADIENT_OPTIMIZABLE>();
  // Objective value at the optimal point
  double obj;
  // Number of iterations
  [[maybe_unused]] int niter = 0;

  LUCID_CHECK_ARGUMENT_EXPECTED(x_out.size() == lb_.size() || lb_.size() == 0, "x_out.size() != lb.size()",
                                x_out.size(), lb_.size());

  // Create solver and function object
  if (is_bounded()) {
    LBFGSpp::LBFGSBSolver<double> solver(to_lbfgsb(parameters_));
    niter = solver.minimize(f, x_out, obj, lb_, ub_);
  } else {
    LBFGSpp::LBFGSSolver<double> solver(to_lbfgs(parameters_));
    niter = solver.minimize(f, x_out, obj);
  }

  LUCID_DEBUG_FMT("number_iterations = {}, objective_value = {}", niter, obj);
  LUCID_DEBUG_FMT("solution = {}", Vector{x_out});

  gradient_estimator.set(Parameter::GRADIENT_OPTIMIZABLE, static_cast<Vector>(x_out));
}

std::ostream& operator<<(std::ostream& os, const LbfgsParameters& lbgs_parameters) {
  return os << "LbfgsParameters( m( " << lbgs_parameters.m << " ) epsilon( " << lbgs_parameters.epsilon
            << " ) epsilon_rel( " << lbgs_parameters.epsilon_rel << " ) past( " << lbgs_parameters.past << " ) delta( "
            << lbgs_parameters.delta << " ) max_iterations( " << lbgs_parameters.max_iterations << " ) max_submin( "
            << lbgs_parameters.max_submin << " ) max_linesearch( " << lbgs_parameters.max_linesearch << " ) min_step( "
            << lbgs_parameters.min_step << " ) max_step( " << lbgs_parameters.max_step << " ) ftol( "
            << lbgs_parameters.ftol << " ) wolfe( " << lbgs_parameters.wolfe << " ) )";
}

std::ostream& operator<<(std::ostream& os, const LbfgsTuner& tuner) {
  return os << "LbfgsTuner( bounded( " << tuner.is_bounded() << " ) parameters( " << tuner.parameters() << " )";
}

}  // namespace lucid
