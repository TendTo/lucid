/**
 * @author Room 6.030
 * @copyright 2025 lucid
 * @licence BSD 3-Clause License
 * @file
 */
#include "LbfgsTuner.h"

#include <LBFGS.h>

#include <iostream>
#include <memory>

#include "lucid/lib/eigen.h"
#include "lucid/model/GradientOptimizable.h"
#include "lucid/util/error.h"
#include "lucid/util/logging.h"

namespace lucid {

namespace {

LBFGSpp::LBFGSParam<double> to_lbfgs(const LbgsParameters& parameters) {
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

}  // namespace

LbfgsTuner::LbfgsTuner(const LbgsParameters& parameters) : parameters_{parameters} {
#ifndef NCHECK
  to_lbfgs(parameters).check_param();
#endif
}

void LbfgsTuner::tune_impl([[maybe_unused]] Estimator& estimator, ConstMatrixRef training_inputs,
                           ConstMatrixRef training_outputs) const {
  LUCID_DEBUG_FMT("LbfgsOptimiser::tune_impl([{}x{}, [{}x{}])", training_inputs.rows(), training_inputs.cols(),
                  training_outputs.rows(), training_outputs.cols());
  LUCID_TRACE_FMT("LbfgsOptimiser::tune_impl({}, {})", training_inputs, training_outputs);

  LUCID_CHECK_ARGUMENT(dynamic_cast<GradientOptimizable*>(&estimator) != nullptr, "estimator",
                       "not an instance of GradientOptimizable");
  GradientOptimizable& gradient_estimator = static_cast<GradientOptimizable&>(estimator);

  // Create solver and function object
  LBFGSpp::LBFGSSolver<double> solver(to_lbfgs(parameters_));

  // Initial guess is the current gradient-optimizable parameters
  Eigen::VectorXd x_out = gradient_estimator.get<Parameter::GRADIENT_OPTIMIZABLE>();
  // x will be overwritten to be the best point found
  double obj;
  const auto f = [&gradient_estimator, &training_inputs, &training_outputs](const Eigen::VectorXd& x,
                                                                            Eigen::VectorXd& grad) {
    gradient_estimator.set(Parameter::GRADIENT_OPTIMIZABLE, static_cast<Vector>(x));
    gradient_estimator.consolidate(training_inputs, training_outputs);
    // TODO(tend): copy elements instead of allocating a new vector
    grad = gradient_estimator.gradient();
    return gradient_estimator.objective_value();
  };
  const int niter = solver.minimize(f, x_out, obj);

  LUCID_DEBUG_FMT("LbfgsOptimiser::tune_impl(): number_iterations = {}, objective_value = {}", niter, obj);
  LUCID_TRACE_FMT("LbfgsOptimiser::tune_impl(): solution = {}", Vector{x_out});

  gradient_estimator.set(Parameter::GRADIENT_OPTIMIZABLE, static_cast<Vector>(x_out));
}

}  // namespace lucid
