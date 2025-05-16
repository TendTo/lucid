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
#include "lucid/util/error.h"
#include "lucid/util/logging.h"

namespace lucid {
namespace {
class Rosenbrock {
 private:
  int n;

 public:
  explicit Rosenbrock(const int n_) : n(n_) {}
  double operator()(const Vector& x, Vector& grad) {
    double fx = 0.0;
    for (int i = 0; i < n; i += 2) {
      double t1 = 1.0 - x[i];
      double t2 = 10 * (x[i + 1] - x[i] * x[i]);
      grad[i + 1] = 20 * t2;
      grad[i] = -2.0 * (x[i] * grad[i + 1] + t1);
      fx += t1 * t1 + t2 * t2;
    }
    return fx;
  }
};
}  // namespace

void LbfgsTuner::tune_impl([[maybe_unused]] Estimator& estimator, ConstMatrixRef training_inputs,
                           ConstMatrixRef training_outputs) const {
  LUCID_DEBUG_FMT("LbfgsOptimiser::Optimise([{}x{}, [{}x{}])", training_inputs.rows(), training_inputs.cols(),
                  training_outputs.rows(), training_outputs.cols());
  const int n = 10;
  // Set up parameters
  LBFGSpp::LBFGSParam<double> param;
  param.epsilon = 1e-6;
  param.max_iterations = 100;

  // Create solver and function object
  LBFGSpp::LBFGSSolver<double> solver(param);
  Rosenbrock fun(n);

  // Initial guess
  Vector x_out = Vector::Zero(n);
  // x will be overwritten to be the best point found
  double fx;
  const int niter = solver.minimize(fun, x_out, fx);

  std::cout << niter << " iterations" << std::endl;
  std::cout << "x = \n" << training_inputs.transpose() << std::endl;
  std::cout << "f(x) = " << fx << std::endl;

  LUCID_NOT_IMPLEMENTED();
}

}  // namespace lucid
