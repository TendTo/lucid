/**
 * @author Room 6.030
 * @copyright 2025 lucid
 * @licence BSD 3-Clause License
 * @file
 * LbfgsOptimiser class.
 */
#include "lucid/tuning/LbfgsOptimiser.h"

#include <LBFGS.h>

#include <iostream>

#include "lucid/lib/eigen.h"
#include "lucid/util/logging.h"

namespace lucid::tuning {

namespace {
class Rosenbrock {
 private:
  int n;

 public:
  Rosenbrock(int n_) : n(n_) {}
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

std::unique_ptr<Kernel> LbfgsOptimiser::Optimise(const Kernel& kernel) const {
  LUCID_DEBUG_FMT("LbfgsOptimiser::Optimise({})", kernel);
  const int n = 10;
  // Set up parameters
  LBFGSpp::LBFGSParam<double> param;
  param.epsilon = 1e-6;
  param.max_iterations = 100;

  // Create solver and function object
  LBFGSpp::LBFGSSolver<double> solver(param);
  Rosenbrock fun(n);

  // Initial guess
  Vector x = Vector::Zero(n);
  // x will be overwritten to be the best point found
  double fx;
  const int niter = solver.minimize(fun, x, fx);

  std::cout << niter << " iterations" << std::endl;
  std::cout << "x = \n" << x.transpose() << std::endl;
  std::cout << "f(x) = " << fx << std::endl;

  return std::unique_ptr<Kernel>{};
}

}  // namespace lucid::tuning
