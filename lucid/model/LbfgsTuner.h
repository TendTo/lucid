/**
 * @author Room 6.030
 * @copyright 2025 lucid
 * @licence BSD 3-Clause License
 * @file
 * LbfgsTuner class.
 */
#pragma once

#include <memory>

#include "lucid/model/Kernel.h"
#include "lucid/model/Tuner.h"

namespace lucid {

/**
 * Structure copied from [lbfgscpp](http://github.com/yixuan/LBFGSpp),
 * to avoid a transitive dependency on the external library in the header.
 */
struct LbgsParameters {
  /**
   * The number of corrections to approximate the inverse Hessian matrix.
   * The L-BFGS routine stores the computation results of previous \ref m
   * iterations to approximate the inverse Hessian matrix of the current
   * iteration. This parameter controls the size of the limited memories
   * (corrections). The default value is \c 6. Values less than \c 3 are
   * not recommended. Large values will result in excessive computing time.
   */
  int m{6};
  /**
   * Absolute tolerance for convergence test.
   * This parameter determines the absolute accuracy \f$\epsilon_{abs}\f$
   * with which the solution is to be found. A minimization terminates when
   * \f$||g|| < \max\{\epsilon_{abs}, \epsilon_{rel}||x||\}\f$,
   * where \f$||\cdot||\f$ denotes the Euclidean (L2) norm. The default value is
   * \c 1e-5.
   */
  Scalar epsilon{1e-5};
  /**
   * Relative tolerance for convergence test.
   * This parameter determines the relative accuracy \f$\epsilon_{rel}\f$
   * with which the solution is to be found. A minimization terminates when
   * \f$||g|| < \max\{\epsilon_{abs}, \epsilon_{rel}||x||\}\f$,
   * where \f$||\cdot||\f$ denotes the Euclidean (L2) norm. The default value is
   * \c 1e-5.
   */
  Scalar epsilon_rel{1e-5};
  /**
   * Distance for delta-based convergence test.
   * This parameter determines the distance \f$d\f$ to compute the
   * rate of decrease of the objective function,
   * \f$f_{k-d}(x)-f_k(x)\f$, where \f$k\f$ is the current iteration
   * step. If the value of this parameter is zero, the delta-based convergence
   * test will not be performed. The default value is \c 0.
   */
  int past{0};
  /**
   * Delta for convergence test.
   * The algorithm stops when the following condition is met,
   * \f$|f_{k-d}(x)-f_k(x)|<\delta\cdot\max(1, |f_k(x)|, |f_{k-d}(x)|)\f$, where \f$f_k(x)\f$ is
   * the current function value, and \f$f_{k-d}(x)\f$ is the function value
   * \f$d\f$ iterations ago (specified by the \ref past parameter).
   * The default value is \c 0.
   */
  Scalar delta{0};
  /**
   * The maximum number of iterations.
   * The optimization process is terminated when the iteration count
   * exceeds this parameter. Setting this parameter to zero continues an
   * optimization process until a convergence or error. The default value
   * is \c 0.
   */
  int max_iterations{0};
  /**
   * The line search termination condition.
   * This parameter specifies the line search termination condition that will be used
   * by the LBFGS routine. The default value is `LBFGS_LINESEARCH_BACKTRACKING_STRONG_WOLFE`.
   */
  int linesearch{3};
  /**
   * The maximum number of trials for the line search.
   * This parameter controls the number of function and gradients evaluations
   * per iteration for the line search routine. The default value is \c 20.
   */
  int max_linesearch{20};
  /**
   * The minimum step length allowed in the line search.
   * The default value is \c 1e-20. Usually this value does not need to be
   * modified.
   */
  Scalar min_step{1e-20};
  /**
   * The maximum step length allowed in the line search.
   * The default value is \c 1e+20. Usually this value does not need to be
   * modified.
   */
  Scalar max_step{1e-20};
  /**
   * A parameter to control the accuracy of the line search routine.
   * The default value is \c 1e-4. This parameter should be greater
   * than zero and smaller than \c 0.5.
   */
  Scalar ftol{1e-4};
  /**
   * The coefficient for the Wolfe condition.
   * This parameter is valid only when the line-search
   * algorithm is used with the Wolfe condition.
   * The default value is \c 0.9. This parameter should be greater
   * the \ref ftol parameter and smaller than \c 1.0.
   */
  Scalar wolfe{0.9};
};

/**
 * Optimiser that uses the L-BFGS algorithm.
 */
class LbfgsTuner final : public Tuner {
 public:
  explicit LbfgsTuner(const LbgsParameters& parameters = {});

 private:
  void tune_impl(Estimator& estimator, ConstMatrixRef training_inputs, ConstMatrixRef training_outputs) const override;

  LbgsParameters parameters_;
};

}  // namespace lucid
