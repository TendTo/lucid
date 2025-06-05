/**
 * @author Room 6.030
 * @copyright 2025 lucid
 * @licence BSD 3-Clause License
 * @file
 * LbfgsTuner class.
 */
#pragma once

#include <vector>

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
   * The maximum number of iterations in the subspace minimization.
   * This parameter controls the number of iterations in the subspace
   * minimization routine. The default value is \c 10.
   * Only used when bounds are set.
   */
  int max_submin{10};
  /**
   * The maximum number of trials for the line search.
   * This parameter controls the number of function and gradients evaluations
   * per iteration for the line search routine. The default value is \c 20.
   * Only used when bounds are not set.
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
  Scalar max_step{1e+20};
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
 * It optimizes the parameters of an estimator by minimizing the loss function.
 * It can operate with or without bounds on the parameters.
 * If bounds are provided, they must match the dimension of the solution space.
 * Note that the solution space for isotropic estimators
 * (i.e., those where the parameters are the same for all dimensions)
 * will always have a dimension of 1.
 */
class LbfgsTuner final : public Tuner {
 public:
  /**
   * Construct a new LbfgsTuner with the provided `parameters`.
   * @param parameters optimization parameters for the L-BFGS algorithm
   */
  explicit LbfgsTuner(const LbgsParameters& parameters = {});
  /**
   * Construct a new LbfgsTuner with the provided bounds and `parameters`.
   * The solution will be constrained to the provided bounds.
   * The size of the bounds must match the dimension of the solution space.
   * @pre `lb` and `ub` must have the same size
   * @param lb Lower bounds for the parameters
   * @param ub Upper bounds for the parameters
   * @param parameters optimization parameters for the L-BFGS algorithm
   */
  explicit LbfgsTuner(const Eigen::VectorXd& lb, const Eigen::VectorXd& ub, const LbgsParameters& parameters = {});
  /**
   * Construct a new LbfgsTuner with the provided bounds and `parameters`.
   * The solution will be constrained to the provided bounds.
   * The size of the bounds must match the dimension of the solution space.
   * @param bounds vector of pairs of lower and upper bounds
   * @param parameters optimization parameters for the L-BFGS algorithm
   */
  explicit LbfgsTuner(const std::vector<std::pair<Scalar, Scalar>>& bounds, const LbgsParameters& parameters = {});

  /** @checker{bounded, optimisation} */
  [[nodiscard]] bool is_bounded() const;

 private:
  void tune_impl(Estimator& estimator, ConstMatrixRef training_inputs, ConstMatrixRef training_outputs) const override;

  Eigen::VectorXd lb_;         ///< Lower bounds for the parameters. If empty, no bounds are applied
  Eigen::VectorXd ub_;         ///< Upper bounds for the parameters. If empty, no bounds are applied
  LbgsParameters parameters_;  ///< Optimization parameters for the L-BFGS algorithm
};

}  // namespace lucid
