/**
 * @author c3054737
 * @copyright 2025 lucid
 * @licence BSD 3-Clause License
 * @file
 * MontecarloSimulation class.
 */
#pragma once

#include <cstddef>
#include <string>
#include <utility>

#include "lucid/lib/eigen.h"
#include "lucid/model/Set.h"

namespace lucid {

/**
 * Monte Carlo simulation for estimating the safety probability of a system.
 * The safety probability is estimated by sampling initial states from the initial set
 * and simulating the system dynamics for a given time horizon.
 * The safety probability is then computed as the fraction of samples that do not enter the unsafe set
 * during the simulation.
 * The method also provides a confidence interval for the estimated safety probability
 * using the Chebychev inequality.
 */
class MontecarloSimulation {
 public:
  /** Construct a new MontecarloSimulation object. */
  MontecarloSimulation() = default;

  /**
   * Compute the safety probability with a confidence interval using Monte Carlo simulation.
   * The method samples initial states from the initial set @X0, simulates the system dynamics up to the given
   * `time_horizon` and computes the fraction of samples whose trajectories do not enter the unsafe set @Xu.
   * Note that a trajectory leaving the bounds of the state space is immediately considered safe.
   * The confidence interval is computed using
   * the [Chebychev inequality](https://en.wikipedia.org/wiki/Chebyshev%27s_inequality).
   * Let @f$ X @f$ be a random variable drawn from a Bernoulli distribution with probability @f$ p @f$,
   * i.e., @f$ X \sim \text{Bernoulli}(p) @f$.
   * It follows that the expected value of @f$ X @f$ is @f$ \mathbb{E}[X] = p @f$.
   * @f$ X @f$ indicates whether a trajectory starting within the initial set
   * remains safe (1) or enters the unsafe set (0) within the time horizon.
   * Sampling `num_samples` trajectories independently distributed,
   * we can create a confidence interval around the true safety probability with the
   * [formula](https://doi.org/10.15439/2023F1144)
   * @f[
   * P(|\hat{X} - \mathbb{E}[X]| \geq \varepsilon) \leq \frac{1}{4 n \varepsilon^2} \le (1 - \alpha) ,
   * @f]
   * where @f$ \hat{X} = \frac{1}{n} \sum_{i=1}^{n} X_i @f$ is the sample mean, @f$ n @f$ is the number of samples,
   * @f$ \varepsilon @f$ is the error, and @f$ \alpha @f$ is the confidence level.
   * Hence, having fixed @f$ \alpha @f$ and @f$ n @f$, we can obtain the error bound
   * @f[
   * \varepsilon = \sqrt{\frac{1}{4 n (1 - \alpha)}} ,
   * @f]
   * so that the interval @f$ [\hat{X} - \varepsilon, \hat{X} + \varepsilon] @f$ contains the true safety probability
   * with confidence level @f$ \alpha @f$.
   * @pre `confidence_level` must be in the range @f$ [0, 1) @f$.
   * @pre `num_samples` must be greater than 0.
   * @pre `X_bounds`, `X_init`, and `X_unsafe` must have the same dimension.
   * @pre `time_horizon` must be greater than 0.
   * @note Trajectories that leave the bounds of the state space are considered safe.
   * @param X_bounds set representing the bounds of the state space
   * @param X_init set representing the initial states
   * @param X_unsafe set representing the unsafe states
   * @param system_dynamics function representing the system dynamics
   * @param time_horizon number of time steps to simulate
   * @param confidence_level confidence level for the safety probability interval
   * @param num_samples number of samples to use for the Monte Carlo simulation
   * @return lower and upper bounds of the confidence interval
   */
  [[nodiscard]] std::pair<double, double> safety_probability(
      const Set& X_bounds, const Set& X_init, const Set& X_unsafe,
      const std::function<Vector(ConstVectorRef)>& system_dynamics, std::size_t time_horizon,
      double confidence_level = 0.9, Dimension num_samples = 1000) const;

  [[nodiscard]] std::string to_string() const;
};

std::ostream& operator<<(std::ostream& os, const MontecarloSimulation& sim);

}  // namespace lucid

#ifdef LUCID_INCLUDE_FMT

#include "lucid/util/logging.h"

OSTREAM_FORMATTER(lucid::MontecarloSimulation)

#endif
