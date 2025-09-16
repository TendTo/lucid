/**
 * @author c3054737
 * @copyright 2025 lucid
 * @licence BSD 3-Clause License
 * @file
 * BarrierCertificate class.
 */
#pragma once

#include <iosfwd>

#include "lucid/lib/eigen.h"

namespace lucid {

/**
 * Control barrier certificates (CBCs) are often used to certify the safety of a stochastic systems.
 * A CBC over a domain @XsubRd with respect to an unsafe set @Xu and transition function @f$ f: \mathcal{X} \times
 * \mathcal{U} \to \mathcal{X} @f$, where @f$ \mathcal{U} \subseteq \mathbb{R}^m @f$ is the set of control inputs,
 * is a function @f$ B: \mathcal{X} \to \mathbb{R} @f$ with the following properties:
 * @f[
 * \begin{align*}
 * &\forall x \in \mathcal{X} && B(x) \geq 0 \\
 * &\forall x_0 \in \mathcal{X}_0 && B(x_0) \leq \eta \newline
 * &\forall x_u \in \mathcal{X}_u && B(x_u) \geq \gamma \newline
 * &\forall x \in \mathcal{X}, \forall u \in \mathcal{U} && \mathbb{E}[B(x_+)] - B(x) \leq c
 * \end{align*}
 * @f]
 * for some @f$ \eta, \gamma, c \in \mathbb{R} @f$ with @f$ \gamma > \eta \geq 0@f$ and @f$ c \geq 0 @f$.
 * In this context, @f$ x_+ @f$ is the next state of the system
 * after applying the control input @f$ u @f$ at state @x.
 * If such a function exists, then the safety probability of the system can be bounded by
 * @f[
 * P_\text{safe} \geq 1 - \frac{\eta + c T}{\gamma}
 * @f]
 * where @f$ T @f$ is the time horizon, i.e., the number of time steps to consider.
 */
class BarrierCertificate {
 public:
  /**
   * Construct a new BarrierCertificate object.
   * @pre `T` must be greater than 0.
   * @pre `gamma` must be greater than or equal to 0.
   * @pre `eta` must be greater than or equal to 0.
   * @pre `c` must be greater than or equal to 0.
   * @param T time horizon
   * @param gamma @gamma value in the CBC definition
   * @param eta @eta value in the CBC definition
   * @param c @f$ c @f$ value in the CBC definition
   */
  explicit BarrierCertificate(int T, double gamma, double eta = 0, double c = 0);
  virtual ~BarrierCertificate() = default;

  /**
   * Evaluate the barrier certificate at the given point.
   * @pre `x` must belong to the input space of the barrier certificate.
   * @param x input vector
   * @return value of the barrier certificate at the given point
   */
  [[nodiscard]] double operator()(ConstVectorRef x) const;

  /** @checker{synthesized, barrier certificate} */
  [[nodiscard]] bool is_synthesized() const { return norm_ != 0; }
  /** @getter{norm describing the complexity of the barrier, barrier certificate} */
  [[nodiscard]] double norm() const { return norm_; }
  /** @getter{@gamma value in the CBC definition, barrier certificate} */
  [[nodiscard]] double gamma() const { return gamma_; }
  /** @getter{@eta value in the CBC definition, barrier certificate} */
  [[nodiscard]] double eta() const { return eta_; }
  /** @getter{@f$ c @f$ value in the CBC definition, barrier certificate} */
  [[nodiscard]] double c() const { return c_; }
  /** @getter{safety probability of the system, barrier certificate} */
  [[nodiscard]] double safety() const { return safety_; }
  /** @getter{time horizon, barrier certificate} */
  [[nodiscard]] int T() const { return T_; }

  /**
   * Clone the barrier certificate.
   * Create a new instance of the barrier certificate with the same parameters.
   * If the barrier was synthesized, the new instance will have the same synthesis results.
   * @return new instance of the barrier certificate
   */
  [[nodiscard]] virtual std::unique_ptr<BarrierCertificate> clone() const = 0;

 protected:
  /**
   * Concrete implementation of @ref operator()().
   * @param x input vector
   * @return value of the barrier certificate at the given point
   */
  [[nodiscard]] virtual double apply_impl(ConstVectorRef x) const = 0;

  int T_;          ///< Time horizon
  double gamma_;   ///< @gamma value in the CBC definition
  double eta_;     ///< @eta value in the CBC definition
  double c_;       ///< @f$ c @f$ value in the CBC definition
  double norm_;    ///< Norm describing the complexity of the barrier. It is 0 if the barrier has not been synthesized
  double safety_;  ///< Safety probability of the system. It is 0 if the barrier has not been synthesized
};

std::ostream& operator<<(std::ostream& os, const BarrierCertificate& barrier);

}  // namespace lucid

#ifdef LUCID_INCLUDE_FMT

#include "lucid/util/logging.h"

OSTREAM_FORMATTER(lucid::BarrierCertificate)

#endif
