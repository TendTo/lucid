/**
 * @author c3054737
 * @copyright 2025 lucid
 * @licence BSD 3-Clause License
 * @file
 * FourierBarrierCertificate class.
 */
#pragma once

#include <iosfwd>
#include <memory>

#include "lucid/lib/eigen.h"
#include "lucid/model/Estimator.h"
#include "lucid/model/TruncatedFourierFeatureMap.h"
#include "lucid/verification/BarrierCertificate.h"

namespace lucid {

// Forward declaration
class Optimiser;

/** Parameters for the Fourier barrier certificate synthesis using PSO. */
struct FourierBarrierCertificateParameters {
  double increase = 0.1;    ///< Set size percentage increase factor on the periodic domain
  int num_particles = 40;   ///< Number of particles in the swarm
  double phi_local = 0.5;   ///< Cognitive coefficient
  double phi_global = 0.3;  ///< Social coefficient
  double weight = 0.9;      ///< Inertia weight
  int max_iter = 150;       ///< Maximum number of iterations. 0 means no limit
  double max_vel = 0.0;     ///< Maximum velocity for each particle. 0 means no limit
  double ftol = 1e-8;       ///< Function value tolerance for convergence
  double xtol = 1e-8;       ///< Position change tolerance for convergence
  double C_coeff = 1.0;     ///< Used to either strengthen (>1) or weaken (<1) the conservative coefficient C
  double epsilon = 1.0;     ///< Epsilon parameter (?)
  double b_norm = 0.0;      ///< Target norm for the barrier certificate
  double kappa = 1.0;       ///< Kappa parameter (?)
  int threads = 0;          ///< Number of threads to use. 0 means automatic detection
};

/**
 * Barrier certificate using a Fourier basis as a template for the function.
 * The function is defined as follows:
 * @f[
 * B(x) = \phi_M(x)^T b = \alpha_0 + \sum_{j=1}^{M} \alpha_i \cos{\omega_i^T P(x)} + \beta_i \sin{\omega_i^T P(x)}
 * @f]
 * with
 * @f[
 * b = \begin{bmatrix}
 * \frac{\alpha_0}{\sigma_f^2} & \frac{\alpha_1}{2 \sigma_f^2 \omega_1^2} & \frac{\beta_1}{2 \sigma_f^2 \omega_1^2} &
 * \cdots &
 * \frac{\alpha_M}{2 \sigma_f^2 \omega_M^2} & \frac{\beta_M}{2 \sigma_f^2 \omega_M^2}
 * \end{bmatrix}^T
 * @f]
 * @todo Add more details to the formulation, as there are some undefined symbols used.
 */
class FourierBarrierCertificate final : public BarrierCertificate {
 public:
  using BarrierCertificate::BarrierCertificate;

  /** @overload **/
  bool synthesize(int Q_tilde, const Estimator& estimator, const TruncatedFourierFeatureMap& feature_map,
                  const RectSet& X_bounds, const Set& X_init, const Set& X_unsafe,
                  const FourierBarrierCertificateParameters& parameters = {});

  /**
   * Synthesize the barrier certificate.
   * This is done in multiple steps.
   *
   * **Finding the bounds**
   *
   * First, we find some upper and lower bounds on the value the barrier certificate can assume over @X0 and @Xu.
   * For simplicity, we only explain the process for finding the upper bound over @X0, but the same applies
   * to the lower bound over @X0 and bounds over @Xu.
   * We can obtain the lower bound on @X0 by using the following inequality:
   * @f[
   * B(x) \le \hat{B}^{\mathcal{X}_0}_{\tilde{N}}C_{\tilde{N}}
   * + R^{\mathcal{\tilde{X}}\setminus\mathcal{X}_0}_{\tilde{N}}\underbrace{\frac{1}{\tilde{N}}
   * \sum_{\bar{x} \in \Theta_{\tilde{N}}\setminus\mathcal{X}_0} D^n_{f_{\max}-\tilde{Q}-f_{\max}}(x - \bar{x})}
   * _{A^{\mathcal{\tilde{X}\setminus\mathcal{X}_0}}_\tilde{N}}
   * \quad \forall x \in \mathcal{X}_0 ,
   * @f]
   * where @f$ R^{\mathcal{\tilde{X}}\setminus\mathcal{X}_0}_{\tilde{N}} =
   * \max_{x \in \Theta_\tilde{N} \setminus \mathcal{X}_0}\{\phi_M(x)^Tb\} - \hat{B}_\tilde{N}^{\mathcal{X}_0} @f$
   * is the residual outside @X0,
   * and @f$ C_{\tilde{N}} = \left( 1 - \frac{2 f_{\max}}{\tilde{Q}} \right)^{-\frac{n}{2}} @f$.
   * Note that we can find the upper bound @f$ A^{\mathcal{\tilde{X}\setminus\mathcal{X}_0}}_\tilde{N} @f$
   * by solving an optimisation problem before starting the synthesis, using, e.g.,
   * [particle swarm optimisation (PSO)](https://en.wikipedia.org/wiki/Particle_swarm_optimization).
   * @param optimiser LP optimiser to use for the synthesis
   * @param Q_tilde number of lattice points on periodic domain per dimension
   * @param estimator estimator model to compute the value of the feature map on @xp
   * @param feature_map feature map to apply to the lattice points
   * @param X_bounds bounds of the set @X
   * @param X_init initial set @X0
   * @param X_unsafe unsafe set @Xu
   * @param parameters parameters for barrier synthesis
   * @return true if the synthesis was successful
   * @return false if no solution was found
   */
  bool synthesize(const Optimiser& optimiser, int Q_tilde, const Estimator& estimator,
                  const TruncatedFourierFeatureMap& feature_map, const RectSet& X_bounds, const Set& X_init,
                  const Set& X_unsafe, const FourierBarrierCertificateParameters& parameters = {});

  /** @getter{coefficients of the basis, Fourier barrier certificate} */
  [[nodiscard]] const Vector& coefficients() const { return coefficients_; }

  [[nodiscard]] std::unique_ptr<BarrierCertificate> clone() const override;

 private:
  [[nodiscard]] double apply_impl(ConstVectorRef x) const override;

  /**
   * Utility function called by the optimiser when the synthesis is done.
   * Used to store the results of the synthesis into the barrier certificate object.
   * If the synthesis was unsuccessful, the barrier is left unchanged.
   * @param success true if the synthesis was successful
   * @param obj_val objective value
   * @param coefficients coefficients of the basis
   * @param eta @eta value
   * @param c @f$ c @f$ value
   * @param norm actual norm of the barrier function
   * @param b_norm target norm for the coefficients of the basis
   */
  void optimiser_callback(bool success, double obj_val, const Vector& coefficients, double eta, double c, double norm,
                          double b_norm);

  Vector coefficients_;  ///< Coefficients of the Fourier basis
};

std::ostream& operator<<(std::ostream& os, const FourierBarrierCertificateParameters& params);
std::ostream& operator<<(std::ostream& os, const FourierBarrierCertificate& barrier);

}  // namespace lucid

#ifdef LUCID_INCLUDE_FMT

#include "lucid/util/logging.h"

OSTREAM_FORMATTER(lucid::FourierBarrierCertificateParameters);
OSTREAM_FORMATTER(lucid::FourierBarrierCertificate);

#endif
