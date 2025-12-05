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
#include <string>

#include "lucid/lib/eigen.h"
#include "lucid/model/Estimator.h"
#include "lucid/model/TruncatedFourierFeatureMap.h"
#include "lucid/verification/BarrierCertificate.h"

namespace lucid {

// Forward declaration
class Optimiser;

/** Parameters for the Fourier barrier certificate synthesis using PSO. */
struct FourierBarrierCertificateParameters {
  double set_scaling = 0.1;  ///< Set size percentage set_scaling factor on the periodic domain
  int num_particles = 40;    ///< Number of particles in the swarm
  double phi_local = 0.5;    ///< Cognitive coefficient
  double phi_global = 0.3;   ///< Social coefficient
  double weight = 0.9;       ///< Inertia weight
  int max_iter = 150;        ///< Maximum number of iterations. 0 means no limit
  double max_vel = 0.0;      ///< Maximum velocity for each particle. 0 means no limit
  double ftol = 1e-8;        ///< Function value tolerance for convergence
  double xtol = 1e-8;        ///< Position change tolerance for convergence
  double C_coeff = 1.0;      ///< Used to either strengthen (>1) or weaken (<1) the conservative coefficient C
  double epsilon = 1.0;      ///< Epsilon parameter (?)
  double b_norm = 0.0;       ///< Target norm for the barrier certificate
  double kappa = 1.0;        ///< Kappa parameter (?)
  int threads = 0;           ///< Number of threads to use. 0 means automatic detection

  [[nodiscard]] std::string to_string() const;
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
 */
class FourierBarrierCertificate final : public BarrierCertificate {
 public:
  using BarrierCertificate::BarrierCertificate;

  static double compute_A(int lattice_resolution, int f_max, const RectSet& pi, const RectSet& X_tilde, const Set& X,
                          const Matrix& lattice, const FourierBarrierCertificateParameters& parameters);
  static double compute_A(int lattice_resolution, int f_max, const RectSet& X_tilde, const Set& X,
                          const FourierBarrierCertificateParameters& parameters);

  /** @overload **/
  bool synthesize(int lattice_resolution, const Estimator& estimator, const TruncatedFourierFeatureMap& feature_map,
                  const RectSet& X_bounds, const Set& X_init, const Set& X_unsafe,
                  const FourierBarrierCertificateParameters& parameters = {});

  /**
   * Synthesize the barrier certificate.
   * This is done in multiple steps.
   *
   * ### Bounding the contribution from outside the sets of interest
   *
   * Let @Xn be the periodic domain induced by the Fourier feature map, i.e., the smallest hyperrectangle such that
   * the lowest non-zero frequency used by the barrier completes one period in.
   * Then, we compute the contribution of all the points in the periodic domain
   * outside the sets of interest @X, @X0, and @Xu using Particle Swarm Optimisation (PSO).
   * The overapproximation of these contribution are defined as
   * @f[
   * A^{\tilde{\mathcal{X}}\setminus\mathbb{S}}_{\tilde{N}} \coloneqq \frac{1}{\tilde{N}}
   * \sum_{\bar{x}\in\Theta_{\tilde{N}}\setminus\mathbb{S}} 
   * D^n_{f_{\text{max}},\tilde{Q}-f_{\text{max}}}(x-\bar{x}),
   * @f]
   * where @f$\mathbb{S}\in\{\mathcal{X},\mathcal{X}_0,\mathcal{X}_u\}@f$, @f$\Theta_{\tilde{N}}@f$
   * is a lattice of cardinality @f$\tilde{N}@f$ on the periodic domain,
   * and @f$D^n_{f_{\text{max}},\tilde{Q}-f_{\text{max}}}@f$ is the ValleePoussinKernel.
   *
   * ### Linear Program formulation
   *
   * Having determined the lattices @f$\smash{\Theta_{\hat{N}}}@f$ and @f$\smash{\Theta_{\tilde{N}}}@f$, we form the
   * discrete sets @f$\{ x_0^{(1)}, \ldots, x_0^{(\hat{N}_0)} \} \subset \mathcal{X}_0@f$ and @f$\{ x_u^{(1)}, \ldots,
   * x_u^{(\hat{N}_u)} \} \subset \mathcal{X}_u@f$ of cardinality @f$\hat{N}_0,\hat{N}_u\in\N@f$, respectively. For
   * given @f$\bar{B}@f$ and @f$\gamma@f$, we obtain the following LP:
   *
   * @f[
   * \begin{align*}
   *                      & \min_{\substack{b, c, \eta \\
   *             \check{B}_{\tilde{N}}^{\mathcal{X}_0},\hat{B}_{\tilde{N}}^{\mathcal{X}_u},
   *             \check{B}_\Delta^{\mathcal{X}},
   *             \hat{B}_{\tilde{N}}^{{\mathcal{X}}}
   * \\
   *             \hat{B}_{\tilde{N}}^{\tilde{\mathcal{X}}\setminus\mathcal{X}_0},
   *             \check{B}_{\tilde{N}}^{\tilde{\mathcal{X}}\setminus\mathcal{X}_u},
   *             \check{B}_{\tilde{N}}^{\tilde{\mathcal{X}}\setminus\mathcal{X}},
   *             \hat{B}_\Delta^{\tilde{\mathcal{X}}\setminus\mathcal{X}}
   *             }} \quad &                            & \eta + cT, & &                         \\ & \text{subject
   * to}\quad
   *                      &                            & \check{B}_{\tilde{N}}^{\mathcal{X}_0}\leq\phi_M(x_0^{(i)})^\top
   * b\leq\hat{\eta}, \quad &                                                                        &
   * i=1,\ldots,\hat{N}_0,   \\ &                            & & \hat{\gamma}\leq\phi_M(x_u^{(i)})^\top
   * b\leq\hat{B}_{\tilde{N}}^{\mathcal{X}_u},
   *             \quad    &                            & i=1,\ldots,\hat{N}_u, \\ &                            & &
   * \check{B}_\Delta^{\mathcal{X}}\leq\phi_M(x^{(i)})^\top(Hb - b) \leq \hat{\Delta},
   *             \quad    &                            & i=1,\ldots,\hat{N}, \\ &                            & &
   * \hat{\xi}\leq\phi_M(x^{(i)})^\top b\leq\hat{B}_{\tilde{N}}^{{\mathcal{X}}},
   *             \quad    &                            & i=1,\ldots,\hat{N}, \\ &                            & &
   * \phi_M(x^{(i)})^\top b\leq\hat{B}_{\tilde{N}}^{\tilde{\mathcal{X}}\setminus\mathcal{X}_0},
   *             \quad    &                            & i=1,\ldots,\tilde{N}-\hat{N}_0, \\ & & &
   * \check{B}_{\tilde{N}}^{\tilde{\mathcal{X}}\setminus\mathcal{X}_u}\leq\phi_M(x^{(i)})^\top b,
   *             \quad    &                            & i=1,\ldots,\tilde{N}-\hat{N}_u, \\ & & &
   * \check{B}_{\tilde{N}}^{\tilde{\mathcal{X}}\setminus\mathcal{X}}\leq\phi_M(x^{(i)})^\top b,
   *             \quad    &                            & i=1,\ldots,\tilde{N}-\hat{N}, \\ &                            &
   * & \phi_M(x^{(i)})^\top (Hb - b)\leq\hat{B}_\Delta^{\tilde{\mathcal{X}}\setminus\mathcal{X}},
   *             \quad    &                            & i=1,\ldots,\tilde{N}-\hat{N}, \\ &                            &
   * & c\geq 0,\,\gamma>\eta\geq 0,\, b\in\mathbb{R}^{2M+1},                          &                       &
   * \end{align*}
   * @f]
   *
   * with @f$\kappa\geq\sigma_f@f$, @f$\bar{B}\geq\left|\left|b\right|\right|_2@f$, and constraint-tightening
   * coefficients
   *
   * @f[
   * \begin{align*}
   *       \hat{\eta}   & \coloneqq \frac{2\eta +
   * (C_{\tilde{N}}-1)\check{B}_{\tilde{N}}^{\mathcal{X}_0}-2A^{\tilde{\mathcal{X}}\setminus\mathcal{X}_0}_{\tilde{N}}
   * \hat{B}_{\tilde{N}}^{\tilde{\mathcal{X}}\setminus\mathcal{X}_0}}
   * {C_{\tilde{N}}-2A^{\tilde{\mathcal{X}}\setminus\mathcal{X}_0}_{\tilde{N}}+1},
   *                    & & \hat{\gamma} \coloneqq \frac{2\gamma +
   * (C_{\tilde{N}}-1)\hat{B}_{\tilde{N}}^{\mathcal{X}_u}-2A^{\tilde{\mathcal{X}}\setminus\mathcal{X}_u}_{\tilde{N}}
   * \check{B}_{\tilde{N}}^{\tilde{\mathcal{X}}\setminus\mathcal{X}_u}}
   * {C_{\tilde{N}}-2A^{\tilde{\mathcal{X}}\setminus\mathcal{X}_u}_{\tilde{N}}+1},
   * \\
   *       \hat{\Delta} & \coloneqq \frac{2(c - \varepsilon\bar{B}\kappa) +
   * (C_{\tilde{N}}-1)\check{B}_\Delta^\mathcal{X}-2A^{\tilde{\mathcal{X}}\setminus\mathcal{X}}_{\tilde{N}}
   * \hat{B}_{\Delta}^{\tilde{\mathcal{X}}\setminus\mathcal{X}}}
   * {C_{\tilde{N}}-2A^{\tilde{\mathcal{X}}\setminus\mathcal{X}}_{\tilde{N}}+1},
   *                    & & \hat{\xi} \coloneqq
   * \frac{(C_{\tilde{N}}-1)\hat{B}_{\tilde{N}}^{{\mathcal{X}}}-2A^{\tilde{\mathcal{X}}\setminus\mathcal{X}}_{\tilde{N}}
   * \check{B}_{\tilde{N}}^{\tilde{\mathcal{X}}\setminus\mathcal{X}}}
   * {C_{\tilde{N}}-2A^{\tilde{\mathcal{X}}\setminus\mathcal{X}}_{\tilde{N}}+1}
   * .
   * \end{align*}
   * @f]
   *
   * Any solution to the LP that satisfies @f$\left|\left|b\right|\right|_2\leq\bar{B}@f$ determines a valid barrier.
   *
   *
   * @param optimiser LP optimiser to use for the synthesis
   * @param lattice_resolution number of lattice points on periodic domain per dimension
   * @param estimator estimator model to compute the value of the feature map on @xp
   * @param feature_map feature map to apply to the lattice points
   * @param X_bounds bounds of the set @X
   * @param X_init initial set @X0
   * @param X_unsafe unsafe set @Xu
   * @param parameters parameters for barrier synthesis
   * @return true if the synthesis was successful
   * @return false if no solution was found
   * @see [Bounding Multivariate Trigonometric Polynomials](https://doi.org/10.1109/TSP.2018.2883925)
   */
  bool synthesize(const Optimiser& optimiser, int lattice_resolution, const Estimator& estimator,
                  const TruncatedFourierFeatureMap& feature_map, const RectSet& X_bounds, const Set& X_init,
                  const Set& X_unsafe, const FourierBarrierCertificateParameters& parameters = {});

  /** @getter{coefficients of the basis, Fourier barrier certificate} */
  [[nodiscard]] const Vector& coefficients() const { return coefficients_; }

  [[nodiscard]] std::unique_ptr<BarrierCertificate> clone() const override;

  [[nodiscard]] std::string to_string() const override;

 private:
  [[nodiscard]] double apply_impl(ConstVectorRef x) const override;

  /**
   * Utility function called by the optimiser when the synthesis is done.
   * Used to store the results of the synthesis into the barrier certificate object.
   * If the synthesis was unsuccessful, the barrier is left unchanged.
   * @param success true if the synthesis was successful
   * @param obj_val objective value
   * @param coefficients coefficients of the basis
   * @param eta @eta_ value
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
