/**
 * @author c3054737
 * @copyright 2025 lucid
 * @licence BSD 3-Clause License
 * @file
 * FourierBarrierCertificate class.
 */
#pragma once
#include <iosfwd>

#include "lucid/lib/eigen.h"
#include "lucid/model/TruncatedFourierFeatureMap.h"
#include "lucid/verification/BarrierCertificate.h"

namespace lucid {

// Forward declaration
class Optimiser;

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

  /**
   * Synthesize the barrier certificate by solving a linear optimisation problem.
   * @param fx_lattice lattice obtained from the initial set after applying the feature map
   * @param fxp_lattice lattice obtained from the successor of the initial set after applying the feature map
   * @param fx0_lattice lattice obtained from the initial set after applying the feature map
   * @param fxu_lattice lattice obtained from the unsafe set after applying the feature map
   * @param feature_map feature map used to obtain the lattices
   * @param num_frequency_samples_per_dim number of frequency samples per dimension
   * @param C_coeff coefficient used in the conservative term of the optimisation problem
   * @param epsilon tolerance for the optimisation problem
   * @param target_norm target norm for the coefficients of the basis
   * @param b_kappa bound on the RKHS norm of the barrier certificate
   * @return true if the synthesis was successful
   * @return false if no solution was found
   */
  bool synthesize(ConstMatrixRef fx_lattice, ConstMatrixRef fxp_lattice, ConstMatrixRef fx0_lattice,
                  ConstMatrixRef fxu_lattice, const TruncatedFourierFeatureMap& feature_map,
                  Dimension num_frequency_samples_per_dim, double C_coeff = 1, double epsilon = 0,
                  double target_norm = 1, double b_kappa = 1);
  /**
   * Synthesize the barrier certificate by solving a linear optimisation problem.
   * @param optimiser optimiser to use for the synthesis
   * @param fx_lattice lattice obtained from the initial set after applying the feature map
   * @param fxp_lattice lattice obtained from the successor of the initial set after applying the feature map
   * @param fx0_lattice lattice obtained from the initial set after applying the feature map
   * @param fxu_lattice lattice obtained from the unsafe set after applying the feature map
   * @param feature_map feature map used to obtain the lattices
   * @param num_frequency_samples_per_dim number of frequency samples per dimension
   * @param C_coeff coefficient used in the conservative term of the optimisation problem
   * @param epsilon tolerance for the optimisation problem
   * @param target_norm target norm for the coefficients of the basis
   * @param b_kappa bound on the RKHS norm of the barrier certificate
   * @return true if the synthesis was successful
   * @return false if no solution was found
   */
  bool synthesize(const Optimiser& optimiser, ConstMatrixRef fx_lattice, ConstMatrixRef fxp_lattice,
                  ConstMatrixRef fx0_lattice, ConstMatrixRef fxu_lattice, const TruncatedFourierFeatureMap& feature_map,
                  Dimension num_frequency_samples_per_dim, double C_coeff = 1, double epsilon = 0,
                  double target_norm = 1, double b_kappa = 1);

  /** @getter{coefficients of the basis, Fourier barrier certificate} */
  [[nodiscard]] const Vector& coefficients() const { return coefficients_; }

  [[nodiscard]] std::unique_ptr<BarrierCertificate> clone() const override;

 private:
  [[nodiscard]] double apply_impl(ConstVectorRef x) const override;

  /**
   * Utility function called by the optimiser when the synthesis is done.
   * @param success true if the synthesis was successful
   * @param obj_val objective value
   * @param coefficients coefficients of the basis
   * @param eta @eta value
   * @param c @f$ c @f$ value
   * @param norm actual norm of the barrier function
   * @param target_norm target norm for the coefficients of the basis
   */
  void optimiser_callback(bool success, double obj_val, const Vector& coefficients, double eta, double c, double norm,
                          double target_norm);

  Vector coefficients_;  ///< Coefficients of the Fourier basis
};

std::ostream& operator<<(std::ostream& os, const FourierBarrierCertificate& obj);

}  // namespace lucid

#ifdef LUCID_INCLUDE_FMT

#include "lucid/util/logging.h"

OSTREAM_FORMATTER(lucid::FourierBarrierCertificate);

#endif
