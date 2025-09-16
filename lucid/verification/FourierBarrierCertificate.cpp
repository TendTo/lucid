/**
 * @author c3054737
 * @copyright 2025 lucid
 * @licence BSD 3-Clause License
 * @file
 */
#include "lucid/verification/FourierBarrierCertificate.h"

#include "lucid/model/TruncatedFourierFeatureMap.h"
#include "lucid/util/Stats.h"
#include "lucid/util/constants.h"
#include "lucid/util/error.h"
#include "lucid/verification/AlglibOptimiser.h"
#include "lucid/verification/GurobiOptimiser.h"
#include "lucid/verification/HighsOptimiser.h"

namespace lucid {

bool FourierBarrierCertificate::synthesize(ConstMatrixRef fx_lattice, ConstMatrixRef fxp_lattice,
                                           ConstMatrixRef fx0_lattice, ConstMatrixRef fxu_lattice,
                                           const TruncatedFourierFeatureMap& feature_map,
                                           const Dimension num_frequency_samples_per_dim, const double C_coeff,
                                           const double epsilon, const double target_norm, const double b_kappa) {
  if constexpr (constants::GUROBI_BUILD) {
    return synthesize(GurobiOptimiser{}, fx_lattice, fxp_lattice, fx0_lattice, fxu_lattice, feature_map,
                      num_frequency_samples_per_dim, C_coeff, epsilon, target_norm, b_kappa);
  }
  if constexpr (constants::HIGHS_BUILD) {
    return synthesize(HighsOptimiser{}, fx_lattice, fxp_lattice, fx0_lattice, fxu_lattice, feature_map,
                      num_frequency_samples_per_dim, C_coeff, epsilon, target_norm, b_kappa);
  }
  if constexpr (constants::ALGLIB_BUILD) {
    return synthesize(AlglibOptimiser{}, fx_lattice, fxp_lattice, fx0_lattice, fxu_lattice, feature_map,
                      num_frequency_samples_per_dim, C_coeff, epsilon, target_norm, b_kappa);
  }
  LUCID_NOT_SUPPORTED_MISSING_BUILD_DEPENDENCY("synthesize", "Gurobi, HiGHS or Alglib.");
}
bool FourierBarrierCertificate::synthesize(const Optimiser& optimiser, ConstMatrixRef fx_lattice,
                                           ConstMatrixRef fxp_lattice, ConstMatrixRef fx0_lattice,
                                           ConstMatrixRef fxu_lattice, const TruncatedFourierFeatureMap& feature_map,
                                           const Dimension num_frequency_samples_per_dim, const double C_coeff,
                                           const double epsilon, const double target_norm, const double b_kappa) {
  LUCID_CHECK_ARGUMENT_CMP(num_frequency_samples_per_dim, >, 0);
  TimerGuard tg{Stats::Scoped::top() ? &Stats::Scoped::top()->value().barrier_timer : nullptr};
  const Dimension rkhs_dim = fx_lattice.cols();
  const Dimension num_variables = rkhs_dim + 2 + 4;
  const Dimension num_constraints =
      1 + 2 * (fx_lattice.rows() + fx0_lattice.rows() + fxu_lattice.rows() + fxp_lattice.rows());
  // Determine number of samples per dimension required in the associated lattice on the periodic space
  const double num_frequency_samples_per_dim_periodic =
      (num_frequency_samples_per_dim * feature_map.periodic_coefficients().array()).minCoeff();
  // Determines the most sparse sampled dimension
  const double fraction =
      static_cast<double>(feature_map.num_frequencies() - 1) / num_frequency_samples_per_dim_periodic;
  const double C =
      std::pow(1 - C_coeff * 2.0 * fraction, -static_cast<double>(feature_map.x_limits().dimension()) / 2.0);
  const double maxXX_coeff = -(C - 1) / (C + 1);
  const double fctr1 = 2 / (C + 1);
  const double fctr2 = (C - 1) / (C + 1);
  const double unsafe_rhs = fctr1 * gamma_;
  const double kushner_rhs = -fctr1 * epsilon * target_norm * std::abs(feature_map.sigma_f());

  LUCID_DEBUG_FMT("rkhs_dim: {}, num_variables: {}, num_constraints: {}", rkhs_dim, num_variables, num_constraints);
  LUCID_DEBUG_FMT("num_frequencies_per_dim: {}, num_frequency_samples_per_dim_periodic {}",
                  feature_map.num_frequencies(), num_frequency_samples_per_dim_periodic);
  LUCID_DEBUG_FMT("fraction: {}, C_coeff: {}, C: {}", fraction, C_coeff, C);
  LUCID_DEBUG_FMT("maxXX_coeff: {}, fctr1: {}, fctr2: {}, unsafe_rhs: {}, kushner_rhs: {}", maxXX_coeff, fctr1, fctr2,
                  unsafe_rhs, kushner_rhs);

  if (Stats::Scoped::top()) {
    Stats::Scoped::top()->value().num_variables = num_variables;
    Stats::Scoped::top()->value().num_constraints = num_constraints;
  }
  return optimiser.solve_fourier_barrier_synthesis(
      FourierBarrierSynthesisParameters{
          .num_vars = num_variables,
          .num_constraints = num_constraints,
          .fx_lattice = fx_lattice,
          .fxp_lattice = fxp_lattice,
          .fx0_lattice = fx0_lattice,
          .fxu_lattice = fxu_lattice,
          .T = T_,
          .gamma = gamma_,
          .C = C,
          .b_kappa = b_kappa,
          .maxXX_coeff = maxXX_coeff,
          .fctr1 = fctr1,
          .fctr2 = fctr2,
          .unsafe_rhs = unsafe_rhs,
          .kushner_rhs = kushner_rhs,
      },
      std::bind(&FourierBarrierCertificate::optimiser_callback, this, std::placeholders::_1, std::placeholders::_2,
                std::placeholders::_3, std::placeholders::_4, std::placeholders::_5, std::placeholders::_6,
                target_norm));
}

double FourierBarrierCertificate::apply_impl(ConstVectorRef x) const {
  LUCID_CHECK_ARGUMENT_EQ(is_synthesized(), true);
  LUCID_CHECK_ARGUMENT_EQ(x.size(), coefficients_.size());
  return coefficients_.dot(x);
}

void FourierBarrierCertificate::optimiser_callback(bool success, double obj_val, const Vector& coefficients, double eta,
                                                   double c, double norm, double target_norm) {
  if (!success) return;
  coefficients_ = coefficients;
  eta_ = eta;
  c_ = c;
  norm_ = norm;
  safety_ = 1 - obj_val;

  LUCID_DEBUG_FMT("success: {}, obj_val: {}, norm: {}, eta: {}, c: {}", success, obj_val, norm, eta, c);
  LUCID_DEBUG_FMT("coefficients: {}", LUCID_FORMAT_VECTOR(coefficients_));
  LUCID_INFO_FMT("Solution found, objective = {}", obj_val);
  LUCID_INFO_FMT("Satisfaction probability is {:.6f}%", safety_ * 100);
  if (norm > target_norm) {
    LUCID_WARN_FMT("Actual norm exceeds bound: {} > {} (diff: {})", norm, target_norm, norm - target_norm);
  }
}

std::unique_ptr<BarrierCertificate> FourierBarrierCertificate::clone() const {
  return std::make_unique<FourierBarrierCertificate>(*this);
}

std::ostream& operator<<(std::ostream& os, const FourierBarrierCertificate& obj) {
  if (!obj.is_synthesized()) return os << "FourierBarrierCertificate( )";
  return os << "FourierBarrierCertificate( eta( " << obj.eta() << " ) gamma( " << obj.gamma() << " ) c( " << obj.c()
            << " ) norm( " << obj.norm() << " ) T( " << obj.T() << " ) safety( " << obj.safety() << " ) )";
}

}  // namespace lucid
