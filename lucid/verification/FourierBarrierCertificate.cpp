/**
 * @author c3054737
 * @copyright 2025 lucid
 * @licence BSD 3-Clause License
 * @file
 */
#include "lucid/verification/FourierBarrierCertificate.h"

#include <memory>

#include "lucid/lib/psocpp.h"
#include "lucid/model/TruncatedFourierFeatureMap.h"
#include "lucid/model/ValleePoussinKernel.h"
#include "lucid/util/Stats.h"
#include "lucid/util/constants.h"
#include "lucid/util/error.h"
#include "lucid/util/math.h"
#include "lucid/verification/AlglibOptimiser.h"
#include "lucid/verification/GurobiOptimiser.h"
#include "lucid/verification/HighsOptimiser.h"
#include "lucid/verification/SoplexOptimiser.h"

namespace lucid {

namespace {

class Objective {
 public:
  Objective(const int n_tilde, const double Q_tilde, const int f_max, const Matrix& lattice)
      : n_tilde_{n_tilde},
        lattice_{lattice},
        kernel_{ValleePoussinKernel{static_cast<double>(f_max), Q_tilde - static_cast<double>(f_max)}} {}

  /**
   * Wrap angle to [-Period/2, Period/2].
   * @tparam Period period of the angle
   * @param x angle in radians
   * @return wrapped angle in radians
   */
  template <double Period>
  static double wrap_angle(double x) {
    x = std::fmod(x + Period / 2, Period);
    if (x < 0) x += Period;
    return x - Period / 2;
  }

  /**
   * Objective function for PSO.
   * @f[
   * \frac{1}{\tilde{N}} \sum_{\bar{x} \in \Theta_{\tilde{N}} D^n_{f_{\max},\bar{Q}-f_{\max}}(x_i - \bar{x})
   * @f]
   * @tparam Derived type of the input matrix. Allows for Eigen expressions and improve performance
   * @param x current position of the particle, expected to be a column vector, where the function is evaluated
   * @return objective value at the given position `x`
   */
  template <class Derived>
  double operator()(const Eigen::MatrixBase<Derived>& x) const {
    LUCID_ASSERT(x.size() == lattice_.cols(), "The input dimension must be equal to the lattice dimension");
    const auto diffs = -1 * (lattice_.rowwise() - x.col(0).transpose());
    const auto wrapped_diffs = diffs.unaryExpr(&wrap_angle<2.0 * std::numbers::pi>);
    // std::cout << "wrapped_diffs " << LUCID_FORMAT_MATRIX(wrapped_diffs) << std::endl;
    const Matrix res = kernel_(wrapped_diffs);
    LUCID_ASSERT(res.cols() == 1 && res.rows() == lattice_.rows(),
                 "The output of the kernel must be a vector with the same number of rows as the input lattice");
    // We want to maximise the objective value, but PSO minimises it, so we return the negative value
    return -1 * res.sum() / static_cast<double>(n_tilde_);
  }

 private:
  const int n_tilde_;
  const Matrix& lattice_;
  const ValleePoussinKernel kernel_;
};

}  // namespace

std::pair<double, Vector> FourierBarrierCertificate::compute_A_periodic_minus_x(int Q_tilde, int f_max,
                                                                                const RectSet& X_tilde,
                                                                                const RectSet& X, const double increase,
                                                                                const PsoParameters& parameters) const {
  LUCID_TRACE_FMT("({}, {}, {}, {}, {}, {})", Q_tilde, f_max, X_tilde, X, increase, parameters);
  const int n = static_cast<int>(X_tilde.dimension());
  const int n_tilde = lucid::pow(Q_tilde, n);

  // TODO(tend): check if this is necessary
  static RectSet pi{Vector::Constant(n, 0), Vector::Constant(n, 2 * std::numbers::pi)};
  const RectSet X_periodic = (X - X_tilde.lower_bound()) * (2.0 * std::numbers::pi) / X_tilde.sizes();
  const RectSet X_periodic_rescaled{X_periodic.scale(increase, pi, true)};

  LUCID_TRACE_FMT("X_periodic: {}", X_periodic);
  LUCID_TRACE_FMT("X_periodic scaled: {}", X_periodic_rescaled);

  const Matrix lattice{pi.lattice(Q_tilde, false)};
  const Matrix lattice_wo_x{X_periodic_rescaled.exclude(lattice)};

  LUCID_TRACE_FMT("Original lattice size: {}, Lattice without X size: {}", lattice.rows(), lattice_wo_x.rows());

  pso::ParticleSwarmOptimization<double, Objective> optimiser{n_tilde, Q_tilde, f_max, lattice_wo_x};
  Matrix matrix_bounds{2, n};
  matrix_bounds.row(0) = X_periodic.lower_bound().transpose();
  matrix_bounds.row(1) = X_periodic.upper_bound().transpose();

  optimiser.setPhiParticles(parameters.phi_local);
  optimiser.setPhiGlobal(parameters.phi_global);
  optimiser.setInertiaWeightStrategy(pso::ConstantWeight<double>(parameters.weight));
  optimiser.setMaxIterations(parameters.max_iter);
  optimiser.setThreads(parameters.threads);
  optimiser.setMinParticleChange(parameters.xtol);
  optimiser.setMinFunctionChange(parameters.ftol);
  optimiser.setMaxVelocity(parameters.max_vel);
  optimiser.setVerbosity(LUCID_TRACE_ENABLED ? 3 : LUCID_DEBUG_ENABLED ? 1 : 0);

  auto [iterations, converged, fval, xval] = optimiser.minimize(matrix_bounds, parameters.num_particles);

  if (!converged) {
    LUCID_WARN("PSO did not converge");
    return {};
  }

  std::cout << "Computing A^{X_tilde - X} with PSO over " << n_tilde << " lattice points in " << n << " dimensions."
            << std::endl;

  LUCID_DEBUG_FMT("PSO converged in {} iterations with objective value {}", iterations, fval);
  LUCID_DEBUG_FMT("Best position: {}", LUCID_FORMAT_MATRIX(xval));
  return {-fval, xval.transpose()};
}

bool FourierBarrierCertificate::full_synthesize(const int Q_tilde, const TruncatedFourierFeatureMap& feature_map,
                                                const RectSet& X_bounds, const RectSet& X_init, const RectSet& X_unsafe,
                                                const FourierBarrierCertificateParameters& parameters) {
  const int n = static_cast<int>(X_bounds.dimension());
  const int f_max = feature_map.num_frequencies() - 1;

  const RectSet pi{Vector::Constant(n, 0), Vector::Constant(n, 2 * std::numbers::pi)};
  const RectSet X_tilde{feature_map.get_periodic_set()};

  return true;
}

bool FourierBarrierCertificate::synthesize(ConstMatrixRef fx_lattice, ConstMatrixRef fxp_lattice,
                                           ConstMatrixRef fx0_lattice, ConstMatrixRef fxu_lattice,
                                           const TruncatedFourierFeatureMap& feature_map,
                                           const Dimension num_frequency_samples_per_dim, const double C_coeff,
                                           const double epsilon, const double target_norm, const double b_kappa) {
  using DefaultOptimiser =
      std::conditional_t<constants::GUROBI_BUILD, GurobiOptimiser,                                        //
                         std::conditional_t<constants::HIGHS_BUILD, HighsOptimiser,                       //
                                            std::conditional_t<constants::ALGLIB_BUILD, AlglibOptimiser,  //
                                                               SoplexOptimiser>>>;
  return synthesize(DefaultOptimiser{}, fx_lattice, fxp_lattice, fx0_lattice, fxu_lattice, feature_map,
                    num_frequency_samples_per_dim, C_coeff, epsilon, target_norm, b_kappa);
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
      std::pow(1 - C_coeff * 2.0 * fraction, -static_cast<double>(feature_map.X_bounds().dimension()) / 2.0);
  const double fctr1 = 2 / (C + 1);
  const double fctr2 = (C - 1) / (C + 1);
  const double unsafe_rhs = fctr1 * gamma_;
  const double kushner_rhs = -fctr1 * epsilon * target_norm * std::abs(feature_map.sigma_f());

  LUCID_DEBUG_FMT("rkhs_dim: {}, num_variables: {}, num_constraints: {}", rkhs_dim, num_variables, num_constraints);
  LUCID_DEBUG_FMT("num_frequencies_per_dim: {}, num_frequency_samples_per_dim_periodic {}",
                  feature_map.num_frequencies(), num_frequency_samples_per_dim_periodic);
  LUCID_DEBUG_FMT("fraction: {}, C_coeff: {}, C: {}", fraction, C_coeff, C);
  LUCID_DEBUG_FMT("fctr1: {}, fctr2: {}, unsafe_rhs: {}, kushner_rhs: {}", fctr1, fctr2, unsafe_rhs, kushner_rhs);

  if (Stats::Scoped::top()) {
    Stats::Scoped::top()->value().num_variables = num_variables;
    Stats::Scoped::top()->value().num_constraints = num_constraints;
  }
  return optimiser.solve_fourier_barrier_synthesis(
      FourierBarrierSynthesisProblem{
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

std::ostream& operator<<(std::ostream& os, const FourierBarrierCertificateParameters& params) {
  return os << "PsoParameters( "
            << "increase( " << params.increase << " ) "
            << "num_particles( " << params.num_particles << " ) "
            << "phi_local( " << params.phi_local << " ) "
            << "phi_global( " << params.phi_global << " ) "
            << "weight( " << params.weight << " ) "
            << "max_iter( " << params.max_iter << " ) "
            << "max_vel( " << params.max_vel << " ) "
            << "ftol( " << params.ftol << " ) "
            << "xtol( " << params.xtol << " ) "
            << "threads( " << params.threads << " ) "
            << ")";
}
std::ostream& operator<<(std::ostream& os, const FourierBarrierCertificate& obj) {
  if (!obj.is_synthesized()) return os << "FourierBarrierCertificate( )";
  return os << "FourierBarrierCertificate( eta( " << obj.eta() << " ) gamma( " << obj.gamma() << " ) c( " << obj.c()
            << " ) norm( " << obj.norm() << " ) T( " << obj.T() << " ) safety( " << obj.safety() << " ) )";
}

}  // namespace lucid
