/**
 * @author c3054737
 * @copyright 2025 lucid
 * @licence BSD 3-Clause License
 * @file
 */
#include "lucid/verification/FourierBarrierCertificate.h"

#include <memory>
#include <numbers>

#include "lucid/lib/psocpp.h"
#include "lucid/model/Estimator.h"
#include "lucid/model/MultiSet.h"
#include "lucid/model/TruncatedFourierFeatureMap.h"
#include "lucid/model/ValleePoussinKernel.h"
#include "lucid/util/Stats.h"
#include "lucid/util/constants.h"
#include "lucid/util/error.h"
#include "lucid/util/math.h"
#include "lucid/util/random.h"
#include "lucid/verification/AlglibOptimiser.h"
#include "lucid/verification/GurobiOptimiser.h"
#include "lucid/verification/HighsOptimiser.h"
#include "lucid/verification/SoplexOptimiser.h"

namespace lucid {

namespace {

/** Objective functor for PSO optimisation of the Fourier barrier certificate. */
class Objective {
 public:
  /**
   * Construct the objective functor.
   * @param n_tilde number of lattice points
   * @param Q_tilde total number of frequencies
   * @param f_max maximum frequency
   * @param lattice lattice points
   */
  Objective(const int n_tilde, const double Q_tilde, const int f_max, const ConstMatrixRowIndexedView& lattice)
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
    // x - bar{x}
    const auto diffs = -1 * (lattice_.rowwise() - x.col(0).transpose());
    // (x - bar{x}) wrapped between [-pi, pi]
    const auto wrapped_diffs = diffs.unaryExpr(&wrap_angle<2.0 * std::numbers::pi>);
    // Apply the Vallée-Poussin kernel to each row of the wrapped differences
    const Matrix res = kernel_(wrapped_diffs);
    LUCID_ASSERT(res.cols() == 1 && res.rows() == lattice_.rows(),
                 "The output of the kernel must be a vector with the same number of rows as the input lattice");
    // We want to maximise the objective value, but PSO minimises it, so we return the negative value
    return -1 * res.sum() / static_cast<double>(n_tilde_);
  }

 private:
  const int n_tilde_;                         ///< Number of lattice points
  const ConstMatrixRowIndexedView& lattice_;  ///< Lattice points
  const ValleePoussinKernel kernel_;          ///< Vallée-Poussin kernel
};

double compute_A_periodic(int Q_tilde, int f_max, const RectSet& pi, const RectSet& X_tilde, const RectSet& X,
                          const Matrix& lattice, const FourierBarrierCertificateParameters& parameters) {
  LUCID_TRACE_FMT("({}, {}, {}, {}, {})", Q_tilde, f_max, X_tilde, X, parameters);
  const int n = static_cast<int>(X_tilde.dimension());
  const int n_tilde = lucid::pow(Q_tilde, n);

  // TODO(tend): check if this is necessary
  // We map the original set X (which corresponds to X_bounds, X_init or X_unsafe)
  // to the corresponding set in the periodic space [0, 2pi]^n
  const RectSet X_periodic = (X - X_tilde.lower_bound()) * (2.0 * std::numbers::pi) / X_tilde.sizes();
  // Rescale the newly created periodic set by the given increase factor, making sure it stays within [0, 2pi]^n
  const std::unique_ptr<Set> X_periodic_rescaled{X_periodic.scale_wrapped(parameters.increase, pi, true)};

  LUCID_TRACE_FMT("X_periodic: {}", X_periodic);
  LUCID_TRACE_FMT("X_periodic scaled: {}", *X_periodic_rescaled);

  // Only keep the lattice points that are not in X_periodic_rescaled
  auto lattice_wo_x{lattice(X_periodic_rescaled->exclude_mask(lattice), Eigen::placeholders::all)};
  LUCID_CHECK_ARGUMENT_CMP(lattice_wo_x.rows(), >, 0);  // Ensure there are points to evaluate

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
  optimiser.setGen(random::gen);
  optimiser.setVerbosity(LUCID_TRACE_ENABLED ? 3 : LUCID_DEBUG_ENABLED ? 1 : 0);

  auto [iterations, converged, fval, xval] = optimiser.minimize(matrix_bounds, parameters.num_particles);

  if (!converged) {
    LUCID_WARN("PSO did not converge");
    return 0;
  }

  LUCID_DEBUG_FMT("PSO converged in {} iterations with objective value {}", iterations, fval);
  LUCID_DEBUG_FMT("Best position: {}", LUCID_FORMAT_MATRIX(xval));
  return -fval;
}

double compute_A_periodic(const int Q_tilde, const int f_max, const RectSet& pi, const RectSet& X_tilde, const Set& X,
                          const Matrix& lattice, const FourierBarrierCertificateParameters& parameters) {
  if (const auto* X_rect = dynamic_cast<const RectSet*>(&X)) {
    return compute_A_periodic(Q_tilde, f_max, pi, X_tilde, *X_rect, lattice, parameters);
  }

  // TODO(tend): this is a manual way of supporting MultiSet for X. There may be a more direct way.
  if (const auto* X_multi = dynamic_cast<const MultiSet*>(&X)) {
    std::vector<double> As;
    As.reserve(X_multi->sets().size());
    for (const auto& subset : static_cast<const MultiSet&>(X).sets()) {
      auto subset_to_rect = subset->to_rect_set();
      const RectSet* subset_rect = dynamic_cast<const RectSet*>(subset_to_rect.get());
      LUCID_CHECK_ARGUMENT(subset_rect != nullptr, "multi-set", "nested multi-sets are not supported");
      As.push_back(compute_A_periodic(Q_tilde, f_max, pi, X_tilde, *subset_rect, lattice, parameters));
    }
    return *std::ranges::max_element(As);
  }

  LUCID_UNREACHABLE();
}

}  // namespace

bool FourierBarrierCertificate::synthesize(const int Q_tilde, const Estimator& estimator,
                                           const TruncatedFourierFeatureMap& feature_map, const RectSet& X_bounds,
                                           const Set& X_init, const Set& X_unsafe,
                                           const FourierBarrierCertificateParameters& parameters) {
  using DefaultOptimiser =
      std::conditional_t<constants::GUROBI_BUILD, GurobiOptimiser,                                        //
                         std::conditional_t<constants::HIGHS_BUILD, HighsOptimiser,                       //
                                            std::conditional_t<constants::ALGLIB_BUILD, AlglibOptimiser,  //
                                                               SoplexOptimiser>>>;
  return synthesize(DefaultOptimiser{}, Q_tilde, estimator, feature_map, X_bounds, X_init, X_unsafe, parameters);
}

bool FourierBarrierCertificate::synthesize(const Optimiser& optimiser, const int Q_tilde, const Estimator& estimator,
                                           const TruncatedFourierFeatureMap& feature_map, const RectSet& X_bounds,
                                           const Set& X_init, const Set& X_unsafe,
                                           const FourierBarrierCertificateParameters& parameters) {
  const int n = static_cast<int>(X_bounds.dimension());
  const int f_max = feature_map.num_frequencies() - 1;
  LUCID_CHECK_ARGUMENT_CMP(f_max, <=, 2 * Q_tilde + 1);

  // Base periodic set between [0, 2pi] in each dimension
  const RectSet pi{Vector::Constant(n, 0), Vector::Constant(n, 2 * std::numbers::pi)};
  // Find the periodic set encapsulating the original set
  const RectSet X_tilde{feature_map.get_periodic_set()};
  // Create a lattice over the periodic set with no endpoints (since they would wrap around)
  const Matrix pi_lattice{pi.lattice(Q_tilde, false)};

  LUCID_ASSERT((X_tilde.lower_bound().array() <= X_bounds.lower_bound().array()).all() &&
                   (X_bounds.upper_bound().array() <= X_tilde.upper_bound().array()).all(),
               "X_bounds must be contained in X_tilde");

  const double A_x = compute_A_periodic(Q_tilde, f_max, pi, X_tilde, X_bounds, pi_lattice, parameters);
  const double A_x0 = compute_A_periodic(Q_tilde, f_max, pi, X_tilde, *X_init.to_rect_set(), pi_lattice, parameters);
  const double A_xu = compute_A_periodic(Q_tilde, f_max, pi, X_tilde, *X_unsafe.to_rect_set(), pi_lattice, parameters);

  LUCID_INFO_FMT("A^{{X_tilde \\ X}}_{{N_tilde}}: {}", A_x);
  LUCID_INFO_FMT("A^{{X_tilde \\ X_init}}_{{N_tilde}}: {}", A_x0);
  LUCID_INFO_FMT("A^{{X_tilde \\ X_unsafe}}_{{N_tilde}}: {}", A_xu);

  LUCID_CHECK_ARGUMENT_CMP(A_x, >, 0);
  LUCID_CHECK_ARGUMENT_CMP(A_x0, >, 0);
  LUCID_CHECK_ARGUMENT_CMP(A_xu, >, 0);

  // Apply the feature map to all the lattice points
  const Matrix lattice = X_tilde.lattice(Q_tilde, false);
  const Matrix f_lattice{feature_map(lattice)};
  Matrix fp_lattice{estimator(lattice)};
  // We are fixing the zero frequency to the constant value we computed in the feature map
  // The regressor has a hard time learning it on the extreme left and right points, because it tends to 0
  fp_lattice.col(0) = Vector::Constant(fp_lattice.rows(), feature_map.weights()[0] * feature_map.sigma_f());
  const Matrix dn_lattice = f_lattice - fp_lattice;

  LUCID_DEBUG_FMT("X_tilde: {}", X_tilde);
  LUCID_DEBUG_FMT("X_bounds: {}", X_bounds);
  LUCID_DEBUG_FMT("X_init: {}", X_init);
  LUCID_DEBUG_FMT("X_unsafe: {}", X_unsafe);

  const std::unique_ptr<Set> X_bounds_rescaled{X_bounds.scale_wrapped(parameters.increase, X_tilde, true)};
  const std::unique_ptr<Set> X_init_rescaled{X_init.to_rect_set()->scale_wrapped(parameters.increase, X_tilde, true)};
  const std::unique_ptr<Set> X_unsafe_rescaled{
      X_unsafe.to_rect_set()->scale_wrapped(parameters.increase, X_tilde, true)};

  LUCID_DEBUG_FMT("X_bounds rescaled: {}", *X_bounds_rescaled);
  LUCID_DEBUG_FMT("X_init rescaled: {}", *X_init_rescaled);
  LUCID_DEBUG_FMT("X_unsafe rescaled: {}", *X_unsafe_rescaled);

  // Only keep the points inside/outside the sets
  const auto [x_include_mask, x_exclude_mask] = X_bounds_rescaled->include_exclude_masks(lattice);
  const auto [x0_include_mask, x0_exclude_mask] = X_init_rescaled->include_exclude_masks(lattice);
  const auto [xu_include_mask, xu_exclude_mask] = X_unsafe_rescaled->include_exclude_masks(lattice);
  LUCID_CHECK_ARGUMENT_CMP(x_include_mask.size(), >, 0);
  LUCID_CHECK_ARGUMENT_CMP(x_exclude_mask.size(), >, 0);
  LUCID_CHECK_ARGUMENT_CMP(x0_include_mask.size(), >, 0);
  LUCID_CHECK_ARGUMENT_CMP(x0_exclude_mask.size(), >, 0);
  LUCID_CHECK_ARGUMENT_CMP(xu_include_mask.size(), >, 0);
  LUCID_CHECK_ARGUMENT_CMP(xu_exclude_mask.size(), >, 0);

  const double C = std::pow(1 - parameters.C_coeff * 2.0 * f_max / static_cast<double>(Q_tilde), -n / 2.0);
  LUCID_DEBUG_FMT("C = (1 - (2 f_max) / Q_tilde)^(-n/2) = (1 - {:.3f} * 2.0 * {} / {})^(-{}/2) = {:.3}",
                  parameters.C_coeff, f_max, Q_tilde, n, C);
  const double eta_coeff = 2.0 / (C - A_x0 + 1.0);
  LUCID_DEBUG_FMT("eta_coeff: 2 / (C - A_x0 + 1) = 2 / ({:.3} - {:.3} + 1) = {:.3}", C, A_x0, eta_coeff);
  const double min_x0_coeff = (C - A_x0 - 1.0) / (C - A_x0 + 1.0);
  LUCID_DEBUG_FMT("min_x0_coeff: (C - A_x0 - 1) / (C - A_x0 + 1) = ({:.3} - {:.3} - 1) / ({:.3} - {:.3} + 1) = {:.3}",
                  C, A_x0, C, A_x0, min_x0_coeff);
  const double diff_sx0_coeff = A_x0 / (C - A_x0 + 1.0);
  LUCID_DEBUG_FMT("diff_sx0_coeff: A_x0 / (C - A_x0 + 1) = {:.3} / ({:.3} - {:.3} + 1) = {:.3}", A_x0, C, A_x0,
                  diff_sx0_coeff);
  const double gamma_coeff = 2.0 / (C - A_xu + 1.0);
  LUCID_DEBUG_FMT("gamma_coeff: 2 / (C - A_xu + 1) = 2 / ({:.3} - {:.3} + 1) = {:.3}", C, A_xu, gamma_coeff);
  const double max_xu_coeff = (C - A_xu - 1) / (C - A_xu + 1.0);
  LUCID_DEBUG_FMT("max_xu_coeff: (C - A_xu - 1) / (C - A_xu + 1) = ({:.3} - {:.3} - 1) / ({:.3} - {:.3} + 1) = {:.3}",
                  C, A_xu, C, A_xu, max_xu_coeff);
  const double diff_sxu_coeff = A_xu / (C - A_xu + 1.0);
  LUCID_DEBUG_FMT("diff_sxu_coeff: A_xu / (C - A_xu + 1) = {:.3} / ({:.3} - {:.3} + 1) = {:.3}", A_xu, C, A_xu,
                  diff_sxu_coeff);
  const double ebk = parameters.epsilon * parameters.target_norm * parameters.kappa;
  LUCID_DEBUG_FMT("ebk: epsilon * target_norm * kapp = {:.3} * {:.3} * {:.3} = {:.3}", parameters.epsilon,
                  parameters.target_norm, parameters.kappa, ebk);
  const double c_ebk_coeff = 2.0 / (C - A_x + 1.0);
  LUCID_DEBUG_FMT("c_ebk_coeff: 2 / (C - A_x + 1) = 2 / ({:.3} - {:.3} + 1) = {:.3}", C, A_x, c_ebk_coeff);
  const double min_d_coeff = (C - A_x - 1.0) / (C - A_x + 1.0);
  LUCID_DEBUG_FMT("min_d_coeff: (C - A_x - 1) / (C - A_x + 1) = ({:.3} - {:.3} - 1) / ({:.3} - {:.3} + 1) = {:.3}", C,
                  A_x, C, A_x, min_d_coeff);
  const double diff_d_sx_coeff = A_x / (C - A_x + 1.0);
  LUCID_DEBUG_FMT("diff_d_sx_coeff: A_x / (C - A_x + 1) = {:.3} / ({:.3} - {:.3} + 1) = {:.3}", A_x, C, A_x,
                  diff_d_sx_coeff);
  const double max_x_coeff = (C - A_x - 1.0) / (C - A_x + 1.0);
  LUCID_DEBUG_FMT("max_x_coeff: (C - A_x - 1) / (C - A_x + 1) = ({:.3} - {:.3} - 1) / ({:.3} - {:.3} + 1) = {:.3}", C,
                  A_x, C, A_x, max_x_coeff);
  const double diff_sx_coeff = A_x / (C - A_x + 1.0);
  LUCID_DEBUG_FMT("diff_sx_coeff: A_x / (C - A_x + 1) = {:.3} / ({:.3} - {:.3} + 1) = {:.3}", A_x, C, A_x,
                  diff_sx_coeff);

  return optimiser.solve_fourier_barrier_synthesis(
      FourierBarrierSynthesisProblem{.num_vars = -1,
                                     .num_constraints = -1,
                                     .fxn_lattice = f_lattice,
                                     .dn_lattice = dn_lattice,
                                     .x_include_mask = x_include_mask,
                                     .x_exclude_mask = x_exclude_mask,
                                     .x0_include_mask = x0_include_mask,
                                     .x0_exclude_mask = x0_exclude_mask,
                                     .xu_include_mask = xu_include_mask,
                                     .xu_exclude_mask = xu_exclude_mask,
                                     .T = T_,
                                     .gamma = gamma_,
                                     .C = C,
                                     .b_kappa = parameters.kappa,
                                     .eta_coeff = eta_coeff,
                                     .min_x0_coeff = min_x0_coeff,
                                     .diff_sx0_coeff = diff_sx0_coeff,
                                     .gamma_coeff = gamma_coeff,
                                     .max_xu_coeff = max_xu_coeff,
                                     .diff_sxu_coeff = diff_sxu_coeff,
                                     .ebk = ebk,
                                     .c_ebk_coeff = c_ebk_coeff,
                                     .min_d_coeff = min_d_coeff,
                                     .diff_d_sx_coeff = diff_d_sx_coeff,
                                     .max_x_coeff = max_x_coeff,
                                     .diff_sx_coeff = diff_sx_coeff,
                                     .fctr1 = .0,
                                     .fctr2 = .0,
                                     .unsafe_rhs = .0,
                                     .kushner_rhs = .0,
                                     .A_x = A_x,
                                     .A_x0 = A_x0,
                                     .A_xu = A_xu},
      std::bind(&FourierBarrierCertificate::optimiser_callback, this, std::placeholders::_1, std::placeholders::_2,
                std::placeholders::_3, std::placeholders::_4, std::placeholders::_5, std::placeholders::_6,
                parameters.target_norm));
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
      FourierBarrierSynthesisProblem{.num_vars = num_variables,
                                     .num_constraints = num_constraints,
                                     .fxn_lattice = Matrix{},
                                     .dn_lattice = Matrix{},
                                     .x_include_mask = {},
                                     .x_exclude_mask = {},
                                     .x0_include_mask = {},
                                     .x0_exclude_mask = {},
                                     .xu_include_mask = {},
                                     .xu_exclude_mask = {},
                                     .T = T_,
                                     .gamma = gamma_,
                                     .C = C,
                                     .b_kappa = b_kappa,
                                     .eta_coeff = 0.0,
                                     .min_x0_coeff = .0,
                                     .diff_sx0_coeff = .0,
                                     .gamma_coeff = .0,
                                     .max_xu_coeff = .0,
                                     .diff_sxu_coeff = .0,
                                     .ebk = .0,
                                     .c_ebk_coeff = .0,
                                     .min_d_coeff = .0,
                                     .diff_d_sx_coeff = .0,
                                     .max_x_coeff = .0,
                                     .diff_sx_coeff = .0,
                                     .fctr1 = fctr1,
                                     .fctr2 = fctr2,
                                     .unsafe_rhs = unsafe_rhs,
                                     .kushner_rhs = kushner_rhs,
                                     .A_x = 0,
                                     .A_x0 = 0,
                                     .A_xu = 0},
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
  return os << "FourierBarrierCertificateParameters( "
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
