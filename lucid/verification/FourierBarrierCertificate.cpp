/**
 * @author c3054737
 * @copyright 2025 lucid
 * @licence BSD 3-Clause License
 * @file
 */
#include "lucid/verification/FourierBarrierCertificate.h"

#include <memory>
#include <numbers>
#include <string>
#include <utility>
#include <vector>

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

#ifdef LUCID_PYTHON_BUILD
#include "bindings/pylucid/interrupt.h"
#endif

namespace lucid {

namespace {

/** Objective functor for PSO optimisation of the Fourier barrier certificate. */
class Objective {
 public:
  /**
   * Construct the objective functor.
   * @param n_tilde number of lattice points
   * @param lattice_resolution total number of frequencies
   * @param f_max maximum frequency
   * @param lattice lattice points
   */
  Objective(const int n_tilde, const double lattice_resolution, const int f_max,
            const ConstMatrixRowIndexedView& lattice)
      : n_tilde_{n_tilde},
        lattice_{lattice},
        kernel_{ValleePoussinKernel{static_cast<double>(f_max), lattice_resolution - static_cast<double>(f_max)}} {}

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

#ifdef LUCID_PYTHON_BUILD
class PyCallback {
 public:
  using PsoIndex = pso::Index;
  using PsoMatrix = pso::NoCallback<Scalar>::Matrix;
  using PsoVector = pso::NoCallback<Scalar>::Vector;
  bool operator()(const PsoIndex, const PsoMatrix&, const PsoVector&, const PsoIndex) const {
    py_check_signals();
    return true;
  }
};
using PsoOptimiser = pso::ParticleSwarmOptimization<double, Objective, pso::ConstantWeight<Scalar>, PyCallback>;
#else
using PsoOptimiser = pso::ParticleSwarmOptimization<double, Objective>;
#endif

double run_pso(int lattice_resolution, int f_max, const RectSet& X_periodic,
               const ConstMatrixRowIndexedView& filtered_lattice,
               const FourierBarrierCertificateParameters& parameters) {
  const int d = static_cast<int>(X_periodic.dimension());
  const int n_tilde = lucid::pow(lattice_resolution, d);

  if (filtered_lattice.rows() == 0) {
    LUCID_WARN("No lattice points to evaluate in PSO");
    return 0;
  }

  PsoOptimiser optimiser{n_tilde, lattice_resolution, f_max, filtered_lattice};
  // Bounds of the optimization. They ensure the particles stay within the X_periodic set
  Matrix matrix_bounds{2, d};
  matrix_bounds.row(0) = X_periodic.lower_bound().transpose();
  matrix_bounds.row(1) = X_periodic.upper_bound().transpose();
  // Configure optimiser
  optimiser.setPhiParticles(parameters.phi_local);
  optimiser.setPhiGlobal(parameters.phi_global);
  optimiser.setInertiaWeightStrategy(pso::ConstantWeight<double>(parameters.weight));
  optimiser.setMaxIterations(parameters.max_iter);
  optimiser.setThreads(parameters.threads);
  optimiser.setMinParticleChange(parameters.xtol);
  optimiser.setMinFunctionChange(parameters.ftol);
  optimiser.setMaxVelocity(parameters.max_vel);
  optimiser.setGen(random::gen);
  optimiser.setVerbosity(LUCID_TRACE_ENABLED ? 3 : 0);

  auto [iterations, converged, fval, xval] = optimiser.minimize(matrix_bounds, parameters.num_particles);

  if (!converged) {
    LUCID_ERROR("PSO did not converge");
    return 0;
  }

  if (fval < -0.5) LUCID_WARN_FMT("PSO value is large: {}", -fval);

  LUCID_DEBUG_FMT("PSO converged in {} iterations with objective value {}", iterations, fval);
  LUCID_DEBUG_FMT("Best position: {}", LUCID_FORMAT_MATRIX(xval));
  return -fval;
}

double compute_A_impl(int lattice_resolution, int f_max, const RectSet& pi, const RectSet& X_tilde, const RectSet& X,
                      const Matrix& lattice, const FourierBarrierCertificateParameters& parameters) {
  LUCID_TRACE_FMT("({}, {}, {}, {}, {})", lattice_resolution, f_max, X_tilde, X, parameters);

  // We map the original set X (which corresponds to X_bounds, X_init or X_unsafe)
  // to the corresponding set in the periodic space [0, 2pi]^n
  const RectSet X_periodic = (X - X_tilde.lower_bound()) * (2.0 * std::numbers::pi) / X_tilde.sizes();
  // Rescale the newly created periodic set by the given factor, making sure it stays within [0, 2pi]^n
  const std::unique_ptr<Set> X_periodic_rescaled{X_periodic.scale_wrapped(parameters.set_scaling, pi, true)};

  LUCID_TRACE_FMT("X_periodic: {}", X_periodic);
  LUCID_TRACE_FMT("X_periodic scaled: {}", *X_periodic_rescaled);

  // Only keep the lattice points that are not in X_periodic_rescaled
  const auto lattice_wo_x{lattice(X_periodic_rescaled->exclude_mask(lattice), Eigen::placeholders::all)};

  LUCID_TRACE_FMT("Original lattice size: {}, Lattice without X size: {}", lattice.rows(), lattice_wo_x.rows());

  return run_pso(lattice_resolution, f_max, X_periodic, lattice_wo_x, parameters);
}

double compute_A_impl(const int lattice_resolution, const int f_max, const RectSet& pi, const RectSet& X_tilde,
                      const MultiSet& X, const Matrix& lattice, const FourierBarrierCertificateParameters& parameters) {
  std::vector<double> As;
  As.reserve(X.sets().size());
  std::vector<std::unique_ptr<Set>> pi_sets;
  std::vector<std::unique_ptr<Set>> rescaled_pi_sets;
  for (const auto& subset : X.sets()) {
    auto subset_to_rect = subset->to_rect_set();
    const RectSet& subset_rect = *dynamic_cast<const RectSet*>(subset_to_rect.get());

    // We map the original set X (which corresponds to X_bounds, X_init or X_unsafe)
    // to the corresponding set in the periodic space [0, 2pi]^n
    pi_sets.emplace_back(
        std::make_unique<RectSet>((subset_rect - X_tilde.lower_bound()) * (2.0 * std::numbers::pi) / X_tilde.sizes()));
    // Rescale the newly created periodic set by the given factor, making sure it stays within [0, 2pi]^n
    rescaled_pi_sets.emplace_back(pi_sets.back()->scale_wrapped(parameters.set_scaling, pi, true));
  }
  const auto filtered_lattice =
      lattice(MultiSet(std::move(rescaled_pi_sets)).exclude_mask(lattice), Eigen::placeholders::all);
  for (const auto& rescaled_set : pi_sets) {
    As.push_back(
        run_pso(lattice_resolution, f_max, static_cast<RectSet&>(*rescaled_set), filtered_lattice, parameters));
  }
  return *std::ranges::max_element(As);
}

}  // namespace

std::string FourierBarrierCertificateParameters::to_string() const {
  return fmt::format(
      "FourierBarrierCertificateParameters( set_scaling( {} ) max_iter( {} ) num_particles( {} ) threads( {} ) "
      "weight( {} ) phi_local( {} ) phi_global( {} ) xtol( {} ) ftol( {} ) max_vel( {} ) )",
      set_scaling, max_iter, num_particles, threads, weight, phi_local, phi_global, xtol, ftol, max_vel);
}
double FourierBarrierCertificate::compute_A(const int lattice_resolution, const int f_max, const RectSet& pi,
                                            const RectSet& X_tilde, const Set& X, const Matrix& lattice,
                                            const FourierBarrierCertificateParameters& parameters) {
  if (const auto* X_rect = dynamic_cast<const RectSet*>(&X)) {
    return compute_A_impl(lattice_resolution, f_max, pi, X_tilde, *X_rect, lattice, parameters);
  }

  // TODO(tend): this is a manual way of supporting MultiSet for X. There may be a more direct way.
  if (const auto* X_multi = dynamic_cast<const MultiSet*>(&X)) {
    return compute_A_impl(lattice_resolution, f_max, pi, X_tilde, *X_multi, lattice, parameters);
  }

  LUCID_UNREACHABLE();
}
double FourierBarrierCertificate::compute_A(const int lattice_resolution, const int f_max, const RectSet& X_tilde,
                                            const Set& X, const FourierBarrierCertificateParameters& parameters) {
  const RectSet pi{Vector::Constant(X_tilde.dimension(), 0),
                   Vector::Constant(X_tilde.dimension(), 2 * std::numbers::pi)};
  return compute_A(lattice_resolution, f_max, pi, X_tilde, X, pi.lattice(lattice_resolution, false), parameters);
}

bool FourierBarrierCertificate::synthesize(const int lattice_resolution, const Estimator& estimator,
                                           const TruncatedFourierFeatureMap& feature_map, const RectSet& X_bounds,
                                           const Set& X_init, const Set& X_unsafe,
                                           const FourierBarrierCertificateParameters& parameters) {
  using DefaultOptimiser =
      std::conditional_t<constants::GUROBI_BUILD, GurobiOptimiser,                                        //
                         std::conditional_t<constants::HIGHS_BUILD, HighsOptimiser,                       //
                                            std::conditional_t<constants::ALGLIB_BUILD, AlglibOptimiser,  //
                                                               SoplexOptimiser>>>;
  return synthesize(DefaultOptimiser{}, lattice_resolution, estimator, feature_map, X_bounds, X_init, X_unsafe,
                    parameters);
}

bool FourierBarrierCertificate::synthesize(const Optimiser& optimiser, const int lattice_resolution,
                                           const Estimator& estimator, const TruncatedFourierFeatureMap& feature_map,
                                           const RectSet& X_bounds, const Set& X_init, const Set& X_unsafe,
                                           const FourierBarrierCertificateParameters& parameters) {
  LUCID_TRACE_FMT("({}, {}, {}, {}, {}, {}, {})", optimiser, lattice_resolution, feature_map, X_bounds, X_init,
                  X_unsafe, parameters);
  const int n = static_cast<int>(X_bounds.dimension());
  const int f_max = feature_map.num_frequencies() - 1;
  LUCID_CHECK_ARGUMENT_CMP(f_max, <=, 2 * lattice_resolution + 1);

  // Base periodic set between [0, 2pi] in each dimension
  const RectSet pi{Vector::Constant(n, 0), Vector::Constant(n, 2 * std::numbers::pi)};
  // Find the periodic set encapsulating the original set
  const RectSet X_tilde{feature_map.get_periodic_set()};
  // Create a lattice over the periodic set with no endpoints (since they would wrap around)
  const Matrix pi_lattice{pi.lattice(lattice_resolution, false)};

  LUCID_ASSERT((X_tilde.lower_bound().array() <= X_bounds.lower_bound().array()).all() &&
                   (X_bounds.upper_bound().array() <= X_tilde.upper_bound().array()).all(),
               "X_bounds must be contained in X_tilde");

  const double A_x = compute_A(lattice_resolution, f_max, pi, X_tilde, X_bounds, pi_lattice, parameters);
  const double A_x0 = compute_A(lattice_resolution, f_max, pi, X_tilde, *X_init.to_rect_set(), pi_lattice, parameters);
  const double A_xu =
      compute_A(lattice_resolution, f_max, pi, X_tilde, *X_unsafe.to_rect_set(), pi_lattice, parameters);

  LUCID_DEBUG_FMT("A_x: {}", A_x);
  LUCID_DEBUG_FMT("A_x0: {}", A_x0);
  LUCID_DEBUG_FMT("A_xu: {}", A_xu);

  // Apply the feature map to all the lattice points
  const Matrix lattice = X_tilde.lattice(lattice_resolution, false);
  const Matrix f_lattice{feature_map(lattice)};
  Matrix fp_lattice{estimator(lattice)};
  // We are fixing the zero frequency to the constant value we computed in the feature map
  // The regressor has a hard time learning it on the extreme left and right points, because it tends to 0
  fp_lattice.col(0) = Vector::Constant(fp_lattice.rows(), feature_map.weights()[0] * feature_map.sigma_f());
  const Matrix dn_lattice = fp_lattice - f_lattice;

  LUCID_DEBUG_FMT("X_tilde: {}", X_tilde);

  const std::unique_ptr<Set> X_bounds_rescaled{X_bounds.scale_wrapped(parameters.set_scaling, X_tilde, true)};
  const std::unique_ptr<Set> X_init_rescaled{
      X_init.to_rect_set()->scale_wrapped(parameters.set_scaling, X_tilde, true)};
  const std::unique_ptr<Set> X_unsafe_rescaled{
      X_unsafe.to_rect_set()->scale_wrapped(parameters.set_scaling, X_tilde, true)};

  LUCID_DEBUG_FMT("X_bounds rescaled: {}", *X_bounds_rescaled);
  LUCID_DEBUG_FMT("X_init rescaled: {}", *X_init_rescaled);
  LUCID_DEBUG_FMT("X_unsafe rescaled: {}", *X_unsafe_rescaled);

  // Only keep the points inside/outside the sets
  const auto [x_include_mask, x_exclude_mask] = X_bounds_rescaled->include_exclude_masks(lattice);
  const auto [x0_include_mask, x0_exclude_mask] = X_init_rescaled->include_exclude_masks(lattice);
  const auto [xu_include_mask, xu_exclude_mask] = X_unsafe_rescaled->include_exclude_masks(lattice);

  const double C = std::pow(1 - parameters.C_coeff * 2.0 * f_max / static_cast<double>(lattice_resolution), -n / 2.0);
  LUCID_DEBUG_FMT("C = (1 - (2 f_max) / lattice_resolution)^(-n/2) = (1 - {:.3f} * 2.0 * {} / {})^(-{}/2) = {:.3}",
                  parameters.C_coeff, f_max, lattice_resolution, n, C);

  const double x0_denom = C - 2.0 * A_x0 + 1.0;
  LUCID_DEBUG_FMT("x0_denom: C - 2 * A_x0 + 1 = {:.3} - 2 * {:.3} + 1 = {:.3}", C, A_x0, x0_denom);
  LUCID_ASSERT(x0_denom != 0.0, "Denominator for sx0_coeff cannot be zero");
  const double eta_coeff = 2.0 / x0_denom;
  LUCID_DEBUG_FMT("eta_coeff: 2 / x0_denom = 2 / {:.3} = {:.3}", x0_denom, eta_coeff);
  const double min_x0_coeff = (C - 1.0) / x0_denom;
  LUCID_DEBUG_FMT("min_x0_coeff: (C - 1) / x0_denom = ({:.3} - 1) / {:.3} = {:.3}", C, x0_denom, min_x0_coeff);
  const double diff_sx0_coeff = 2.0 * A_x0 / x0_denom;
  LUCID_DEBUG_FMT("diff_sx0_coeff: 2 * A_x0 / x0_denom = 2 * {:.3} / {:.3} = {:.3}", A_x0, x0_denom, diff_sx0_coeff);

  const double xu_denom = C - 2.0 * A_xu + 1.0;
  LUCID_DEBUG_FMT("xu_denom: C - 2 * A_xu + 1 = {:.3} - 2 * {:.3} + 1 = {:.3}", C, A_xu, xu_denom);
  LUCID_ASSERT(xu_denom != 0.0, "Denominator for gamma_coeff cannot be zero");
  const double gamma_coeff = 2.0 / xu_denom;
  LUCID_DEBUG_FMT("gamma_coeff: 2 / xu_denom = 2 / {:.3} = {:.3}", xu_denom, gamma_coeff);
  const double max_xu_coeff = (C - 1.0) / xu_denom;
  LUCID_DEBUG_FMT("max_xu_coeff: (C - 1) / xu_denom = ({:.3} - 1) / {:.3} = {:.3}", C, xu_denom, max_xu_coeff);
  const double diff_sxu_coeff = 2.0 * A_xu / xu_denom;
  LUCID_DEBUG_FMT("diff_sxu_coeff: 2 * A_xu / xu_denom = 2 * {:.3} / {:.3} = {:.3}", A_xu, xu_denom, diff_sxu_coeff);

  const double ebk = parameters.epsilon * parameters.b_norm * parameters.kappa;
  LUCID_DEBUG_FMT("ebk: epsilon * b_norm * kappa = {:.3} * {:.3} * {:.3} = {:.3}", parameters.epsilon,
                  parameters.b_norm, parameters.kappa, ebk);
  const double d_denom = C - 2.0 * A_x + 1.0;
  LUCID_DEBUG_FMT("d_denom: C - 2 * A_x + 1 = {:.3} - 2 * {:.3} + 1 = {:.3}", C, A_x, d_denom);
  LUCID_ASSERT(d_denom != 0.0, "Denominator for d_coeff cannot be zero");
  const double c_ebk_coeff = 2.0 / d_denom;
  LUCID_DEBUG_FMT("c_ebk_coeff: 2 / d_denom = 2 / {:.3} = {:.3}", d_denom, c_ebk_coeff);
  const double min_d_coeff = (C - 1.0) / d_denom;
  LUCID_DEBUG_FMT("min_d_coeff: (C - 1) / d_denom = ({:.3} - 1) / {:.3} = {:.3}", C, d_denom, min_d_coeff);
  const double diff_d_sx_coeff = 2.0 * A_x / d_denom;
  LUCID_DEBUG_FMT("diff_d_sx_coeff: 2 * A_x / d_denom = 2 * {:.3} / {:.3} = {:.3}", A_x, d_denom, diff_d_sx_coeff);

  const double x_denom = C - 2.0 * A_x + 1.0;
  LUCID_DEBUG_FMT("x_denom: C - 2 * A_x + 1 = {:.3} - 2 * {:.3} + 1 = {:.3}", C, A_x, x_denom);
  LUCID_ASSERT(x_denom != 0.0, "Denominator for x_coeff cannot be zero");
  const double max_x_coeff = (C - 1.0) / x_denom;
  LUCID_DEBUG_FMT("max_x_coeff: (C - 1) / x_denom = ({:.3} - 1) / {:.3} = {:.3}", C, x_denom, max_x_coeff);
  const double diff_sx_coeff = 2.0 * A_x / x_denom;
  LUCID_DEBUG_FMT("diff_sx_coeff: 2 * A_x / x_denom = 2 * {:.3} / {:.3} = {:.3}", A_x, x_denom, diff_sx_coeff);

  return optimiser.solve_fourier_barrier_synthesis(
      FourierBarrierSynthesisProblem{
          .num_constraints =
              static_cast<int>(1 + x0_include_mask.size() * 2 + xu_include_mask.size() * 2 + x_include_mask.size() * 4 +
                               x0_exclude_mask.size() + xu_exclude_mask.size() + x_exclude_mask.size() * 2),
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
      },
      std::bind(&FourierBarrierCertificate::optimiser_callback, this, std::placeholders::_1, std::placeholders::_2,
                std::placeholders::_3, std::placeholders::_4, std::placeholders::_5, std::placeholders::_6,
                parameters.b_norm));
}

std::unique_ptr<BarrierCertificate> FourierBarrierCertificate::clone() const {
  return std::make_unique<FourierBarrierCertificate>(*this);
}

std::string FourierBarrierCertificate::to_string() const {
  if (!is_synthesized()) return "FourierBarrierCertificate( )";
  return fmt::format("FourierBarrierCertificate( eta( {} ) gamma( {} ) c( {} ) norm( {} ) T( {} ) safety( {} ) )", eta_,
                     gamma_, c_, norm_, T_, safety_);
}

double FourierBarrierCertificate::apply_impl(ConstVectorRef x) const {
  LUCID_CHECK_ARGUMENT_EQ(is_synthesized(), true);
  LUCID_CHECK_ARGUMENT_EQ(x.size(), coefficients_.size());
  return coefficients_.dot(x);
}

void FourierBarrierCertificate::optimiser_callback(bool success, double obj_val, const Vector& coefficients, double eta,
                                                   double c, double norm, double b_norm) {
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
  if (norm > b_norm) {
    LUCID_WARN_FMT("Actual norm exceeds bound: {} > {} (diff: {})", norm, b_norm, norm - b_norm);
  }
}

std::ostream& operator<<(std::ostream& os, const FourierBarrierCertificateParameters& params) {
  return os << params.to_string();
}

std::ostream& operator<<(std::ostream& os, const FourierBarrierCertificate& obj) { return os << obj.to_string(); }

}  // namespace lucid
