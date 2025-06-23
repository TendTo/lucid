/**
 * @author Room 6.030
 * @copyright 2025 lucid
 * @licence BSD 3-Clause License
 * @file
 */
#include "lucid/verification/AlglibOptimiser.h"

#include <array>
#include <limits>
#include <memory>
#include <span>
#include <string>
#include <utility>
#include <vector>

#include "lucid/lib/alglib.h"
#include "lucid/util/error.h"
#include "lucid/util/logging.h"

namespace lucid {

#ifdef LUCID_ALGLIB_BUILD
namespace {
class AlglibLpProblem {
 public:
  explicit AlglibLpProblem(const alglib::ae_int_t num_vars) : vars_{}, coeffs_{}, state_{}, rep_{} {
    alglib::minlpcreate(num_vars, state_);
    vars_.setlength(num_vars);
    coeffs_.setlength(num_vars);
    for (alglib::ae_int_t i = 0; i < vars_.length(); ++i) vars_[i] = i;
  }

  template <char Op, std::size_t N>
    requires(Op == '<' || Op == '>' || Op == '=')
  void add_constraint(std::span<const double> coeffs, std::array<alglib::ae_int_t, N> additional_vars,
                      std::array<double, N> additional_coeffs, const double rhs) {
    LUCID_ASSERT(coeffs.size() + additional_coeffs.size() <= static_cast<std::size_t>(vars_.length()),
                 "The number of coeffs and additional coeffs must not exceed the tot number of variables.");
    LUCID_ASSERT(coeffs.size() + additional_coeffs.size() <= static_cast<std::size_t>(coeffs_.length()),
                 "The number of coeffs and additional coeffs must not exceed the tot number of coefficients.");

    // Make sure the vars contain the original rkhs variables and the additional variables
    for (alglib::ae_int_t i = 0; i < static_cast<alglib::ae_int_t>(additional_vars.size()); ++i) {
      vars_[vars_.length() - 6 + i] = additional_vars[i];
    }
    // Copy the coefficients into the coeffs_ array
    alglib::ae_int_t i = 0;
    for (const double coeff : coeffs) coeffs_[i++] = coeff;
    for (const double coeff : additional_coeffs) coeffs_[i++] = coeff;

    alglib::minlpaddlc2(state_, vars_, coeffs_, static_cast<alglib::ae_int_t>(coeffs.size() + additional_coeffs.size()),
                        Op == '<' ? alglib::fp_neginf : rhs, Op == '>' ? alglib::fp_posinf : rhs);
  }

  template <char Op, std::size_t N>
    requires(Op == '<' || Op == '>' || Op == '=')
  void add_constraint(std::array<alglib::ae_int_t, N> vars, std::array<double, N> coeffs, const double rhs) {
    alglib::integer_1d_array tot_vars;
    tot_vars.setcontent(static_cast<alglib::ae_int_t>(vars.size()), vars.data());

    alglib::real_1d_array vals;
    vals.attach_to_ptr(static_cast<alglib::ae_int_t>(coeffs.size()), coeffs.data());
    alglib::minlpaddlc2(state_, tot_vars, vals, static_cast<alglib::ae_int_t>(coeffs.size()),
                        Op == '<' ? alglib::fp_neginf : rhs, Op == '>' ? alglib::fp_posinf : rhs);
  }

  template <char Op, std::size_t N>
    requires(Op == '<' || Op == '>' || Op == '=')
  void add_constraint(const double* coeffs_ptr, std::array<alglib::ae_int_t, N> additional_vars,
                      std::array<double, N> additional_coeffs, const double rhs) {
    return add_constraint<Op>(std::span(coeffs_ptr, vars_.length() - 6), additional_vars, additional_coeffs, rhs);
  }

  void set_bounds(alglib::ae_int_t var, const double lb = alglib::fp_neginf, const double ub = alglib::fp_posinf) {
    LUCID_ASSERT(var >= 0, "Variable index must be non-negative.");
    LUCID_ASSERT(var < vars_.length(), "Variable index must be less than the number of variables.");
    LUCID_ASSERT(lb <= ub, "Lower bound must be less than or equal to upper bound.");
    alglib::minlpsetbci(state_, var, lb, ub);
  }

  void set_min_objective(std::span<double> coeffs) {
    alglib::real_1d_array vals;
    vals.attach_to_ptr(static_cast<alglib::ae_int_t>(coeffs.size()), coeffs.data());
    alglib::minlpsetcost(state_, vals);
  }

  void solve() {
    // minlpsetscale(state, s); // TODO(tend): do we want to specify a scaling vector?
    // minlpsetalgoipm(state); // TODO(tend): do we want to specify an algorithm?
    alglib::minlpoptimize(state_);
    alglib::minlpresults(state_, x_, rep_);
  }

  [[nodiscard]] const alglib::minlpstate& state() const { return state_; }
  [[nodiscard]] const alglib::minlpreport& report() const { return rep_; }
  [[nodiscard]] const alglib::real_1d_array& solution() const { return x_; }

 private:
  void init() {}

  alglib::integer_1d_array vars_;
  alglib::real_1d_array coeffs_;
  alglib::real_1d_array x_;
  alglib::minlpstate state_;
  alglib::minlpreport rep_;
};
}  // namespace
#endif

AlglibOptimiser::AlglibOptimiser(const int T, const double gamma, const double epsilon, const double b_norm,
                                 const double b_kappa, const double sigma_f, const double C_coeff)
    : T_{T},
      gamma_{gamma},
      epsilon_{epsilon},
      b_norm_{b_norm},
      b_kappa_{b_kappa},
      sigma_f_{sigma_f},
      C_coeff_{C_coeff} {
  LUCID_CHECK_ARGUMENT_CMP(T, >, 0);
}

#ifdef LUCID_ALGLIB_BUILD
bool AlglibOptimiser::solve(ConstMatrixRef f0_lattice, ConstMatrixRef fu_lattice, ConstMatrixRef phi_mat,
                            ConstMatrixRef w_mat, const Dimension rkhs_dim, const Dimension num_frequencies_per_dim,
                            const Dimension num_frequency_samples_per_dim, const Dimension original_dim,
                            const SolutionCallback& cb) const {
  static_assert(Matrix::IsRowMajor, "Row major order is expected to avoid copy/eval");
  static_assert(std::remove_reference_t<ConstMatrixRef>::IsRowMajor, "Row major order is expected to avoid copy/eval");
  LUCID_CHECK_ARGUMENT_CMP(num_frequency_samples_per_dim, >, 0);
  constexpr double min_num = 1e-8;  // Minimum variable value for numerical stability
  const double max_num = alglib::fp_posinf;
  constexpr double min_eta = 0;
  const auto num_vars = static_cast<alglib::ae_int_t>(rkhs_dim + 2 + 4);
  const double C =
      std::pow(1 - C_coeff_ * 2.0 * num_frequencies_per_dim / num_frequency_samples_per_dim, -original_dim / 2.0);

  // What if we make C as big as it can be?
  // const double C = pow((1 - 2.0 * num_freq_per_dim / (2.0 * num_freq_per_dim + 1)), -original_dim / 2.0);
  LUCID_DEBUG_FMT("C: {}", C);

  AlglibLpProblem lp_problem{num_vars};

  // Specify constraints
  // Variables [b_1, ..., b_nBasis_x, c, eta, minX0, maxXU, maxXX, minDelta] in the verification case
  // Variables [b_1, ..., b_nBasis_x, c, eta, ...
  // SAT(x_1,u_1), ..., SAT(x_n_X,u1), SAT(x_1,u_n_USUpp), ..., SAT(x_n_X,u_n_USUpp), ...
  // SATOR(x_1), ..., SATOR(x_n_X)] in the control case
  const alglib::ae_int_t c = num_vars - 6;         // Index of the c variable
  const alglib::ae_int_t eta = num_vars - 5;       // Index of the eta variable
  const alglib::ae_int_t minX0 = num_vars - 4;     // Index of the minX0 variable
  const alglib::ae_int_t maxXU = num_vars - 3;     // Index of the maxXU variable
  const alglib::ae_int_t maxXX = num_vars - 2;     // Index of the maxXX variable
  const alglib::ae_int_t minDelta = num_vars - 1;  // Index of the minDelta variable

  // Variables related to the feature map
  for (int var = 0; var < rkhs_dim; ++var) lp_problem.set_bounds(var);
  lp_problem.set_bounds(c, 0, max_num);
  lp_problem.set_bounds(eta, min_eta, gamma_ - min_num);  // To enforce a strict inequality
  lp_problem.set_bounds(minDelta, alglib::fp_neginf, max_num);
  for (const alglib::ae_int_t var : std::array{minX0, maxXU, maxXX}) {
    lp_problem.set_bounds(var, 0, max_num);
  }

  const double maxXX_coeff = -(C - 1) / (C + 1);
  const double fctr1 = 2 / (C + 1);
  const double fctr2 = (C - 1) / (C + 1);
  const double unsafe_rhs = fctr1 * gamma_;
  const double kushner_rhs = -fctr1 * epsilon_ * b_norm_ * std::abs(sigma_f_);

  // To obtain only positive safety probabilities, restrict
  // eta + c*T in [0, gamma]
  // 1) eta + c*T >= 0 by design
  // 2) eta + c*T <= gamma
  LUCID_DEBUG("Restricting safety probabilities to be positive");
  lp_problem.add_constraint<'<'>(std::array{eta, c}, std::array{1.0, static_cast<double>(T_)}, gamma_);

  // TODO(tend): since we are using row major order, there is no need to copy each row of the matrix.
  //  Just add an assertion to make sure this is not changed at a later date, breaking the LP problem.
  LUCID_DEBUG_FMT(
      "Positive barrier - {} constraints\n"
      "for all x: [ B(x) >= hatxi ] AND [ B(x) <= maxXX ]\n"
      "hatxi = (C - 1) / (C + 1) * maxXX",
      phi_mat.rows() * 2);
  for (Index row = 0; row < phi_mat.rows(); ++row) {
    // B(x) >= hatxi
    lp_problem.add_constraint<'>'>(phi_mat.row(row).data(), std::array{maxXX}, std::array{maxXX_coeff}, 0.0);
    // B(x) <= maxXX
    lp_problem.add_constraint<'<'>(phi_mat.row(row).data(), std::array{maxXX}, std::array{-1.0}, 0.0);
  }

  LUCID_DEBUG_FMT(
      "Initial constraints - {} constraints\n"
      "for all x_0: [ B(x_0) <= hateta ] AND [ B(x_0) >= minX0 ]\n"
      "hateta = 2 / (C + 1) * eta + (C - 1) / (C + 1) * minX0",
      f0_lattice.rows() * 2);
  for (Index row = 0; row < f0_lattice.rows(); ++row) {
    // B(x_0) <= hateta
    lp_problem.add_constraint<'<'>(f0_lattice.row(row).data(), std::array{eta, minX0}, std::array{-fctr1, -fctr2}, 0.0);
    // B(x_0) >= minX0
    lp_problem.add_constraint<'>'>(f0_lattice.row(row).data(), std::array{minX0}, std::array{-1.0}, 0.0);
  }

  LUCID_DEBUG_FMT(
      "Unsafe constraints - {} constraints\n"
      "for all x_u: [ B(x_u) >= hatgamma ] AND [ B(x_u) <= maxXU ]\n"
      "hatgamma = 2 / (C + 1) * gamma + (C - 1) / (C + 1) * maxXU",
      fu_lattice.rows() * 2);
  for (Index row = 0; row < fu_lattice.rows(); ++row) {
    // B(x_u) >= hatgamma
    lp_problem.add_constraint<'>'>(fu_lattice.row(row).data(), std::array{maxXU}, std::array{-fctr2}, unsafe_rhs);
    // B(x_u) <= maxXU
    lp_problem.add_constraint<'<'>(fu_lattice.row(row).data(), std::array{maxXU}, std::array{-1.0}, 0.0);
  }

  LUCID_DEBUG_FMT(
      "Kushner constraints (verification case) - {} constraints\n"
      "for all x: [ B(xp) - B(x) <= hatDelta ] AND [ B(x) >= minDelta ]\n"
      "hatDelta = 2 / (C + 1) * (c - epsilon*Bnorm*kappa_x) + (C - 1) / (C + 1) * minDelta",
      phi_mat.rows() * 2);
  const Matrix mult{w_mat - b_kappa_ * phi_mat};
  for (Index row = 0; row < mult.rows(); ++row) {
    // B(xp) - B(x) <= hatDelta
    lp_problem.add_constraint<'<'>(mult.row(row).data(), std::array{c, minDelta}, std::array{-fctr1, -fctr2},
                                   kushner_rhs);
    // B(x) >= minDelta
    lp_problem.add_constraint<'>'>(mult.row(row).data(), std::array{minDelta}, std::array{-1.0}, 0.0);
  }

  // Objective function (η + cT)
  std::vector<double> cost_data(num_vars, 0.0);
  cost_data[c] = T_ / gamma_;
  cost_data[eta] = 1.0 / gamma_;
  lp_problem.set_min_objective(cost_data);

  LUCID_INFO("Optimizing");
  lp_problem.solve();

  // Check if the problem is infeasible
  if (lp_problem.report().terminationtype < 1 && lp_problem.report().terminationtype > 4) {
    LUCID_INFO_FMT("No solution found, optimization status = {}", lp_problem.report().terminationtype);
    cb(false, 0, Vector{}, 0, 0, 0);
    return false;
  }

  LUCID_INFO_FMT("Solution found, objective = {}", lp_problem.report().f);
  LUCID_INFO_FMT("Satisfaction probability is {:.6f}%", (1 - lp_problem.report().f) * 100);

  const Vector solution{Vector::NullaryExpr(rkhs_dim, [&lp_problem](Index i) { return lp_problem.solution()[i]; })};
  double actual_norm = solution.norm();
  LUCID_INFO_FMT("Actual norm: {}", actual_norm);
  if (actual_norm > b_norm_) {
    LUCID_WARN_FMT("Actual norm exceeds bound: {} > {} (diff: {})", actual_norm, b_norm_, actual_norm - b_norm_);
  }

  cb(true, lp_problem.report().f, solution, lp_problem.solution()[eta], lp_problem.solution()[c], actual_norm);
  return true;
}
#else
bool AlglibOptimiser::solve(ConstMatrixRef, ConstMatrixRef, ConstMatrixRef, ConstMatrixRef, Dimension, Dimension,
                            Dimension, Dimension, const SolutionCallback&) const {
  LUCID_NOT_SUPPORTED_MISSING_DEPENDENCY("AlglibOptimiser::solve", "alglib");
  return false;
}
#endif  // LUCID_GUROBI_BUILD

}  // namespace lucid
