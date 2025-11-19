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
#include <ostream>
#include <span>
#include <string>
#include <utility>
#include <vector>

#include "lucid/lib/alglib.h"
#include "lucid/util/Stats.h"
#include "lucid/util/Timer.h"
#include "lucid/util/error.h"
#include "lucid/util/logging.h"

namespace lucid {
#ifdef LUCID_ALGLIB_BUILD
namespace {

enum RetCode {
  UNBOUNDED = -4,
  INFEASIBLE = -3,
  INFEASIBLE_OR_UNBOUNDED = -2,
  OK = 1,  // 2, 3, 4 are also OK
  MAX_ITERATIONS = 5,
  STOPPING_CONDITIONS_TOO_STRINGENT = 7
};

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
      vars_[vars_.length() - FourierBarrierSynthesisProblem::num_extra_vars + i] = additional_vars[i];
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
    alglib::minlpaddlc2(state_, tot_vars, vals, tot_vars.length(), Op == '<' ? alglib::fp_neginf : rhs,
                        Op == '>' ? alglib::fp_posinf : rhs);
  }

  template <char Op, std::size_t N>
    requires(Op == '<' || Op == '>' || Op == '=')
  void add_constraint(const double* coeffs_ptr, std::array<alglib::ae_int_t, N> additional_vars,
                      std::array<double, N> additional_coeffs, const double rhs) {
    return add_constraint<Op>(std::span(coeffs_ptr, vars_.length() - FourierBarrierSynthesisProblem::num_extra_vars),
                              additional_vars, additional_coeffs, rhs);
  }

  void set_bounds(alglib::ae_int_t var, const double lb = alglib::fp_neginf, const double ub = alglib::fp_posinf) {
    LUCID_ASSERT(var >= 0, "Variable index must be non-negative.");
    LUCID_ASSERT(var < vars_.length(), "Variable index must be less than the number of variables.");
    LUCID_ASSERT(lb <= ub, "Lower bound must be less than or equal to upper bound.");
    alglib::minlpsetbci(state_, var, lb, ub);
  }

  template <char Op>
    requires(Op == '<' || Op == '>')
  void add_min_max_bounds(ConstMatrixRef lattice, const std::vector<Index>& mask, const Dimension var,
                          [[maybe_unused]] const std::string& set_name) {
    LUCID_DEBUG_FMT(
        "Xn/{} lattice constraints - {} constraints\n"
        "for all x in Xn/{}: [ B(x) {}= {}_Xn/{} ]",
        set_name, mask.size(), set_name, Op, Op == '<' ? "max" : "min", set_name);
    for (const Index row : mask) {
      add_constraint<Op>(lattice.row(row).data(), std::array{var}, std::array{-1.0}, 0.0);
    }
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
  alglib::integer_1d_array vars_;
  alglib::real_1d_array coeffs_;
  alglib::real_1d_array x_;
  alglib::minlpstate state_;
  alglib::minlpreport rep_;
};
}  // namespace

bool AlglibOptimiser::solve_fourier_barrier_synthesis_impl(const FourierBarrierSynthesisProblem& problem,
                                                           const SolutionCallback& cb) const {
  static_assert(Matrix::IsRowMajor, "Row major order is expected to avoid copy/eval");
  static_assert(std::remove_reference_t<ConstMatrixRef>::IsRowMajor, "Row major order is expected to avoid copy/eval");

  const double max_num = alglib::fp_posinf;
  const auto& [num_constraints, fxn_lattice, dn_lattice, x_include_mask, x_exclude_mask, x0_include_mask,
               x0_exclude_mask, xu_include_mask, xu_exclude_mask, T, gamma, eta_coeff, min_x0_coeff, diff_sx0_coeff,
               gamma_coeff, max_xu_coeff, diff_sxu_coeff, ebk, c_ebk_coeff, min_d_coeff, diff_d_sx_coeff, max_x_coeff,
               diff_sx_coeff] = problem;

  const int num_vars = static_cast<int>(fxn_lattice.cols() + FourierBarrierSynthesisProblem::num_extra_vars);
  AlglibLpProblem lp_problem{num_vars};

  std::array<Dimension, FourierBarrierSynthesisProblem::num_extra_vars> special_vars;
  for (std::size_t i = 0; i < special_vars.size(); ++i) {
    special_vars[i] = static_cast<Dimension>(fxn_lattice.cols() + i);
  }
  const auto [c, eta, min_x0, max_sx0, max_xu, min_sxu, max_x, min_sx, min_d, max_d_sx] = special_vars;

  // Variables related to the feature map
  // -inf <= b_i <= inf
  for (int var = 0; var < fxn_lattice.cols(); ++var) lp_problem.set_bounds(var);
  // 0 <= c <= inf
  lp_problem.set_bounds(c, 0, max_num);
  // 0 <= eta < gamma | To enforce a strict inequality, we sub a small number from gamma
  lp_problem.set_bounds(eta, 0, gamma - FourierBarrierSynthesisProblem::tolerance);
  // 0 <= var <= inf
  for (Dimension var : std::array{min_x0, max_xu, max_x}) lp_problem.set_bounds(var, 0, max_num);
  // -inf <= var <= inf
  for (Dimension var : std::array{min_d, max_sx0, min_sxu, min_sx, max_d_sx}) {
    lp_problem.set_bounds(var, -max_num, max_num);
  }

  // To obtain only positive safety probabilities, restrict
  // eta + c*T in [0, gamma]
  // 1) eta + c*T >= 0 by design
  // 2) eta + c*T <= gamma
  LUCID_DEBUG("Restricting safety probabilities to be positive");
  lp_problem.add_constraint<'<'>(std::array{eta, c}, std::array{1.0, static_cast<double>(T)}, gamma);

  LUCID_DEBUG_FMT(
      "X0 lattice constraints - {} constraints\n"
      "for all x_0: [ B(x_0) >= min_X0] AND [ B(x_0) <= hateta ]\n"
      "hateta = eta_coeff * eta + min_x0_coeff * min_X0 - diff_sx0_coeff * max_sx0\n"
      "hateta = {} * eta + {} * min_X0 - {} * max_sx0",
      x0_include_mask.size() * 2, eta_coeff, min_x0_coeff, diff_sx0_coeff);
  for (Index row : x0_include_mask) {
    // B(x_0) >= min_x0
    lp_problem.add_constraint<'>'>(fxn_lattice.row(row).data(), std::array{min_x0}, std::array{-1.0}, 0.0);
    // B(x_0) <= hateta
    lp_problem.add_constraint<'<'>(fxn_lattice.row(row).data(), std::array{eta, min_x0, max_sx0},
                                   std::array{-eta_coeff, -min_x0_coeff, diff_sx0_coeff}, 0.0);
  }

  LUCID_DEBUG_FMT(
      "Xu lattice constraints - {} constraints\n"
      "for all x_u: [ B(x_u) <= max_Xu ] AND [ B(x_u) >= hatgamma ] \n"
      "hatgamma = gamma_coeff * gamma + max_Xu_coeff * max_Xu - diff_sxu_coeff * min_sxu\n"
      "hatgamma = {} + {} * max_Xu - {} * min_sxu",
      xu_include_mask.size() * 2, gamma_coeff * gamma, max_xu_coeff, diff_sxu_coeff);
  for (Index row : xu_include_mask) {
    // B(x_u) <= max_xu
    lp_problem.add_constraint<'<'>(fxn_lattice.row(row).data(), std::array{max_xu}, std::array{-1.0}, 0.0);
    // B(x_u) >= hatgamma
    lp_problem.add_constraint<'>'>(fxn_lattice.row(row).data(), std::array{max_xu, min_sxu},
                                   std::array{-max_xu_coeff, diff_sxu_coeff}, gamma_coeff * gamma);
  }

  LUCID_DEBUG_FMT(
      "Kushner constraints (verification case) - {} constraints\n"
      "for all x: [ B(xp) - B(x) >= min_d ] AND [ B(xp) - B(x) <= hatDelta ] AND \n"
      "hatDelta = c_ebk_coeff * (c - ebk) + min_d_coeff * min_d - diff_d_sx_coeff * max_d_sx\n"
      "hatDelta = {} * (c - {}) + {} * min_d - {} * max_d_sx",
      x_include_mask.size() * 2, c_ebk_coeff, ebk, min_d_coeff, diff_d_sx_coeff);
  for (Index row : x_include_mask) {
    // B(x) >= minDelta
    lp_problem.add_constraint<'>'>(dn_lattice.row(row).data(), std::array{min_d}, std::array{-1.0}, 0.0);
    // B(xp) - B(x) <= hatDelta
    lp_problem.add_constraint<'<'>(dn_lattice.row(row).data(), std::array{c, min_d, max_d_sx},
                                   std::array{-c_ebk_coeff, -min_d_coeff, diff_d_sx_coeff}, -c_ebk_coeff * ebk);
  }

  LUCID_DEBUG_FMT(
      "Positive barrier - {} constraints\n"
      "for all x: [ B(x) <= max_X ] AND [ B(x) >= hatxi ]\n"
      "hatxi = max_x_coeff * max_X - diff_sx_coeff * min_sx\n"
      "hatxi = {} * max_X - {} * min_sx",
      x_include_mask.size() * 2, max_x_coeff, diff_sx_coeff);
  for (Index row : x_include_mask) {
    // B(x) <= max_x
    lp_problem.add_constraint<'<'>(fxn_lattice.row(row).data(), std::array{max_x}, std::array{-1.0}, 0.0);
    // B(x) >= hatxi
    lp_problem.add_constraint<'>'>(fxn_lattice.row(row).data(), std::array{max_x, min_sx},
                                   std::array{-max_x_coeff, diff_sx_coeff}, 0.0);
  }

  lp_problem.add_min_max_bounds<'<'>(fxn_lattice, x0_exclude_mask, max_sx0, "X0");
  lp_problem.add_min_max_bounds<'>'>(fxn_lattice, xu_exclude_mask, min_sxu, "Xu");
  lp_problem.add_min_max_bounds<'>'>(fxn_lattice, x_exclude_mask, min_sx, "X");
  lp_problem.add_min_max_bounds<'<'>(dn_lattice, x_exclude_mask, max_d_sx, "dX");

  // Objective function (η + cT)
  std::vector<double> cost_data(num_vars, 0.0);
  cost_data[c] = T / gamma;
  cost_data[eta] = 1.0 / gamma;
  lp_problem.set_min_objective(cost_data);

  LUCID_INFO("Optimizing");
  lp_problem.solve();

  // Check if the problem is infeasible
  if (lp_problem.report().terminationtype < 1 || lp_problem.report().terminationtype > 4) {
    LUCID_INFO_FMT("No solution found, optimization status = {}", lp_problem.report().terminationtype);
    cb(false, 0, Vector{}, 0, 0, 0);
    return false;
  }

  const Vector solution{
      Vector::NullaryExpr(fxn_lattice.cols(), [&lp_problem](const Index i) { return lp_problem.solution()[i]; })};
  cb(true, lp_problem.report().f, solution, lp_problem.solution()[eta], lp_problem.solution()[c], solution.norm());
  return true;
}

#else
bool AlglibOptimiser::solve_fourier_barrier_synthesis_impl(const FourierBarrierSynthesisProblem&,
                                                           const SolutionCallback&) const {
  LUCID_NOT_SUPPORTED_MISSING_BUILD_DEPENDENCY("AlglibOptimiser::solve_fourier_barrier_synthesis_impl", "alglib");
}
#endif  // LUCID_ALGLIB_BUILD

std::string AlglibOptimiser::to_string() const {
  return fmt::format("AlglibOptimiser( problem_log_file( {} ) iis_log_file( {} ) )", problem_log_file_, iis_log_file_);
}

std::ostream& operator<<(std::ostream& os, const AlglibOptimiser& optimiser) { return os << optimiser.to_string(); }

}  // namespace lucid
