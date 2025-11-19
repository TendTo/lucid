/**
 * @author Room 6.030
 * @copyright 2025 lucid
 * @licence BSD 3-Clause License
 * @file
 */
#include "lucid/verification/GurobiOptimiser.h"

#include <limits>
#include <memory>
#include <ostream>
#include <span>
#include <string>
#include <utility>

#include "lucid/lib/gurobi.h"
#include "lucid/util/Stats.h"
#include "lucid/util/Timer.h"
#include "lucid/util/error.h"
#include "lucid/util/logging.h"

#ifdef LUCID_PYTHON_BUILD
#include "bindings/pylucid/interrupt.h"
#endif

#ifndef NLOG
#define LUCID_MODEL_ADD_CONSTRAINT(model, expr, op, rhs, name, should_log)   \
  do {                                                                       \
    if (should_log) {                                                        \
      (model).addConstr(expr, op, rhs).set(GRB_StringAttr_ConstrName, name); \
    } else {                                                                 \
      (model).addConstr(expr, op, rhs);                                      \
    }                                                                        \
  } while (0)
#else
#define LUCID_MODEL_ADD_CONSTRAINT(model, expr, op, rhs, name, should_log) model.addConstr(expr, op, rhs)
#endif

namespace lucid {

#ifdef LUCID_GUROBI_BUILD

namespace {

#ifdef LUCID_PYTHON_BUILD
class PyInterruptCallback final : public GRBCallback {
 public:
  void callback() override { py_check_signals(); }
};
#endif

/**
 * Add lower and upper bounds to a Gurobi variable such that
 * @f[
 * \texttt{lb} \leq \texttt{var} \leq \texttt{ub} .
 * @f]
 * Setting `lb` to `-infinity` or `ub` to `infinity` removes the respective bound.
 * @param lb lower bound
 * @param var Gurobi variable
 * @param ub  upper bound
 */
void set_var_bounds(const double lb, GRBVar& var, const double ub) {
  var.set(GRB_DoubleAttr_LB, lb);
  var.set(GRB_DoubleAttr_UB, ub);
}

/**
 * Given a `lattice`, add constraints to the `model` such that
 * @f[
 * x b^T \leq x \texttt{var} \quad \forall x \in \texttt{lattice(mask)} .
 * @f]
 * The verse of the inequality is determined by the template parameter `Op`.
 * @tparam Op comparison operator, either '<' or '>'
 * @param model Gurobi model
 * @param bs barrier coefficients we are looking for
 * @param lattice lattice of points
 * @param mask mask used to filter the lattice points
 * @param var variable representing the minimum/maximum value of the barrier over the set
 * @param set_name name of the set for logging/debug purposes
 * @param should_log whether to add names to the constraints for logging/debug purposes
 */
template <char Op>
  requires(Op == '<' || Op == '>')
void add_min_max_bounds(GRBModel& model, const std::span<GRBVar>& bs, ConstMatrixRef lattice,
                        const std::vector<Index>& mask, const GRBVar& var, [[maybe_unused]] const std::string& set_name,
                        [[maybe_unused]] const bool should_log) {
  LUCID_DEBUG_FMT(
      "Xn/{} lattice constraints - {} constraints\n"
      "for all x in Xn/{}: [ B(x) {}= {}_Xn/{} ]",
      set_name, mask.size(), set_name, Op, Op == '<' ? "max" : "min", set_name);
  for (Index row : mask) {
    GRBLinExpr expr{};
    expr.addTerms(lattice.row(row).data(), bs.data(), static_cast<int>(lattice.cols()));

    expr -= var;
    LUCID_MODEL_ADD_CONSTRAINT(model, expr, Op, 0,
                               fmt::format("B(Xn/{0})>={1}_Xn/{0}[{2}]", set_name, Op == '<' ? "max" : "min", row),
                               should_log);
  }
}

}  // namespace

bool GurobiOptimiser::solve(ConstMatrixRef f0_lattice, ConstMatrixRef fu_lattice, ConstMatrixRef phi_mat,
                            ConstMatrixRef w_mat, Dimension rkhs_dim, Dimension num_frequencies_per_dim,
                            Dimension num_frequency_samples_per_dim, Dimension original_dim,
                            const SolutionCallback& cb) const {
  TimerGuard tg{Stats::Scoped::top() ? &Stats::Scoped::top()->value().optimiser_timer : nullptr};
  static_assert(Matrix::IsRowMajor, "Row major order is expected to avoid copy/eval");
  static_assert(std::remove_reference_t<ConstMatrixRef>::IsRowMajor, "Row major order is expected to avoid copy/eval");
  LUCID_CHECK_ARGUMENT_CMP(num_frequency_samples_per_dim, >, 0);
  constexpr double min_num = 1e-8;  // Minimum variable value for numerical stability
  constexpr double max_num = std::numeric_limits<double>::infinity();
  constexpr double min_eta = 0;
  // make_me_periodic = 2 * boost::math::constants::pi / omega_low;
  // const int num_frequency_samples_per_dim_periodic = floor(num_frequency_samples_per_dim * make_me_periodic); //
  // Determine number of samples per dimension required in the associated lattice on the periodic space const int
  // fraction = num_frequencies_per_dim /
  //                           num_frequency_samples_per_dim_periodic;  // Determines the most sparsliest sampled
  //                           dimension
  // const double C = std::pow(1 - C_coeff_ * 2.0 * fraction, -original_dim / 2.0);
  const double C =
      std::pow(1 - C_coeff_ * 2.0 * num_frequencies_per_dim / num_frequency_samples_per_dim, -original_dim / 2.0);
  LUCID_DEBUG_FMT("C: {}", C);

  if (Stats::Scoped::top()) {
    Stats::Scoped::top()->value().num_variables = rkhs_dim + 2 + 4;
    Stats::Scoped::top()->value().num_constraints =
        1 + 2 * (phi_mat.rows() + f0_lattice.rows() + fu_lattice.rows() + phi_mat.rows());
  }

  GRBEnv env{true};
  env.set(GRB_IntParam_OutputFlag, LUCID_DEBUG_ENABLED);
  env.start();
  GRBModel model{env};
  model.set(GRB_DoubleParam_FeasibilityTol, 1e-9);
  model.set(GRB_DoubleParam_TimeLimit, 10000);
#ifdef LUCID_PYTHON_BUILD
  PyInterruptCallback callback;
  model.setCallback(&callback);
#endif

  // Specify constraints
  // Variables [b_1, ..., b_nBasis_x, c, eta, minX0, maxXU, maxXX, minDelta] in the verification case
  // Variables [b_1, ..., b_nBasis_x, c, eta, ...
  // SAT(x_1,u_1), ..., SAT(x_n_X,u1), SAT(x_1,u_n_USUpp), ..., SAT(x_n_X,u_n_USUpp), ...
  // SATOR(x_1), ..., SATOR(x_n_X)] in the control case
  int nVars = static_cast<int>(rkhs_dim + 2 + 4);
  const bool should_log = should_log_problem();
  std::unique_ptr<GRBVar[]> vars_{model.addVars(nVars)};
  const std::span<GRBVar> vars{vars_.get(), static_cast<std::size_t>(nVars)};
  GRBVar& c = vars[vars.size() - 6];
  GRBVar& eta = vars[vars.size() - 5];
  GRBVar& minX0 = vars[vars.size() - 4];
  GRBVar& maxXU = vars[vars.size() - 3];
  GRBVar& maxXX = vars[vars.size() - 2];
  GRBVar& minDelta = vars[vars.size() - 1];

#ifndef NLOG
  if (should_log) {
    c.set(GRB_StringAttr_VarName, "c");
    eta.set(GRB_StringAttr_VarName, "eta");
    minX0.set(GRB_StringAttr_VarName, "minX0");
    maxXU.set(GRB_StringAttr_VarName, "maxXU");
    maxXX.set(GRB_StringAttr_VarName, "maxXX");
    minDelta.set(GRB_StringAttr_VarName, "minDelta");
  }
#endif

  // Variables related to the feature map
  int idx = 0;
  for (GRBVar& var : vars.subspan(0, rkhs_dim)) {
    var.set(GRB_DoubleAttr_LB, -max_num);
    var.set(GRB_DoubleAttr_UB, max_num);
#ifndef NLOG
    if (should_log) var.set(GRB_StringAttr_VarName, fmt::format("b[{}]", idx++));
#endif
  }
  c.set(GRB_DoubleAttr_LB, 0);
  c.set(GRB_DoubleAttr_UB, max_num);
  eta.set(GRB_DoubleAttr_LB, min_eta);
  eta.set(GRB_DoubleAttr_UB, gamma_ - min_num);  // To enforce a strict inequality, we sub a small number to the ub
  minDelta.set(GRB_DoubleAttr_LB, -max_num);
  minDelta.set(GRB_DoubleAttr_UB, max_num);
  for (GRBVar& var : std::array{minX0, maxXU, maxXX}) {
    var.set(GRB_DoubleAttr_LB, 0);
    var.set(GRB_DoubleAttr_UB, max_num);
  }

  const double fctr1 = 2 / (C + 1);
  const double fctr2 = (C - 1) / (C + 1);
  const double unsafe_rhs = fctr1 * gamma_;
  const double kushner_rhs = -fctr1 * epsilon_ * b_norm_ * std::abs(sigma_f_);

  // To obtain only positive safety probabilities, restrict
  // eta + c*T in [0, gamma]
  // 1) eta + c*T >= 0 by design
  // 2) eta + c*T <= gamma
  LUCID_DEBUG("Restricting safety probabilities to be positive");
  LUCID_MODEL_ADD_CONSTRAINT(model, eta + T_ * c, GRB_LESS_EQUAL, gamma_, "eta+c*T<=gamma", should_log);

  LUCID_DEBUG_FMT(
      "Positive barrier - {} constraints\n"
      "for all x: [ B(x) >= hatxi ] AND [ B(x) <= maxXX ]\n"
      "hatxi = (C - 1) / (C + 1) * maxXX",
      phi_mat.rows() * 2);
  for (Index row = 0; row < phi_mat.rows(); ++row) {
    GRBLinExpr expr{};
    expr.addTerms(phi_mat.row(row).data(), vars_.get(), static_cast<int>(phi_mat.cols()));
    expr += maxXX * -fctr2;
    LUCID_MODEL_ADD_CONSTRAINT(model, expr, GRB_GREATER_EQUAL, 0, fmt::format("B(x)>=hatxi[{}]", row), should_log);
    expr.remove(maxXX);
    expr += -maxXX;
    LUCID_MODEL_ADD_CONSTRAINT(model, expr, GRB_LESS_EQUAL, 0, fmt::format("B(x)<=maxXX[{}]", row), should_log);
  }

  LUCID_DEBUG_FMT(
      "Initial constraints - {} constraints\n"
      "for all x_0: [ B(x_0) <= hateta ] AND [ B(x_0) >= minX0 ]\n"
      "hateta = 2 / (C + 1) * eta + (C - 1) / (C + 1) * minX0",
      f0_lattice.rows() * 2);
  for (Index row = 0; row < f0_lattice.rows(); ++row) {
    GRBLinExpr expr{};
    expr.addTerms(f0_lattice.row(row).data(), vars_.get(), static_cast<int>(f0_lattice.cols()));
    expr += -fctr1 * eta - fctr2 * minX0;
    LUCID_MODEL_ADD_CONSTRAINT(model, expr, GRB_LESS_EQUAL, 0, fmt::format("B(x_0)<=hateta[{}]", row), should_log);
    expr.remove(eta);
    expr.remove(minX0);
    expr += -minX0;
    LUCID_MODEL_ADD_CONSTRAINT(model, expr, GRB_GREATER_EQUAL, 0, fmt::format("B(x_0)>=minX0[{}]", row), should_log);
  }

  LUCID_DEBUG_FMT(
      "Unsafe constraints - {} constraints\n"
      "for all x_u: [ B(x_u) >= hatgamma ] AND [ B(x_u) <= maxXU ]\n"
      "hatgamma = 2 / (C + 1) * gamma + (C - 1) / (C + 1) * maxXU",
      fu_lattice.rows() * 2);
  for (Index row = 0; row < fu_lattice.rows(); ++row) {
    GRBLinExpr expr{};
    expr.addTerms(fu_lattice.row(row).data(), vars_.get(), static_cast<int>(fu_lattice.cols()));
    expr += -fctr2 * maxXU;
    LUCID_MODEL_ADD_CONSTRAINT(model, expr, GRB_GREATER_EQUAL, unsafe_rhs, fmt::format("B(x_u)>=hatgamma[{}]", row),
                               should_log);
    expr.remove(maxXU);
    expr += -maxXU;
    LUCID_MODEL_ADD_CONSTRAINT(model, expr, GRB_LESS_EQUAL, 0, fmt::format("B(x_u)<=maxXU[{}]", row), should_log);
  }

  LUCID_DEBUG_FMT(
      "Kushner constraints (verification case) - {} constraints\n"
      "for all x: [ B(xp) - B(x) <= hatDelta ] AND [ B(x) >= minDelta ]\n"
      "hatDelta = 2 / (C + 1) * (c - epsilon*Bnorm*kappa_x) + (C - 1) / (C + 1) * minDelta",
      phi_mat.rows() * 2);
  const Matrix mult{w_mat - b_kappa_ * phi_mat};
  for (Index row = 0; row < mult.rows(); ++row) {
    GRBLinExpr expr{};
    expr.addTerms(mult.row(row).data(), vars_.get(), static_cast<int>(mult.cols()));
    expr += -fctr1 * c - fctr2 * minDelta;  // TODO(tend): c − εB̄κ
    LUCID_MODEL_ADD_CONSTRAINT(model, expr, GRB_LESS_EQUAL, kushner_rhs, fmt::format("B(xp)-B(x)<=hatDelta[{}]", row),
                               should_log);
    expr.remove(c);
    expr.remove(minDelta);
    expr += -minDelta;
    LUCID_MODEL_ADD_CONSTRAINT(model, expr, GRB_GREATER_EQUAL, 0, fmt::format("B(xp)-B(x)>=minDelta[{}]", row),
                               should_log);
  }

  // Objective function (η + cT)
  model.setObjective(GRBLinExpr{(eta + c * T_) / gamma_}, GRB_MINIMIZE);

  LUCID_INFO("Optimizing");
  model.optimize();
  if (!problem_log_file_.empty()) model.write(problem_log_file_);

  if (model.get(GRB_IntAttr_SolCount) == 0) {
    model.computeIIS();
    LUCID_INFO_FMT("No solution found, optimization status = {}", model.get(GRB_IntAttr_Status));
    if (!iis_log_file_.empty()) model.write(iis_log_file_);
    cb(false, 0, Vector{}, 0, 0, 0);
    return false;
  }

  LUCID_INFO_FMT("Solution found, objective = {}", model.get(GRB_DoubleAttr_ObjVal));
  LUCID_INFO_FMT("Satisfaction probability is {:.6f}%", (1 - model.get(GRB_DoubleAttr_ObjVal)) * 100);

  const Vector solution{Vector::NullaryExpr(rkhs_dim, [&vars](Index i) { return vars[i].get(GRB_DoubleAttr_X); })};
  double actual_norm = solution.norm();
  LUCID_INFO_FMT("Actual norm: {}", actual_norm);
  if (actual_norm > b_norm_) {
    LUCID_WARN_FMT("Actual norm exceeds bound: {} > {} (diff: {})", actual_norm, b_norm_, actual_norm - b_norm_);
  }

  cb(true, model.get(GRB_DoubleAttr_ObjVal), solution, eta.get(GRB_DoubleAttr_X), c.get(GRB_DoubleAttr_X), actual_norm);
  return true;
}

bool GurobiOptimiser::solve_fourier_barrier_synthesis_impl(const FourierBarrierSynthesisProblem& problem,
                                                           const SolutionCallback& cb) const {
  static_assert(Matrix::IsRowMajor, "Row major order is expected to avoid copy/eval");
  static_assert(std::remove_reference_t<ConstMatrixRef>::IsRowMajor, "Row major order is expected to avoid copy/eval");

  const auto& [num_vars, num_constraints, fxn_lattice, dn_lattice, x_include_mask, x_exclude_mask, x0_include_mask,
               x0_exclude_mask, xu_include_mask, xu_exclude_mask, T, gamma, C, b_kappa, eta_coeff, min_x0_coeff,
               diff_sx0_coeff, gamma_coeff, max_xu_coeff, diff_sxu_coeff, ebk, c_ebk_coeff, min_d_coeff,
               diff_d_sx_coeff, max_x_coeff, diff_sx_coeff, fctr1, fctr2, unsafe_rhs, kushner_rhs, A_x, A_x0, A_xu] =
      problem;

  constexpr double tolerance = 1e-8;  // Minimum variable value for numerical stability
  constexpr double max_num = std::numeric_limits<double>::infinity();
  constexpr Dimension num_special_vars = 10;

  GRBEnv env{true};
  env.set(GRB_IntParam_OutputFlag, LUCID_DEBUG_ENABLED);
  env.start();
  GRBModel model{env};
  model.set(GRB_DoubleParam_FeasibilityTol, tolerance);
  model.set(GRB_DoubleParam_TimeLimit, 10000);
#ifdef LUCID_PYTHON_BUILD
  PyInterruptCallback callback;
  model.setCallback(&callback);
#endif

  // Specify constraints
  // Variables [b_1, ..., b_nBasis_x, c, eta, minX0, maxXU, maxXX, minDelta] in the verification case
  // Variables [b_1, ..., b_nBasis_x, c, eta, ...
  // SAT(x_1,u_1), ..., SAT(x_n_X,u1), SAT(x_1,u_n_USUpp), ..., SAT(x_n_X,u_n_USUpp), ...
  // SATOR(x_1), ..., SATOR(x_n_X)] in the control case
#ifndef NLOG
  const bool should_log = should_log_problem();
#else
  constexpr bool should_log = false;
#endif
  const std::unique_ptr<GRBVar[]> vars_{model.addVars(static_cast<int>(fxn_lattice.cols() + num_special_vars))};
  const std::span<GRBVar> vars{vars_.get(), static_cast<std::size_t>(fxn_lattice.cols() + num_special_vars)};
  const std::span<GRBVar> bs{vars.subspan(0, fxn_lattice.cols())};
  std::array<GRBVar, num_special_vars> special_vars;
  LUCID_ASSERT(bs.size() + special_vars.size() == vars.size(), "Variable size mismatch");
  for (std::size_t i = 0; i < special_vars.size(); ++i) {
    special_vars[i] = vars[vars.size() - special_vars.size() + i];
  }
  auto& [c, eta, min_x0, max_sx0, max_xu, min_sxu, max_x, min_sx, min_d, max_d_sx] = special_vars;

  if (should_log) {
    c.set(GRB_StringAttr_VarName, "c");
    eta.set(GRB_StringAttr_VarName, "eta");
    min_x0.set(GRB_StringAttr_VarName, "min_X0");
    max_sx0.set(GRB_StringAttr_VarName, "max_Xn/X0");
    max_xu.set(GRB_StringAttr_VarName, "max_Xu");
    min_sxu.set(GRB_StringAttr_VarName, "min_Xn/Xu");
    max_x.set(GRB_StringAttr_VarName, "max_X");
    min_sx.set(GRB_StringAttr_VarName, "min_Xn/X");
    min_d.set(GRB_StringAttr_VarName, "min_Delta");
    max_d_sx.set(GRB_StringAttr_VarName, "max_Delta_sX");
  }

  // Variables related to the feature map
  int idx = 0;
  for (GRBVar& var : bs) {
    set_var_bounds(-max_num, var, max_num);
    if (should_log) var.set(GRB_StringAttr_VarName, fmt::format("b[{}]", idx++));
  }
  // 0 <= c <= inf
  set_var_bounds(0, c, max_num);
  // 0 <= eta < gamma | To enforce a strict inequality, we sub a small number from gamma
  set_var_bounds(0, eta, gamma - tolerance);
  // -inf <= min_d <= inf
  set_var_bounds(-max_num, min_d, max_num);
  // 0 <= var <= inf
  for (GRBVar& var : std::array{min_x0, max_xu, max_x}) set_var_bounds(0, var, max_num);
  // -inf <= var <= inf
  for (GRBVar& var : std::array{max_sx0, min_sxu, min_sx, max_d_sx}) {
    set_var_bounds(-max_num, var, max_num);
  }

  // To obtain only positive safety probabilities, restrict
  // eta + c*T in [0, gamma]
  // 1) eta + c*T >= 0 by design
  // 2) eta + c*T <= gamma
  LUCID_DEBUG("Restricting safety probabilities to be positive");
  LUCID_MODEL_ADD_CONSTRAINT(model, eta + c * T, '<', gamma, "eta+c*T<=gamma", should_log);

  LUCID_DEBUG_FMT(
      "X0 lattice constraints - {} constraints\n"
      "for all x_0: [ B(x_0) >= min_X0] AND [ B(x_0) <= hateta ]\n"
      "hateta = eta_coeff * eta + min_x0_coeff * min_X0 - diff_sx0_coeff * max_sx0\n"
      "hateta = {} * eta + {} * min_X0 - {} * max_sx0",
      x0_include_mask.size() * 2, eta_coeff, min_x0_coeff, diff_sx0_coeff);
  for (Index row : x0_include_mask) {
    GRBLinExpr expr{};
    expr.addTerms(fxn_lattice.row(row).data(), bs.data(), static_cast<int>(fxn_lattice.cols()));

    expr -= min_x0;
    LUCID_MODEL_ADD_CONSTRAINT(model, expr, '>', 0, fmt::format("B(x_0)>=min_X0[{}]", row), should_log);
    expr.remove(min_x0);

    expr -= eta_coeff * eta + min_x0_coeff * min_x0 - diff_sx0_coeff * max_sx0;
    LUCID_MODEL_ADD_CONSTRAINT(model, expr, '<', 0, fmt::format("B(x_0)<=hateta[{}]", row), should_log);
  }

  LUCID_DEBUG_FMT(
      "Xu lattice constraints - {} constraints\n"
      "for all x_u: [ B(x_u) <= max_Xu ] AND [ B(x_u) >= hatgamma ] \n"
      "hatgamma = gamma_coeff * gamma + max_Xu_coeff * max_Xu - diff_sxu_coeff * min_sxu\n"
      "hatgamma = {} + {} * max_Xu - {} * min_sxu",
      xu_include_mask.size() * 2, gamma_coeff * gamma, max_xu_coeff, diff_sxu_coeff);
  for (Index row : xu_include_mask) {
    GRBLinExpr expr{};
    expr.addTerms(fxn_lattice.row(row).data(), bs.data(), static_cast<int>(fxn_lattice.cols()));

    expr -= max_xu;
    LUCID_MODEL_ADD_CONSTRAINT(model, expr, '<', 0, fmt::format("B(x_u)<=max_Xu[{}]", row), should_log);
    expr.remove(max_xu);

    expr -= max_xu_coeff * max_xu - diff_sxu_coeff * min_sxu;
    LUCID_MODEL_ADD_CONSTRAINT(model, expr, '>', gamma_coeff * gamma, fmt::format("B(x_u)>=hatgamma[{}]", row),
                               should_log);
  }

  LUCID_DEBUG_FMT(
      "Kushner constraints (verification case) - {} constraints\n"
      "for all x: [ B(xp) - B(x) >= min_d ] AND [ B(xp) - B(x) <= hatDelta ] AND \n"
      "hatDelta = c_ebk_coeff * (c - ebk) + min_d_coeff * min_d - diff_d_sx_coeff * max_d_sx\n"
      "hatDelta = {} * (c - {}) + {} * min_d - {} * max_d_sx",
      x_include_mask.size() * 2, c_ebk_coeff, ebk, min_d_coeff, diff_d_sx_coeff);
  for (Index row : x_include_mask) {
    GRBLinExpr expr{};
    expr.addTerms(dn_lattice.row(row).data(), bs.data(), static_cast<int>(dn_lattice.cols()));

    expr -= min_d;
    LUCID_MODEL_ADD_CONSTRAINT(model, expr, '>', 0, fmt::format("B(xp)-B(x)>=minDelta[{}]", row), should_log);
    expr.remove(min_d);

    expr -= c_ebk_coeff * c + min_d_coeff * min_d - diff_d_sx_coeff * max_d_sx;
    LUCID_MODEL_ADD_CONSTRAINT(model, expr, '<', -c_ebk_coeff * ebk, fmt::format("B(xp)-B(x)<=hatDelta[{}]", row),
                               should_log);
  }

  LUCID_DEBUG_FMT(
      "Positive barrier - {} constraints\n"
      "for all x: [ B(x) <= max_X ] AND [ B(x) >= hatxi ]\n"
      "hatxi = max_x_coeff * max_X - diff_sx_coeff * min_sx\n"
      "hatxi = {} * max_X - {} * min_sx",
      x_include_mask.size() * 2, max_x_coeff, diff_sx_coeff);
  for (Index row : x_include_mask) {
    GRBLinExpr expr{};
    expr.addTerms(fxn_lattice.row(row).data(), bs.data(), static_cast<int>(fxn_lattice.cols()));

    expr -= max_x;
    LUCID_MODEL_ADD_CONSTRAINT(model, expr, '<', 0, fmt::format("B(x)<=max_x[{}]", row), should_log);
    expr.remove(max_x);

    expr -= max_x_coeff * max_x - diff_sx_coeff * min_sx;
    LUCID_MODEL_ADD_CONSTRAINT(model, expr, '>', 0, fmt::format("B(x)>=hatxi[{}]", row), should_log);
  }

  add_min_max_bounds<'<'>(model, bs, fxn_lattice, x0_exclude_mask, max_sx0, "X0", should_log);
  add_min_max_bounds<'>'>(model, bs, fxn_lattice, xu_exclude_mask, min_sxu, "Xu", should_log);
  add_min_max_bounds<'>'>(model, bs, fxn_lattice, x_exclude_mask, min_sx, "X", should_log);
  add_min_max_bounds<'<'>(model, bs, dn_lattice, x_exclude_mask, max_d_sx, "dX", should_log);

  // Objective function (η + cT)
  model.setObjective(GRBLinExpr{(eta + c * T) / gamma}, GRB_MINIMIZE);

  if (!problem_log_file_.empty()) model.write(problem_log_file_);
  LUCID_INFO("Optimizing");
  model.optimize();

  if (model.get(GRB_IntAttr_SolCount) == 0) {
    model.computeIIS();
    LUCID_INFO_FMT("No solution found, optimization status = {}", model.get(GRB_IntAttr_Status));
    if (!iis_log_file_.empty()) model.write(iis_log_file_);
    cb(false, 0, Vector{}, 0, 0, 0);
    return false;
  }

  // Print the value of each variable
  LUCID_DEBUG_FMT("c: {}", c.get(GRB_DoubleAttr_X));
  LUCID_DEBUG_FMT("eta: {}", eta.get(GRB_DoubleAttr_X));
  LUCID_DEBUG_FMT("min_X0: {}", min_x0.get(GRB_DoubleAttr_X));
  LUCID_DEBUG_FMT("max_sx0: {}", max_sx0.get(GRB_DoubleAttr_X));
  LUCID_DEBUG_FMT("max_xu: {}", max_xu.get(GRB_DoubleAttr_X));
  LUCID_DEBUG_FMT("min_sxu: {}", min_sxu.get(GRB_DoubleAttr_X));
  LUCID_DEBUG_FMT("max_x: {}", max_x.get(GRB_DoubleAttr_X));
  LUCID_DEBUG_FMT("min_sx: {}", min_sx.get(GRB_DoubleAttr_X));
  LUCID_DEBUG_FMT("min_d: {}", min_d.get(GRB_DoubleAttr_X));
  LUCID_DEBUG_FMT("max_d_sx: {}", max_d_sx.get(GRB_DoubleAttr_X));

  const Vector solution{
      Vector::NullaryExpr(fxn_lattice.cols(), [&vars](const Index i) { return vars[i].get(GRB_DoubleAttr_X); })};
  cb(true, model.get(GRB_DoubleAttr_ObjVal), solution, eta.get(GRB_DoubleAttr_X), c.get(GRB_DoubleAttr_X),
     solution.norm());
  return true;
}

#else
bool GurobiOptimiser::solve(ConstMatrixRef, ConstMatrixRef, ConstMatrixRef, ConstMatrixRef, Dimension, Dimension,
                            Dimension, Dimension, const SolutionCallback&) const {
  LUCID_NOT_SUPPORTED_MISSING_BUILD_DEPENDENCY("GurobiOptimiser::solve", "Gurobi");
  return false;
}
bool GurobiOptimiser::solve_fourier_barrier_synthesis_impl(const FourierBarrierSynthesisProblem&,
                                                           const SolutionCallback&) const {
  LUCID_NOT_SUPPORTED_MISSING_BUILD_DEPENDENCY("GurobiOptimiser::solve_fourier_barrier_synthesis_impl", "Gurobi");
  return false;
}
std::pair<Vector, Vector> GurobiOptimiser::bounding_box(ConstMatrixRef, ConstVectorRef) {
  LUCID_NOT_SUPPORTED_MISSING_BUILD_DEPENDENCY("GurobiOptimiser::bounding_box", "Gurobi");
  return {Vector{}, Vector{}};
}
#endif  // LUCID_GUROBI_BUILD

std::string GurobiOptimiser::to_string() const {
  return fmt::format("GurobiOptimiser( problem_log_file( {} ) iis_log_file( {} ) )", problem_log_file_, iis_log_file_);
}

std::ostream& operator<<(std::ostream& os, const GurobiOptimiser& optimiser) { return os << optimiser.to_string(); }

}  // namespace lucid
