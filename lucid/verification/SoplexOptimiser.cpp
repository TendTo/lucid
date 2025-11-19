/**
 * @author c3054737
 * @copyright 2025 lucid
 * @licence BSD 3-Clause License
 * @file
 */
#include "lucid/verification/SoplexOptimiser.h"

#include <ostream>
#include <string>
#include <vector>

#include "lucid/lib/soplex.h"
#include "lucid/util/error.h"

#ifdef LUCID_PYTHON_BUILD
#include "bindings/pylucid/interrupt.h"
#endif

#ifndef NLOG
#define LUCID_MODEL_ADD_CONSTRAINT(model, lhs, vec, rhs, name, names, should_log) \
  do {                                                                            \
    rows.add(lhs, vec, rhs);                                                      \
    if (should_log) names.emplace_back(name);                                     \
  } while (0)
#else
#define LUCID_MODEL_ADD_CONSTRAINT(model, lhs, vec, rhs, name, names, should_log) rows.add(lhs, vec, rhs)
#endif

namespace lucid {

#ifdef LUCID_SOPLEX_BUILD

namespace {

/**
 * Given a `lattice`, add constraints to the `model` such that
 * @f[
 * \phi(x)^T b \lesseqgtr \texttt{var} \quad \forall x \in \texttt{lattice(mask)} .
 * @f]
 * The verse of the inequality is determined by the template parameter `Op`.
 * @tparam Op comparison operator, either '<' or '>'
 * @param rows problem's rows
 * @param constraint_names names of the constraints for logging/debug purposes
 * @param lattice lattice of points
 * @param mask mask used to filter the lattice points
 * @param var variable representing the minimum/maximum value of the barrier over the set
 * @param set_name name of the set for logging/debug purposes
 * @param should_log whether to add names to the constraints for logging/debug purposes
 */
template <char Op>
  requires(Op == '<' || Op == '>')
void add_min_max_bounds(soplex::LPRowSetReal& rows, std::vector<std::string>& constraint_names, ConstMatrixRef lattice,
                        const std::vector<Index>& mask, const int var, [[maybe_unused]] const std::string& set_name,
                        [[maybe_unused]] const bool should_log) {
  LUCID_DEBUG_FMT(
      "Xn/{} lattice constraints - {} constraints\n"
      "for all x in Xn/{}: [ B(x) {}= {}_Xn/{} ]",
      set_name, mask.size(), set_name, Op, Op == '<' ? "max" : "min", set_name);
  for (Index row : mask) {
    soplex::DSVectorReal vec(static_cast<int>(lattice.cols()) + 1);
    for (Index col = 0; col < lattice.cols(); ++col) vec.add(static_cast<int>(col), lattice(row, col));
    vec.add(var, -1.0);
    LUCID_MODEL_ADD_CONSTRAINT(rows, Op == '>' ? 0.0 : -soplex::infinity, vec, Op == '<' ? 0.0 : soplex::infinity,
                               fmt::format("B(x_0)>=min_x0[{}]", row), constraint_names, should_log);
  }
}

}  // namespace

bool SoplexOptimiser::solve_fourier_barrier_synthesis_impl(const FourierBarrierSynthesisProblem& problem,
                                                           const SolutionCallback& cb) const {
  static_assert(Matrix::IsRowMajor, "Row major order is expected to avoid copy/eval");
  static_assert(std::remove_reference_t<ConstMatrixRef>::IsRowMajor, "Row major order is expected to avoid copy/eval");

  const auto& [num_constraints, fxn_lattice, dn_lattice, x_include_mask, x_exclude_mask, x0_include_mask,
               x0_exclude_mask, xu_include_mask, xu_exclude_mask, T, gamma, eta_coeff, min_x0_coeff, diff_sx0_coeff,
               gamma_coeff, max_xu_coeff, diff_sxu_coeff, ebk, c_ebk_coeff, min_d_coeff, diff_d_sx_coeff, max_x_coeff,
               diff_sx_coeff] = problem;

#ifndef NLOG
  const bool should_log = should_log_problem();
#else
  constexpr bool should_log = false;
#endif

  // Default SoPlex parameters
  soplex::SoPlex spx;
  spx.setRealParam(soplex::SoPlex::TIMELIMIT, 10000.0);
  spx.setRealParam(soplex::SoPlex::FEASTOL, 10e-9);
  spx.setIntParam(soplex::SoPlex::READMODE, soplex::SoPlex::READMODE_REAL);
  spx.setIntParam(soplex::SoPlex::SOLVEMODE, soplex::SoPlex::SOLVEMODE_RATIONAL);
  spx.setIntParam(soplex::SoPlex::CHECKMODE, soplex::SoPlex::CHECKMODE_RATIONAL);
  spx.setIntParam(soplex::SoPlex::OBJSENSE, soplex::SoPlex::OBJSENSE_MINIMIZE);
  spx.setIntParam(soplex::SoPlex::VERBOSITY,
                  LUCID_DEBUG_ENABLED ? soplex::SoPlex::VERBOSITY_DEBUG : soplex::SoPlex::VERBOSITY_ERROR);

  // Specify constraints
  // Variables [b_1, ..., b_nBasis_x, c, eta, minX0, maxXU, maxXX, minDelta] in the verification case
  // Variables [b_1, ..., b_nBasis_x, c, eta, ...
  // SAT(x_1,u_1), ..., SAT(x_n_X,u1), SAT(x_1,u_n_USUpp), ..., SAT(x_n_X,u_n_USUpp), ...
  // SATOR(x_1), ..., SATOR(x_n_X)] in the control case
  const int num_vars = static_cast<int>(fxn_lattice.cols() + FourierBarrierSynthesisProblem::num_extra_vars);
  soplex::LPColSetReal cols(num_vars);
  soplex::LPRowSetReal rows(num_constraints);

  soplex::NameSet var_names_set{should_log ? num_vars : 0};
  soplex::NameSet constraint_names_set{should_log ? num_constraints : 0};
  std::vector<std::string> var_names;
  std::vector<std::string> constraint_names;
  var_names.reserve(should_log ? num_vars : 0);
  constraint_names.reserve(should_log ? num_constraints : 0);

  // Variables related to the feature map
  for (Dimension i = 0; i < fxn_lattice.cols(); ++i) {
    // -inf <= b_i <= inf
    cols.add(soplex::LPColReal{0.0, soplex::DSVectorReal(), soplex::infinity, -soplex::infinity});
    if (should_log) var_names.emplace_back(fmt::format("b[{}]", i));
  }
  std::array<int, FourierBarrierSynthesisProblem::num_extra_vars> special_vars;
  for (std::size_t i = 0; i < special_vars.size(); ++i) {
    special_vars[i] = static_cast<int>(fxn_lattice.cols() + i);
  }
  const auto [c, eta, min_x0, max_sx0, max_xu, min_sxu, max_x, min_sx, min_d, max_d_sx] = special_vars;

  // c (Objective function (η + cT)) | 0 <= c <= inf
  cols.add(soplex::LPColReal{static_cast<double>(T), soplex::DSVectorReal(), soplex::infinity, 0});
  // eta (Objective function (η + cT)) | 0 <= eta < gamma | To enforce a strict inequality, we sub a tol from gamma
  cols.add(soplex::LPColReal{1.0, soplex::DSVectorReal(), gamma - FourierBarrierSynthesisProblem::tolerance, 0});
  // 0 <= min_x0 <= inf
  cols.add(soplex::LPColReal{0.0, soplex::DSVectorReal(), soplex::infinity, 0});
  // -inf <= max_sx0 <= inf
  cols.add(soplex::LPColReal{0.0, soplex::DSVectorReal(), soplex::infinity, -soplex::infinity});
  // 0 <= max_xu <= inf
  cols.add(soplex::LPColReal{0.0, soplex::DSVectorReal(), soplex::infinity, 0});
  // -inf <= min_sxu <= inf
  cols.add(soplex::LPColReal{0.0, soplex::DSVectorReal(), soplex::infinity, -soplex::infinity});
  // 0 <= max_x <= inf
  cols.add(soplex::LPColReal{0.0, soplex::DSVectorReal(), soplex::infinity, 0});
  // -inf <= min_sx <= inf
  cols.add(soplex::LPColReal{0.0, soplex::DSVectorReal(), soplex::infinity, -soplex::infinity});
  // -inf <= min_d <= inf
  cols.add(soplex::LPColReal{0.0, soplex::DSVectorReal(), soplex::infinity, -soplex::infinity});
  // -inf <= max_d_sx <= inf
  cols.add(soplex::LPColReal{0.0, soplex::DSVectorReal(), soplex::infinity, -soplex::infinity});

  if (should_log) {
    var_names.emplace_back("c");
    var_names.emplace_back("eta");
    var_names.emplace_back("min_x0");
    var_names.emplace_back("max_sx0");
    var_names.emplace_back("max_xu");
    var_names.emplace_back("min_sxu");
    var_names.emplace_back("max_x");
    var_names.emplace_back("min_sx");
    var_names.emplace_back("min_d");
    var_names.emplace_back("max_d_sx");
  }

  // To obtain only positive safety probabilities, restrict
  // eta + c*T in [0, gamma]
  // 1) eta + c*T >= 0 by design
  // 2) eta + c*T <= gamma
  {
    LUCID_DEBUG("Restricting safety probabilities to be positive");
    soplex::DSVectorReal vec(2);
    vec.add(eta, 1.0);
    vec.add(c, static_cast<double>(T));
    LUCID_MODEL_ADD_CONSTRAINT(rows, -soplex::infinity, vec, gamma, "eta+c*T<=gamma", constraint_names, should_log);
  }

  LUCID_DEBUG_FMT(
      "X0 lattice constraints - {} constraints\n"
      "for all x_0: [ B(x_0) >= min_X0] AND [ B(x_0) <= hateta ]\n"
      "hateta = eta_coeff * eta + min_x0_coeff * min_X0 - diff_sx0_coeff * max_sx0\n"
      "hateta = {} * eta + {} * min_X0 - {} * max_sx0",
      x0_include_mask.size() * 2, eta_coeff, min_x0_coeff, diff_sx0_coeff);
  for (Index row : x0_include_mask) {
    soplex::DSVectorReal vec(static_cast<int>(fxn_lattice.cols()) + 3);
    for (Index col = 0; col < fxn_lattice.cols(); ++col) vec.add(static_cast<int>(col), fxn_lattice(row, col));
    // B(x0) >= min_x0
    vec.add(min_x0, -1.0);
    LUCID_MODEL_ADD_CONSTRAINT(rows, 0.0, vec, soplex::infinity, fmt::format("B(x_0)>=min_x0[{}]", row),
                               constraint_names, should_log);
    // B(x0) <= hateta
    vec.add(eta, -eta_coeff);
    vec.value(vec.pos(min_x0)) = -min_x0_coeff;
    vec.add(max_sx0, diff_sx0_coeff);
    LUCID_MODEL_ADD_CONSTRAINT(rows, -soplex::infinity, vec, 0.0, fmt::format("B(x_0)<=hateta[{}]", row),
                               constraint_names, should_log);
  }

  LUCID_DEBUG_FMT(
      "Xu lattice constraints - {} constraints\n"
      "for all x_u: [ B(x_u) <= max_Xu ] AND [ B(x_u) >= hatgamma ] \n"
      "hatgamma = gamma_coeff * gamma + max_Xu_coeff * max_Xu - diff_sxu_coeff * min_sxu\n"
      "hatgamma = {} + {} * max_Xu - {} * min_sxu",
      xu_include_mask.size() * 2, gamma_coeff * gamma, max_xu_coeff, diff_sxu_coeff);
  for (Index row : xu_include_mask) {
    soplex::DSVectorReal vec(static_cast<int>(fxn_lattice.cols()) + 1);
    for (Index col = 0; col < fxn_lattice.cols(); ++col) vec.add(static_cast<int>(col), fxn_lattice(row, col));
    // B(xu) <= max_xu
    vec.add(max_xu, -1.0);
    LUCID_MODEL_ADD_CONSTRAINT(rows, -soplex::infinity, vec, 0.0, fmt::format("B(x_u)<=max_Xu[{}]", row),
                               constraint_names, should_log);
    // B(xu) >= hatgamma
    vec.value(vec.pos(max_xu)) = -max_xu_coeff;
    vec.add(min_sxu, diff_sxu_coeff);
    LUCID_MODEL_ADD_CONSTRAINT(rows, gamma_coeff * gamma, vec, soplex::infinity,
                               fmt::format("B(x_u)>=hatgamma[{}]", row), constraint_names, should_log);
  }

  LUCID_DEBUG_FMT(
      "Kushner constraints (verification case) - {} constraints\n"
      "for all x: [ B(xp) - B(x) >= min_d ] AND [ B(xp) - B(x) <= hatDelta ] AND \n"
      "hatDelta = c_ebk_coeff * (c - ebk) + min_d_coeff * min_d - diff_d_sx_coeff * max_d_sx\n"
      "hatDelta = {} * (c - {}) + {} * min_d - {} * max_d_sx",
      x_include_mask.size() * 2, c_ebk_coeff, ebk, min_d_coeff, diff_d_sx_coeff);
  for (Index row : x_include_mask) {
    soplex::DSVectorReal vec(static_cast<int>(dn_lattice.cols()) + 2);
    for (Index col = 0; col < dn_lattice.cols(); ++col) vec.add(static_cast<int>(col), dn_lattice(row, col));
    // B(xp) - B(x) >= min_d
    vec.add(min_d, -1.0);
    LUCID_MODEL_ADD_CONSTRAINT(rows, 0.0, vec, soplex::infinity, fmt::format("B(xp)-B(x)>=min_d[{}]", row),
                               constraint_names, should_log);
    // B(xp) - B(x) <= hatDelta
    vec.add(c, -c_ebk_coeff);
    vec.value(vec.pos(min_d)) = -min_d_coeff;
    vec.add(max_d_sx, diff_d_sx_coeff);
    LUCID_MODEL_ADD_CONSTRAINT(rows, -soplex::infinity, vec, -c_ebk_coeff * ebk,
                               fmt::format("B(xp)-B(x)<=hatDelta[{}]", row), constraint_names, should_log);
  }

  LUCID_DEBUG_FMT(
      "Positive barrier - {} constraints\n"
      "for all x: [ B(x) <= max_X ] AND [ B(x) >= hatxi ]\n"
      "hatxi = max_x_coeff * max_X - diff_sx_coeff * min_sx\n"
      "hatxi = {} * max_X - {} * min_sx",
      x_include_mask.size() * 2, max_x_coeff, diff_sx_coeff);
  for (Index row : x_include_mask) {
    soplex::DSVectorReal vec(static_cast<int>(fxn_lattice.cols()) + 1);
    for (Index col = 0; col < fxn_lattice.cols(); ++col) vec.add(static_cast<int>(col), fxn_lattice(row, col));
    // B(x) <= max_x
    vec.add(max_x, -1.0);
    LUCID_MODEL_ADD_CONSTRAINT(rows, -soplex::infinity, vec, 0.0, fmt::format("B(x)<=max_x[{}]", row), constraint_names,
                               should_log);
    // B(x) >= hatxi
    vec.value(vec.pos(max_x)) = -max_x_coeff;
    vec.add(min_sx, diff_sx_coeff);
    LUCID_MODEL_ADD_CONSTRAINT(rows, 0.0, vec, soplex::infinity, fmt::format("B(x)>=hatxi[{}]", row), constraint_names,
                               should_log);
  }

  add_min_max_bounds<'<'>(rows, constraint_names, fxn_lattice, x0_exclude_mask, max_sx0, "X0", should_log);
  add_min_max_bounds<'>'>(rows, constraint_names, fxn_lattice, xu_exclude_mask, min_sxu, "Xu", should_log);
  add_min_max_bounds<'>'>(rows, constraint_names, fxn_lattice, x_exclude_mask, min_sx, "X", should_log);
  add_min_max_bounds<'<'>(rows, constraint_names, dn_lattice, x_exclude_mask, max_d_sx, "dX", should_log);

#ifndef NLOG
  for (const std::string& name : var_names) var_names_set.add(name.c_str());
  for (const std::string& name : constraint_names) constraint_names_set.add(name.c_str());
#endif

  spx.addColsReal(cols);
  spx.addRowsReal(rows);

  if (!problem_log_file_.empty()) {
    [[maybe_unused]] const bool res =
        spx.writeFile(problem_log_file_.c_str(), constraint_names_set.size() > 0 ? &constraint_names_set : nullptr,
                      var_names_set.size() > 0 ? &var_names_set : nullptr);
    LUCID_ASSERT(res, "error writing problem file");
  }

  LUCID_INFO("Optimizing");
#ifdef LUCID_PYTHON_BUILD
  spx.optimize();
  // TODO(tend): add support for interrupting SoPlex
  // volatile bool interrupt = false;
  // volatile bool finished = false;
  // std::thread interrupt_checker_thread([&spx, &interrupt, &finished]() {
  //   spx.optimize(&interrupt);
  //   LUCID_ERROR_FMT("soplex interrupt value: {}", interrupt);
  //   finished = true;
  // });
  // while (!finished && !interrupt) {
  //   py_interrupt_flag(&interrupt);
  //   LUCID_INFO("waiting...");
  //   if (interrupt) LUCID_ERROR("interrupt");
  //   std::this_thread::sleep_for(std::chrono::seconds(1));
  // }
  // interrupt_checker_thread.join();
#else
  spx.optimize();
#endif

  if (!spx.hasSol()) {
    LUCID_INFO_FMT("No solution found, optimization status = {}", spx.status());
    if (spx.hasDualFarkas() && !iis_log_file_.empty())
      spx.writeStateReal(iis_log_file_.c_str(), constraint_names_set.size() > 0 ? &constraint_names_set : nullptr,
                         var_names_set.size() > 0 ? &var_names_set : nullptr);
    cb(false, 0, Vector{}, 0, 0, 0);
    return false;
  }

  soplex::VectorReal sol{num_vars};
  [[maybe_unused]] const bool res = spx.getPrimal(sol);
  LUCID_ASSERT(res, "error getting solution");
  const Vector solution{
      Vector::NullaryExpr(fxn_lattice.cols(), [&sol](const Index i) { return sol[static_cast<int>(i)]; })};
  cb(true, spx.objValueReal(), solution, sol[eta], sol[c], solution.norm());
  return true;
}
#else
bool SoplexOptimiser::solve_fourier_barrier_synthesis_impl(const FourierBarrierSynthesisProblem&,
                                                           const SolutionCallback&) const {
  LUCID_NOT_SUPPORTED_MISSING_BUILD_DEPENDENCY("SoplexOptimiser::solve_fourier_barrier_synthesis_impl", "SoPlex");
}
#endif  // LUCID_SOPLEX_BUILD

std::string SoplexOptimiser::to_string() const {
  return fmt::format("SoplexOptimiser(problem_log_file( {} ) iis_log_file( {} ) )", problem_log_file_, iis_log_file_);
}

std::ostream& operator<<(std::ostream& os, const SoplexOptimiser& optimiser) { return os << optimiser.to_string(); }

}  // namespace lucid
