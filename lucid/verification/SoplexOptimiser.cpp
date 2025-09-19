/**
 * @author c3054737
 * @copyright 2025 lucid
 * @licence BSD 3-Clause License
 * @file
 */
#include "lucid/verification/SoplexOptimiser.h"

#include <gurobi_c++.h>

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
bool SoplexOptimiser::solve_fourier_barrier_synthesis_impl(const FourierBarrierSynthesisParameters& params,
                                                           const SolutionCallback& cb) const {
  const auto& [num_vars, num_constraints, fx_lattice, fxp_lattice, fx0_lattice, fxu_lattice, T, gamma_val, C, b_kappa,
               fctr1, fctr2, unsafe_rhs, kushner_rhs] = params;
  static_assert(Matrix::IsRowMajor, "Row major order is expected to avoid copy/eval");
  static_assert(std::remove_reference_t<ConstMatrixRef>::IsRowMajor, "Row major order is expected to avoid copy/eval");
  constexpr double min_num = 1e-8;  // Minimum variable value for numerical stability
  constexpr double min_eta = 0;

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
  const bool should_log = should_log_problem();
  soplex::LPColSetReal cols(static_cast<int>(num_vars));
  soplex::LPRowSetReal rows(static_cast<int>(num_constraints));

#ifndef NLOG
  soplex::NameSet var_names_set{should_log ? static_cast<int>(num_vars) : 0};
  soplex::NameSet constraint_names_set{should_log ? static_cast<int>(num_constraints) : 0};
  std::vector<std::string> var_names;
  std::vector<std::string> constraint_names;
  var_names.reserve(should_log ? num_vars : 0);
  constraint_names.reserve(should_log ? num_constraints : 0);
#else
  soplex::NameSet var_names_set{};
  soplex::NameSet constraint_names_set{};
#endif

  // Variables related to the feature map
  for (Dimension i = 0; i < fx_lattice.cols(); ++i) {
    cols.add(soplex::LPColReal{0.0, soplex::DSVectorReal(), soplex::infinity, -soplex::infinity});
#ifndef NLOG
    if (should_log) var_names.emplace_back(fmt::format("b[{}]", i));
#endif
  }
  // c (Objective function (η + cT))
  const int c = cols.num();
  cols.add(soplex::LPColReal{static_cast<double>(T), soplex::DSVectorReal(), soplex::infinity, 0});
  // eta (Objective function (η + cT))
  const int eta = cols.num();
  cols.add(soplex::LPColReal{1.0, soplex::DSVectorReal(), gamma_val - min_num, min_eta});
  // minDelta
  const int minDelta = cols.num();
  cols.add(soplex::LPColReal{0.0, soplex::DSVectorReal(), soplex::infinity, -soplex::infinity});

  int minX0, maxXU, maxXX;
  for (int* var : std::array{&minX0, &maxXU, &maxXX}) {
    // minX0, maxXU, maxXX
    *var = cols.num();
    cols.add(soplex::LPColReal{0.0, soplex::DSVectorReal(), soplex::infinity, 0});
  }
#ifndef NLOG
  if (should_log) {
    var_names.emplace_back("c");
    var_names.emplace_back("eta");
    var_names.emplace_back("minDelta");
    var_names.emplace_back("minX0");
    var_names.emplace_back("maxXU");
    var_names.emplace_back("maxXX");
  }
#endif

  // To obtain only positive safety probabilities, restrict
  // eta + c*T in [0, gamma]
  // 1) eta + c*T >= 0 by design
  // 2) eta + c*T <= gamma
  {
    LUCID_DEBUG("Restricting safety probabilities to be positive");
    soplex::DSVectorReal vec(2);
    vec.add(eta, 1.0);
    vec.add(c, static_cast<double>(T));
    LUCID_MODEL_ADD_CONSTRAINT(rows, -soplex::infinity, vec, gamma_val, "eta+c*T<=gamma", constraint_names, should_log);
  }

  LUCID_DEBUG_FMT(
      "Positive barrier - {} constraints\n"
      "for all x: [ B(x) >= hatxi ] AND [ B(x) <= maxXX ]\n"
      "hatxi = (C - 1) / (C + 1) * maxXX",
      fx_lattice.rows() * 2);
  for (Index row = 0; row < fx_lattice.rows(); ++row) {
    soplex::DSVectorReal vec(static_cast<int>(fx_lattice.cols()) + 1);
    for (Index col = 0; col < fx_lattice.cols(); ++col) vec.add(static_cast<int>(col), fx_lattice(row, col));
    vec.add(maxXX, -fctr2);
    LUCID_MODEL_ADD_CONSTRAINT(rows, 0.0, vec, soplex::infinity, fmt::format("B(x)>=hatxi[{}]", row), constraint_names,
                               should_log);
    vec.value(vec.pos(maxXX)) = -1.0;
    LUCID_MODEL_ADD_CONSTRAINT(rows, -soplex::infinity, vec, 0.0, fmt::format("B(x)<=maxXX[{}]", row), constraint_names,
                               should_log);
  }

  LUCID_DEBUG_FMT(
      "Initial constraints - {} constraints\n"
      "for all x_0: [ B(x_0) <= hateta ] AND [ B(x_0) >= minX0 ]\n"
      "hateta = 2 / (C + 1) * eta + (C - 1) / (C + 1) * minX0",
      fx0_lattice.rows() * 2);
  for (Index row = 0; row < fx0_lattice.rows(); ++row) {
    soplex::DSVectorReal vec(static_cast<int>(fx0_lattice.cols()) + 2);
    for (Index col = 0; col < fx0_lattice.cols(); ++col) vec.add(static_cast<int>(col), fx0_lattice(row, col));
    vec.add(eta, -fctr1);
    vec.add(minX0, -fctr2);
    LUCID_MODEL_ADD_CONSTRAINT(rows, -soplex::infinity, vec, 0.0, fmt::format("B(x_0)<=hateta[{}]", row),
                               constraint_names, should_log);
    vec.remove(vec.pos(eta));
    vec.value(vec.pos(minX0)) = -1.0;
    LUCID_MODEL_ADD_CONSTRAINT(rows, 0.0, vec, soplex::infinity, fmt::format("B(x_0)>=minX0[{}]", row),
                               constraint_names, should_log);
  }

  LUCID_DEBUG_FMT(
      "Unsafe constraints - {} constraints\n"
      "for all x_u: [ B(x_u) >= hatgamma ] AND [ B(x_u) <= maxXU ]\n"
      "hatgamma = 2 / (C + 1) * gamma + (C - 1) / (C + 1) * maxXU",
      fxu_lattice.rows() * 2);
  for (Index row = 0; row < fxu_lattice.rows(); ++row) {
    soplex::DSVectorReal vec(static_cast<int>(fxu_lattice.cols()) + 1);
    for (Index col = 0; col < fxu_lattice.cols(); ++col) vec.add(static_cast<int>(col), fxu_lattice(row, col));
    vec.add(maxXU, -fctr2);
    LUCID_MODEL_ADD_CONSTRAINT(rows, unsafe_rhs, vec, soplex::infinity, fmt::format("B(x_u)>=hatgamma[{}]", row),
                               constraint_names, should_log);
    vec.value(vec.pos(maxXU)) = -1.0;
    LUCID_MODEL_ADD_CONSTRAINT(rows, -soplex::infinity, vec, 0.0, fmt::format("B(x_u)<=maxXU[{}]", row),
                               constraint_names, should_log);
  }

  LUCID_DEBUG_FMT(
      "Kushner constraints (verification case) - {} constraints\n"
      "for all x: [ B(xp) - B(x) <= hatDelta ] AND [ B(x) >= minDelta ]\n"
      "hatDelta = 2 / (C + 1) * (c - epsilon*Bnorm*kappa_x) + (C - 1) / (C + 1) * minDelta",
      fx_lattice.rows() * 2);
  const Matrix mult{fxp_lattice - b_kappa * fx_lattice};
  for (Index row = 0; row < mult.rows(); ++row) {
    soplex::DSVectorReal vec(static_cast<int>(mult.cols()) + 2);
    for (Index col = 0; col < fxu_lattice.cols(); ++col) vec.add(static_cast<int>(col), mult(row, col));
    vec.add(c, -fctr1);
    vec.add(minDelta, -fctr2);
    LUCID_MODEL_ADD_CONSTRAINT(rows, -soplex::infinity, vec, 0.0, fmt::format("B(xp)-B(x)<=hatDelta[{}]", row),
                               constraint_names, should_log);
    vec.remove(vec.pos(c));
    vec.value(vec.pos(minDelta)) = -1.0;
    LUCID_MODEL_ADD_CONSTRAINT(rows, 0.0, vec, soplex::infinity, fmt::format("B(xp)-B(x)>=minDelta[{}]", row),
                               constraint_names, should_log);
  }

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
  // volatile bool interrupt = false;
  // volatile bool finished = false;
  // std::thread interrupt_checker_thread([&interrupt, &finished]() {
  //   while (!finished && !interrupt) {
  //     LUCID_INFO("interrupt");
  //     py_interrupt_flag(&interrupt);
  //     if (interrupt) LUCID_ERROR("interrupt");
  //     std::this_thread::sleep_for(std::chrono::seconds(1));
  //   }
  // });
  // spx.optimize(&interrupt);
  // finished = true;
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

  soplex::VectorReal sol{static_cast<int>(num_vars)};
  [[maybe_unused]] const bool res = spx.getPrimal(sol);
  LUCID_ASSERT(res, "error getting solution");
  const Vector solution{Vector::NullaryExpr(fx_lattice.cols(), [&sol](const Index i) { return sol[static_cast<int>(i)]; })};
  cb(true, spx.objValueReal(), solution, sol[eta], sol[c], solution.norm());
  return true;
}
#else
bool SoplexOptimiser::solve_fourier_barrier_synthesis_impl(const FourierBarrierSynthesisParameters&,
                                                           const SolutionCallback&) const {
  LUCID_NOT_SUPPORTED_MISSING_BUILD_DEPENDENCY("SoplexOptimiser::solve_fourier_barrier_synthesis", "SoPlex");
  return false;
}
#endif  // LUCID_SOPLEX_BUILD

}  // namespace lucid
