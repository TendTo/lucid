/**
 * @author c3054737
 * @copyright 2025 lucid
 * @licence BSD 3-Clause License
 * @file
 * HexalyOptimiser class.
 */
#include "lucid/verification/HexalyOptimiser.h"

#include <gurobi_c.h>

#include <iostream>

#include "lucid/lib/hexaly.h"
#include "lucid/util/logging.h"

#ifndef NLOG
#define LUCID_MODEL_ADD_CONSTRAINT(model, expr, name, should_log) \
  do {                                                            \
    if (should_log) {                                             \
      hexaly::HxExpression _exp{expr};                            \
      _exp.setName(name);                                         \
      (model).addConstraint(_exp);                                \
    } else {                                                      \
      (model).addConstraint(expr);                                \
    }                                                             \
  } while (0)
#else
#define LUCID_MODEL_ADD_CONSTRAINT(model, expr, name, should_log) (model).addConstraint(expr);
#endif

namespace lucid {

#ifdef LUCID_HEXALY_BUILD

namespace {
#ifdef LUCID_PYTHON_BUILD
class PyInterruptCallback final : public hexaly::HxCallback {
 public:
  void callback(hexaly::HexalyOptimizer&, hexaly::HxCallbackType) override { py_check_signals(); }
};
#endif
}  // namespace

bool HexalyOptimiser::solve_fourier_barrier_synthesis_impl(const FourierBarrierSynthesisParameters& params,
                                                           const SolutionCallback& cb) const {
  const auto& [num_vars, num_constraints, fx_lattice, fxp_lattice, fx0_lattice, fxu_lattice, T, gamma_val, C, b_kappa,
               fctr1, fctr2, unsafe_rhs, kushner_rhs] = params;
  static_assert(Matrix::IsRowMajor, "Row major order is expected to avoid copy/eval");
  static_assert(std::remove_reference_t<ConstMatrixRef>::IsRowMajor, "Row major order is expected to avoid copy/eval");
  constexpr double min_num = 1e-8;  // Minimum variable value for numerical stability
  constexpr double max_num = std::numeric_limits<double>::max();
  constexpr double min_eta = 0;

  hexaly::HexalyOptimizer optimizer;
  optimizer.getParam().setVerbosity(LUCID_TRACE_ENABLED ? 2 : LUCID_DEBUG_ENABLED ? 1 : 0);
  optimizer.getParam().setTimeLimit(10000);
  // model.set(GRB_DoubleParam_FeasibilityTol, 1e-9);
#ifdef LUCID_PYTHON_BUILD
  PyInterruptCallback callback;
  optimizer.getParam().setTimeBetweenTicks(2);  // seconds
  optimizer.addCallback(hexaly::CT_TimeTicked, &callback);
#endif

  hexaly::HxModel model = optimizer.getModel();

  // Specify constraints
  // Variables [b_1, ..., b_nBasis_x, c, eta, minX0, maxXU, maxXX, minDelta] in the verification case
  // Variables [b_1, ..., b_nBasis_x, c, eta, ...
  // SAT(x_1,u_1), ..., SAT(x_n_X,u1), SAT(x_1,u_n_USUpp), ..., SAT(x_n_X,u_n_USUpp), ...
  // SATOR(x_1), ..., SATOR(x_n_X)] in the control case
  const bool should_log = should_log_problem();
  std::vector<hexaly::HxExpression> vars;
  vars.reserve(static_cast<std::size_t>(num_vars));

  // Variables related to the feature map
  for (Dimension i = 0; i < fx_lattice.cols(); ++i) {
    vars.emplace_back(model.floatVar(-max_num, max_num));
#ifndef NLOG
    if (should_log) vars.back().setName(fmt::format("b[{}]", i));
#endif
  }
  vars.emplace_back(model.floatVar(0.0, max_num));  // c
#ifndef NLOG
  if (should_log) vars.back().setName("c");
#endif
  vars.emplace_back(model.floatVar(min_eta, gamma_val - min_num));  // eta
#ifndef NLOG
  if (should_log) vars.back().setName("eta");
#endif
  vars.emplace_back(model.floatVar(-max_num, max_num));  // minDelta
#ifndef NLOG
  if (should_log) vars.back().setName("minDelta");
#endif
  for (const auto& var : std::array{"minX0", "maxXU", "maxXX"}) {
    vars.emplace_back(model.floatVar(0.0, max_num));
#ifndef NLOG
    if (should_log) vars.back().setName(var);
#endif
  }

  hexaly::HxExpression& c = vars[vars.size() - 6];
  hexaly::HxExpression& eta = vars[vars.size() - 5];
  hexaly::HxExpression& minDelta = vars[vars.size() - 4];
  hexaly::HxExpression& minX0 = vars[vars.size() - 3];
  hexaly::HxExpression& maxXU = vars[vars.size() - 2];
  hexaly::HxExpression& maxXX = vars[vars.size() - 1];

  // To obtain only positive safety probabilities, restrict
  // eta + c*T in [0, gamma]
  // 1) eta + c*T >= 0 by design
  // 2) eta + c*T <= gamma
  LUCID_DEBUG("Restricting safety probabilities to be positive");
  LUCID_MODEL_ADD_CONSTRAINT(model, eta + T * c <= gamma_val, "eta+c*T<=gamma", should_log);

  LUCID_DEBUG_FMT(
      "Positive barrier - {} constraints\n"
      "for all x: [ B(x) >= hatxi ] AND [ B(x) <= maxXX ]\n"
      "hatxi = (C - 1) / (C + 1) * maxXX",
      fx_lattice.rows() * 2);
  for (Index row = 0; row < fx_lattice.rows(); ++row) {
    hexaly::HxExpression expr{0 * vars.front()};
    for (std::size_t i = 0; i < vars.size(); ++i) expr = expr + vars[i] * fx_lattice.row(row).data()[i];

    LUCID_MODEL_ADD_CONSTRAINT(model, expr + maxXX * -fctr2 >= 0, fmt::format("B(x)>=hatxi[{}]", row), should_log);
    LUCID_MODEL_ADD_CONSTRAINT(model, expr - maxXX <= 0, fmt::format("B(x)<=maxXX[{}]", row), should_log);
  }

  LUCID_DEBUG_FMT(
      "Initial constraints - {} constraints\n"
      "for all x_0: [ B(x_0) <= hateta ] AND [ B(x_0) >= minX0 ]\n"
      "hateta = 2 / (C + 1) * eta + (C - 1) / (C + 1) * minX0",
      fx0_lattice.rows() * 2);
  for (Index row = 0; row < fx0_lattice.rows(); ++row) {
    hexaly::HxExpression expr{0 * vars.front()};
    for (std::size_t i = 0; i < vars.size(); ++i) expr = expr + vars[i] * fx0_lattice.row(row).data()[i];

    LUCID_MODEL_ADD_CONSTRAINT(model, expr + -fctr1 * eta - fctr2 * minX0 <= 0, fmt::format("B(x_0)<=hateta[{}]", row),
                               should_log);
    LUCID_MODEL_ADD_CONSTRAINT(model, expr - minX0 >= 0, fmt::format("B(x_0)>=minX0[{}]", row), should_log);
  }

  LUCID_DEBUG_FMT(
      "Unsafe constraints - {} constraints\n"
      "for all x_u: [ B(x_u) >= hatgamma ] AND [ B(x_u) <= maxXU ]\n"
      "hatgamma = 2 / (C + 1) * gamma + (C - 1) / (C + 1) * maxXU",
      fxu_lattice.rows() * 2);
  for (Index row = 0; row < fxu_lattice.rows(); ++row) {
    hexaly::HxExpression expr{0 * vars.front()};
    for (std::size_t i = 0; i < vars.size(); ++i) expr = expr + vars[i] * fxu_lattice.row(row).data()[i];

    LUCID_MODEL_ADD_CONSTRAINT(model, expr - fctr2 >= unsafe_rhs, fmt::format("B(x_u)>=hatgamma[{}]", row), should_log);
    LUCID_MODEL_ADD_CONSTRAINT(model, expr - maxXU <= 0, fmt::format("B(x_u)<=maxXU[{}]", row), should_log);
  }

  LUCID_DEBUG_FMT(
      "Kushner constraints (verification case) - {} constraints\n"
      "for all x: [ B(xp) - B(x) <= hatDelta ] AND [ B(x) >= minDelta ]\n"
      "hatDelta = 2 / (C + 1) * (c - epsilon*Bnorm*kappa_x) + (C - 1) / (C + 1) * minDelta",
      fx_lattice.rows() * 2);
  const Matrix mult{fxp_lattice - b_kappa * fx_lattice};
  for (Index row = 0; row < mult.rows(); ++row) {
    hexaly::HxExpression expr{0 * vars.front()};
    for (std::size_t i = 0; i < vars.size(); ++i) expr = expr + vars[i] * mult.row(row).data()[i];

    // TODO(tend): c − εB̄κ
    LUCID_MODEL_ADD_CONSTRAINT(model, expr - fctr1 * c - fctr2 * minDelta <= kushner_rhs,
                               fmt::format("B(xp)-B(x)<=hatDelta[{}]", row), should_log);
    LUCID_MODEL_ADD_CONSTRAINT(model, expr - minDelta >= 0, fmt::format("B(xp)-B(x)>=minDelta[{}]", row), should_log);
  }

  // Objective function (η + cT)
  hexaly::HxExpression obj{eta + c * T};
  model.minimize(obj);
  model.close();

  std::cout << "--- HEXALY Model ---" << std::endl;
  std::cout << model.toString() << std::endl;
  std::cout << "--- ------------ ---" << std::endl;

  if (!problem_log_file_.empty()) optimizer.saveEnvironment(problem_log_file_);
  LUCID_INFO("Optimizing");
  optimizer.solve();

  if (optimizer.getSolution().getStatus() != hexaly::HxSolutionStatus::SS_Optimal) {
    hexaly::HxInconsistency iis{optimizer.computeInconsistency()};
    LUCID_INFO_FMT("No solution found, optimization status = {}",
                   static_cast<int>(optimizer.getSolution().getStatus()));
    if (!iis_log_file_.empty()) {
      std::ofstream file{iis_log_file_};
      file << iis.toString();
      file.close();
    }
    cb(false, 0, Vector{}, 0, 0, 0);
    return false;
  }

  hexaly::HxSolution result = optimizer.getSolution();
  const Vector solution{Vector::NullaryExpr(
      fx_lattice.cols(), [&vars, &result](const Index i) { return result.getDoubleValue(vars[i]); })};
  cb(true, result.getDoubleValue(obj), solution, result.getDoubleValue(eta), result.getDoubleValue(c), solution.norm());
  return true;
}

#else

bool HexalyOptimiser::solve_fourier_barrier_synthesis_impl(const FourierBarrierSynthesisParameters&,
                                                           const SolutionCallback&) const {
  LUCID_NOT_SUPPORTED_MISSING_BUILD_DEPENDENCY("HexalyOptimiser::solve_fourier_barrier_synthesis", "Hexaly");
  return false;
}

#endif

}  // namespace lucid