/**
 * @author Room 6.030
 * @copyright 2025 lucid
 * @licence BSD 3-Clause License
 * @file
 */
#include "lucid/verification/GurobiLinearOptimiser.h"

#include <limits>
#include <memory>
#include <span>
#include <string>
#include <utility>

#include "lucid/lib/gurobi.h"
#include "lucid/util/error.h"
#include "lucid/util/logging.h"

#ifndef NLOG
#define LUCID_MODEL_ADD_CONSTRAINT(model, expr, op, rhs, name) \
  model.addConstr(expr, op, rhs).set(GRB_StringAttr_ConstrName, name)
#else
#define LUCID_MODEL_ADD_CONSTRAINT(model, expr, op, rhs, name) model.addConstr(expr, op, rhs)
#endif

namespace lucid {

GurobiLinearOptimiser::GurobiLinearOptimiser(const int T, const double gamma, const double epsilon, const double b_norm,
                                             const double b_kappa, const double sigma_f, const double C_coeff,
                                             std::string problem_log_file, std::string iis_log_file)
    : T_{T},
      gamma_{gamma},
      epsilon_{epsilon},
      b_norm_{b_norm},
      b_kappa_{b_kappa},
      sigma_f_{sigma_f},
      C_coeff_{C_coeff},
      problem_log_file_{std::move(problem_log_file)},
      iis_log_file_{std::move(iis_log_file)} {
  LUCID_CHECK_ARGUMENT_CMP(T, >, 0);
  LUCID_CHECK_ARGUMENT_EXPECTED(
      problem_log_file_.empty() || (problem_log_file_.ends_with(".lp") || problem_log_file_.ends_with(".mps")),
      "problem_log_file", problem_log_file_, "must be a valid file path with .lp or .mps extension");
  LUCID_CHECK_ARGUMENT_EXPECTED(iis_log_file_.empty() || iis_log_file_.ends_with(".ilp"), "iis_log_file", iis_log_file_,
                                "must be a valid file path with .ilp extension");
}

#ifdef LUCID_GUROBI_BUILD
bool GurobiLinearOptimiser::solve(ConstMatrixRef f0_lattice, ConstMatrixRef fu_lattice, ConstMatrixRef phi_mat,
                                  ConstMatrixRef w_mat, const Dimension rkhs_dim,
                                  const Dimension num_frequencies_per_dim,
                                  const Dimension num_frequency_samples_per_dim, const Dimension original_dim,
                                  const SolutionCallback& cb) const {
  LUCID_CHECK_ARGUMENT_CMP(num_frequency_samples_per_dim, >, 0);
  constexpr double min_num = 1e-8;  // Minimum variable value for numerical stability
  constexpr double max_num = std::numeric_limits<double>::infinity();
  constexpr double min_eta = 0;
  const double C =
      std::pow(1 - C_coeff_ * 2.0 * num_frequencies_per_dim / num_frequency_samples_per_dim, -original_dim / 2.0);
  // What if we make C as big as it can be?
  // const double C = pow((1 - 2.0 * num_freq_per_dim / (2.0 * num_freq_per_dim + 1)), -original_dim / 2.0);
  LUCID_DEBUG_FMT("C: {}", C);

  GRBEnv env{true};
  env.set(GRB_IntParam_OutputFlag, LUCID_DEBUG_ENABLED);
  env.start();
  GRBModel model{env};
  model.set(GRB_IntAttr_ModelSense, GRB_MINIMIZE);
  model.set(GRB_DoubleParam_FeasibilityTol, 1e-9);
  model.set(GRB_DoubleParam_TimeLimit, 10000);

  // Specify constraints
  // Variables [b_1, ..., b_nBasis_x, c, eta, minX0, maxXU, maxXX, minDelta] in the verification case
  // Variables [b_1, ..., b_nBasis_x, c, eta, ...
  // SAT(x_1,u_1), ..., SAT(x_n_X,u1), SAT(x_1,u_n_USUpp), ..., SAT(x_n_X,u_n_USUpp), ...
  // SATOR(x_1), ..., SATOR(x_n_X)] in the control case
  int nVars = static_cast<int>(rkhs_dim + 2 + 4);
  std::unique_ptr<GRBVar[]> vars_{model.addVars(nVars)};
  const std::span<GRBVar> vars{vars_.get(), static_cast<std::size_t>(nVars)};
  GRBVar& c = vars[vars.size() - 6];
  GRBVar& eta = vars[vars.size() - 5];
  GRBVar& minX0 = vars[vars.size() - 4];
  GRBVar& maxXU = vars[vars.size() - 3];
  GRBVar& maxXX = vars[vars.size() - 2];
  GRBVar& minDelta = vars[vars.size() - 1];

#ifndef NLOG
  c.set(GRB_StringAttr_VarName, "c");
  eta.set(GRB_StringAttr_VarName, "eta");
  minX0.set(GRB_StringAttr_VarName, "minX0");
  maxXU.set(GRB_StringAttr_VarName, "maxXU");
  maxXX.set(GRB_StringAttr_VarName, "maxXX");
  minDelta.set(GRB_StringAttr_VarName, "minDelta");
#endif

  // Variables related to the feature map
  int idx = 0;
  for (GRBVar& var : vars.subspan(0, rkhs_dim)) {
    var.set(GRB_DoubleAttr_LB, -max_num);
    var.set(GRB_DoubleAttr_UB, max_num);
#ifndef NLOG
    var.set(GRB_StringAttr_VarName, fmt::format("b[{}]", idx++));
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
  LUCID_MODEL_ADD_CONSTRAINT(model, eta + T_ * c, GRB_LESS_EQUAL, gamma_, "eta+c*T<=gamma");

  LUCID_DEBUG_FMT(
      "Positive barrier - {} constraints\n"
      "for all x: [ B(x) >= hatxi ] AND [ B(x) <= maxXX ]\n"
      "hatxi = (C - 1) / (C + 1) * maxXX",
      phi_mat.rows() * 2);
  for (Index row = 0; row < phi_mat.rows(); ++row) {
    static_assert(phi_mat.IsRowMajor, "Row major order is expected to avoid copy/eval");
    GRBLinExpr expr{};
    expr.addTerms(phi_mat.row(row).data(), vars_.get(), static_cast<int>(phi_mat.cols()));
    expr += maxXX * maxXX_coeff;
    LUCID_MODEL_ADD_CONSTRAINT(model, expr, GRB_GREATER_EQUAL, 0, fmt::format("B(x)>=hatxi[{}]", row));
    expr.remove(maxXX);
    expr += -maxXX;
    LUCID_MODEL_ADD_CONSTRAINT(model, expr, GRB_LESS_EQUAL, 0, fmt::format("B(x)<=maxXX[{}]", row));
  }

  LUCID_DEBUG_FMT(
      "Initial constraints - {} constraints\n"
      "for all x_0: [ B(x_0) <= hateta ] AND [ B(x_0) >= minX0 ]\n"
      "hateta = 2 / (C + 1) * eta + (C - 1) / (C + 1) * minX0",
      f0_lattice.rows() * 2);
  for (Index row = 0; row < f0_lattice.rows(); ++row) {
    static_assert(f0_lattice.IsRowMajor, "Row major order is expected to avoid copy/eval");
    GRBLinExpr expr{};
    expr.addTerms(f0_lattice.row(row).data(), vars_.get(), static_cast<int>(f0_lattice.cols()));
    expr += -fctr1 * eta - fctr2 * minX0;
    LUCID_MODEL_ADD_CONSTRAINT(model, expr, GRB_LESS_EQUAL, 0, fmt::format("B(x_0)<=hateta[{}]", row));
    expr.remove(eta);
    expr.remove(minX0);
    expr += -minX0;
    LUCID_MODEL_ADD_CONSTRAINT(model, expr, GRB_GREATER_EQUAL, 0, fmt::format("B(x_0)>=minX0[{}]", row));
  }

  LUCID_DEBUG_FMT(
      "Unsafe constraints - {} constraints\n"
      "for all x_u: [ B(x_u) >= hatgamma ] AND [ B(x_u) <= maxXU ]\n"
      "hatgamma = 2 / (C + 1) * gamma + (C - 1) / (C + 1) * maxXU",
      fu_lattice.rows() * 2);
  for (Index row = 0; row < fu_lattice.rows(); ++row) {
    static_assert(fu_lattice.IsRowMajor, "Row major order is expected to avoid copy/eval");
    GRBLinExpr expr{};
    expr.addTerms(fu_lattice.row(row).data(), vars_.get(), static_cast<int>(fu_lattice.cols()));
    expr += -fctr2 * maxXU;
    LUCID_MODEL_ADD_CONSTRAINT(model, expr, GRB_GREATER_EQUAL, unsafe_rhs, fmt::format("B(x_u)>=hatgamma[{}]", row));
    expr.remove(maxXU);
    expr += -maxXU;
    LUCID_MODEL_ADD_CONSTRAINT(model, expr, GRB_LESS_EQUAL, 0, fmt::format("B(x_u)<=maxXU[{}]", row));
  }

  LUCID_DEBUG_FMT(
      "Kushner constraints (verification case) - {} constraints\n"
      "for all x: [ B(xp) - B(x) <= hatDelta ] AND [ B(x) >= minDelta ]\n"
      "hatDelta = 2 / (C + 1) * (c - epsilon*Bnorm*kappa_x) + (C - 1) / (C + 1) * minDelta",
      phi_mat.rows() * 2);
  const Matrix mult{w_mat - b_kappa_ * phi_mat};
  for (Index row = 0; row < mult.rows(); ++row) {
    static_assert(mult.IsRowMajor, "Row major order is expected to avoid copy/eval");
    GRBLinExpr expr{};
    expr.addTerms(mult.row(row).data(), vars_.get(), static_cast<int>(mult.cols()));
    expr += -fctr1 * c - fctr2 * minDelta;  // TODO(tend): c − εB̄κ
    LUCID_MODEL_ADD_CONSTRAINT(model, expr, GRB_LESS_EQUAL, kushner_rhs, fmt::format("B(xp)-B(x)<=hatDelta[{}]", row));
    expr.remove(c);
    expr.remove(minDelta);
    expr += -minDelta;
    LUCID_MODEL_ADD_CONSTRAINT(model, expr, GRB_GREATER_EQUAL, 0, fmt::format("B(xp)-B(x)>=minDelta[{}]", row));
  }

  // Objective function (η + cT)
  model.setObjective(GRBLinExpr{(eta + c * T_) / gamma_});

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
#else
bool GurobiLinearOptimiser::solve(ConstMatrixRef, ConstMatrixRef, ConstMatrixRef, ConstMatrixRef, Dimension, Dimension,
                                  Dimension, Dimension, const SolutionCallback&) const {
  LUCID_NOT_SUPPORTED_MISSING_DEPENDENCY("GurobiLinearOptimiser::solve", "Gurobi");
  return false;
}
#endif  // LUCID_GUROBI_BUILD

}  // namespace lucid
