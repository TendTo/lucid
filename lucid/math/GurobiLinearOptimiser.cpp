/**
 * @author c3054737
 * @copyright 2025 lucid
 * @licence BSD 3-Clause License
 * @file
 */
#include "lucid/math/GurobiLinearOptimiser.h"

#include <span>

#include "lucid/lib/gurobi.h"
#include "lucid/util/error.h"
#include "lucid/util/logging.h"

namespace lucid {

bool GurobiLinearOptimiser::solve(ConstMatrixRef f0_lattice, ConstMatrixRef fu_lattice, ConstMatrixRef phi_mat,
                                  ConstMatrixRef w_mat, const Dimension rkhs_dim,
                                  const Dimension num_frequencies_per_dim,
                                  const Dimension num_frequency_samples_per_dim, const Dimension original_dim,
                                  const SolutionCallback& cb) {
  constexpr double min_num = 0;  // %1e-13; % Minimum variable value for numerical stability
  constexpr double max_num = std::numeric_limits<double>::infinity();
  constexpr double min_eta = 0;
  const double C = pow((1 - 2.0 * num_frequencies_per_dim / num_frequency_samples_per_dim), -original_dim / 2.0);
  // What if we make C as big as it can be?
  // const double C = pow((1 - 2.0 * num_freq_per_dim / (2.0 * num_freq_per_dim + 1)), -original_dim / 2.0);
  LUCID_DEBUG_FMT("C: {}", C);

  GRBEnv env{true};
  env.start();
  GRBModel model{env};
  model.set(GRB_IntAttr_ModelSense, GRB_MINIMIZE);
  model.set(GRB_DoubleParam_FeasibilityTol, 1e-9);
  model.set(GRB_DoubleParam_TimeLimit, 10000);
  model.set(GRB_IntParam_OutputFlag, 0);

  // Specify constraints
  // Variables [b_1, ..., b_nBasis_x, c, eta, minX0, maxXU, maxXX, minDelta] in the verification case
  // Variables [b_1, ..., b_nBasis_x, c, eta, ...
  // SAT(x_1,u_1), ..., SAT(x_n_X,u1), SAT(x_1,u_n_USUpp), ..., SAT(x_n_X,u_n_USUpp), ...
  // SATOR(x_1), ..., SATOR(x_n_X)] in the control case
  int nVars = static_cast<int>(rkhs_dim + 2 + 4);
  std::unique_ptr<GRBVar> vars_{model.addVars(nVars)};
  const std::span<GRBVar> vars{vars_.get(), static_cast<std::size_t>(nVars)};
  GRBVar& c = vars[vars.size() - 6];
  GRBVar& eta = vars[vars.size() - 5];
  GRBVar& minX0 = vars[vars.size() - 4];
  GRBVar& maxXU = vars[vars.size() - 3];
  GRBVar& maxXX = vars[vars.size() - 2];
  GRBVar& minDelta = vars[vars.size() - 1];

  // Variables related to the feature map [0, 71)
  for (GRBVar& var : vars.subspan(0, rkhs_dim)) {
    var.set(GRB_DoubleAttr_LB, -max_num);
    var.set(GRB_DoubleAttr_UB, max_num);
  }
  c.set(GRB_DoubleAttr_LB, 0);
  c.set(GRB_DoubleAttr_UB, max_num);
  eta.set(GRB_DoubleAttr_LB, min_eta);
  eta.set(GRB_DoubleAttr_UB, gamma_ - min_num);
  for (GRBVar& var : std::array{minX0, maxXU, maxXX, minDelta}) {
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
  model.addConstr(eta + T_ * c, GRB_LESS_EQUAL, gamma_);

  LUCID_DEBUG(
      "Positive barrier\n"
      "for all x: [ B(x) >= hatxi ] AND [ B(x) <= maxXX ]\n"
      "hatxi = (C - 1) / (C + 1) * maxXX");
  for (Index row = 0; row < phi_mat.rows(); ++row) {
    GRBLinExpr expr{};
    expr.addTerms(phi_mat.row(row).eval().data(), vars_.get(), static_cast<int>(phi_mat.cols()));
    expr += maxXX * maxXX_coeff;
    model.addConstr(expr, GRB_GREATER_EQUAL, 0);
    expr -= maxXX * maxXX_coeff;
    expr += -maxXX;
    model.addConstr(expr, GRB_LESS_EQUAL, 0);
  }

  LUCID_DEBUG(
      "Initial constraints\n"
      "for all x_0: [ B(x_0) <= hateta ] AND [ B(x_0) >= minX0 ]\n"
      "hateta = 2 / (C + 1) * eta + (C - 1) / (C + 1) * minX0");
  for (Index row = 0; row < f0_lattice.rows(); ++row) {
    GRBLinExpr expr{};
    expr.addTerms(f0_lattice.row(row).eval().data(), vars_.get(), static_cast<int>(f0_lattice.cols()));
    expr += -fctr1 * eta - fctr2 * minX0;
    model.addConstr(expr, GRB_LESS_EQUAL, 0);
    expr -= -fctr1 * eta - fctr2 * minX0;
    expr += -minX0;
    model.addConstr(expr, GRB_GREATER_EQUAL, 0);
  }

  LUCID_DEBUG(
      "Unsafe constraints\n"
      "for all x_u: [ B(x_u) >= hatgamma ] AND [ B(x_u) <= maxXU ]\n"
      "hatgamma = 2 / (C + 1) * gamma + (C - 1) / (C + 1) * maxXU");
  for (Index row = 0; row < fu_lattice.rows(); ++row) {
    GRBLinExpr expr{};
    expr.addTerms(fu_lattice.row(row).eval().data(), vars_.get(), static_cast<int>(fu_lattice.cols()));
    expr += -fctr2 * maxXU;
    model.addConstr(expr, GRB_GREATER_EQUAL, unsafe_rhs);
    expr -= -fctr2 * maxXU;
    expr += -maxXU;
    model.addConstr(expr, GRB_LESS_EQUAL, 0);
  }

  LUCID_DEBUG(
      "Kushner constraints (verification case)\n"
      "for all x: [ B_plus(x) - B(x) <= hatDelta ] AND [ B(x) >= minDelta ]\n"
      "hatDelta = 2 / (C + 1) * (c - epsilon*Bnorm*kappa_x) + (C - 1) / (C + 1) * minDelta");
  auto mult = w_mat - b_kappa_ * phi_mat;
  for (Index row = 0; row < w_mat.rows(); ++row) {
    GRBLinExpr expr{};
    expr.addTerms(mult.row(row).eval().data(), vars_.get(), static_cast<int>(mult.cols()));
    expr += -fctr1 * c - fctr2 * minDelta;
    model.addConstr(expr, GRB_LESS_EQUAL, kushner_rhs);
  }
  for (Index row = 0; row < phi_mat.rows(); ++row) {
    GRBLinExpr expr{-minDelta};
    expr.addTerms(phi_mat.row(row).eval().data(), vars_.get(), static_cast<int>(phi_mat.cols()));
    model.addConstr(expr, GRB_GREATER_EQUAL, 0);
  }

  // Objective function
  model.setObjective(GRBLinExpr{T_ / gamma_ * c + 1 / gamma_ * eta});

  LUCID_INFO("Optimizing");
  model.optimize();

  if (model.get(GRB_IntAttr_SolCount) == 0) {
    LUCID_WARN_FMT("No solution found, optimization status = {}", model.get(GRB_IntAttr_Status));
    cb(false, 0, 0, 0, 0);
    return false;
  }

  LUCID_INFO_FMT("Solution found, objective = {}", model.get(GRB_DoubleAttr_ObjVal));
  LUCID_INFO_FMT("Satisfaction probability is {:.6f}% percent", 1 - model.get(GRB_DoubleAttr_ObjVal));

  auto solution{Vector::NullaryExpr(rkhs_dim, [&vars](Index i) { return vars[i].get(GRB_DoubleAttr_X); })};
  double actual_norm = solution.norm();
  LUCID_INFO_FMT("Actual norm: {}", actual_norm);
  if (actual_norm > b_norm_) {
    LUCID_WARN_FMT("Actual norm exceeds bound: {} > {} (diff: {})", actual_norm, b_norm_, actual_norm - b_norm_);
  }
  double eta_result = eta.get(GRB_DoubleAttr_X);
  double c_result = c.get(GRB_DoubleAttr_X);
  LUCID_INFO_FMT("eta: {}", eta_result);
  LUCID_INFO_FMT("c: {}", c_result);

  cb(true, model.get(GRB_DoubleAttr_ObjVal), eta.get(GRB_DoubleAttr_X), c.get(GRB_DoubleAttr_X), actual_norm);
  return true;
}

}  // namespace lucid
