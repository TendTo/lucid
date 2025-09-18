/**
 * @author Room 6.030
 * @copyright 2025 lucid
 * @licence BSD 3-Clause License
 * @file
 * HighsOptimiser class.
 */
#include "lucid/verification/HighsOptimiser.h"

#include <limits>
#include <string>
#include <utility>
#include <vector>

#include "lucid/lib/highs.h"
#include "lucid/util/Stats.h"
#include "lucid/util/Timer.h"
#include "lucid/util/error.h"

#ifdef LUCID_PYTHON_BUILD
#include "bindings/pylucid/interrupt.h"
#endif

#ifndef NLOG
#define LUCID_FORMAT_NAME(should_log, str, ...) (should_log) ? fmt::format(str, __VA_ARGS__) : ""
#else
#define LUCID_FORMAT_NAME(should_log, str, ...) ""
#endif

namespace lucid {

#ifdef LUCID_HIGHS_BUILD
namespace {

#ifdef LUCID_PYTHON_BUILD
void interrupt_callback(const int, const char*, const HighsCallbackDataOut*, HighsCallbackDataIn*, void*) {
  py_check_signals();
}
#endif

constexpr double infinity = std::numeric_limits<double>::infinity();
constexpr double neg_infinity = -std::numeric_limits<double>::infinity();

class HighsLpProblem {
 public:
  explicit HighsLpProblem(const Dimension num_vars, const Dimension num_constraints, const bool should_log)
      : triplets_{}, A_{num_constraints, num_vars}, model_{}, should_log_{should_log} {
    LUCID_ASSERT(num_vars > 0, "Number of variables must be greater than 0.");
    LUCID_ASSERT(num_constraints > 0, "Number of constraints must be greater than 0.");

    model_.lp_.num_col_ = static_cast<HighsInt>(num_vars);
    model_.lp_.num_row_ = static_cast<HighsInt>(num_constraints);
    model_.lp_.sense_ = ObjSense::kMinimize;
    model_.lp_.col_cost_ = std::vector<double>(num_vars, 0);
    model_.lp_.col_lower_ = std::vector<double>(num_vars, neg_infinity);
    model_.lp_.col_upper_ = std::vector<double>(num_vars, infinity);
    model_.lp_.row_lower_.reserve(num_constraints);
    model_.lp_.row_upper_.reserve(num_constraints);

    triplets_.reserve(num_vars * num_constraints);

#ifndef NLOG
    if (should_log_) {
      model_.lp_.row_names_.reserve(num_constraints);
      model_.lp_.row_names_.reserve(num_constraints);
      model_.lp_.col_names_.resize(num_vars);
      for (Dimension i = 0; i < num_vars; i++) model_.lp_.col_names_[i] = fmt::format("b[{}]", i);
    }
#endif
    LUCID_ASSERT(model_.lp_.col_cost_.size() == static_cast<std::size_t>(A_.cols()),
                 "The number of objective coefficients must match the number of variables.");
  }

  template <char Op, std::size_t N>
    requires(Op == '<' || Op == '>' || Op == '=')
  void add_constraint(std::span<const double> coeffs, const std::array<Dimension, N>& additional_vars,
                      const std::array<double, N>& additional_coeffs, const double rhs, std::string name = "") {
    LUCID_ASSERT(coeffs.size() == model_.lp_.col_cost_.size() - 6,
                 "The number of coeffs must match the number of variables.");
    LUCID_ASSERT(coeffs.size() + additional_coeffs.size() <= model_.lp_.col_cost_.size(),
                 "The number of coeffs and additional coeffs must not exceed the tot number of variables.");
    LUCID_ASSERT(coeffs.size() + additional_coeffs.size() <= model_.lp_.col_cost_.size(),
                 "The number of coeffs and additional coeffs must not exceed the tot number of coefficients.");

    // Make sure the vars contain the original rkhs variables and the additional variables
    const Dimension new_row_idx = model_.lp_.row_lower_.size();
    LUCID_ASSERT(new_row_idx < A_.rows(),
                 "The number of constraints must not exceed the number of rows in the matrix.");
    for (std::size_t i = 0; i < coeffs.size(); i++) {
      LUCID_ASSERT(i < model_.lp_.col_cost_.size(), "Variable index must be less than the number of variables.");
      triplets_.emplace_back(new_row_idx, static_cast<Dimension>(i), coeffs[i]);
    }
    add_constraint<Op>(additional_vars, additional_coeffs, rhs, std::move(name));
  }

  template <char Op, std::size_t N>
    requires(Op == '<' || Op == '>' || Op == '=')
  void add_constraint(const std::array<Dimension, N>& vars, const std::array<double, N>& coeffs, const double rhs,
                      [[maybe_unused]] std::string name = "") {
#ifndef NLOG
    if (should_log_) model_.lp_.row_names_.emplace_back(std::move(name));
#endif

    const Dimension new_row_idx = model_.lp_.row_lower_.size();
    LUCID_ASSERT(new_row_idx < A_.rows(),
                 "The number of constraints must not exceed the number of rows in the matrix.");

    for (std::size_t i = 0; i < vars.size(); i++) {
      LUCID_ASSERT(vars[i] >= 0, "Variable index must be non-negative.");
      LUCID_ASSERT(vars[i] < A_.cols(), "Variable index must be less than the number of columns in the matrix.");
      triplets_.emplace_back(new_row_idx, vars[i], coeffs[i]);
    }

    model_.lp_.row_lower_.push_back(Op != '<' ? rhs : neg_infinity);
    model_.lp_.row_upper_.push_back(Op != '>' ? rhs : infinity);
    LUCID_ASSERT(model_.lp_.row_upper_.size() == model_.lp_.row_lower_.size(),
                 "The number of lower and upper bounds must be equal.");
  }

  template <char Op, std::size_t N>
    requires(Op == '<' || Op == '>' || Op == '=')
  void add_constraint(const double* obj_coeffs_ptr, std::array<Dimension, N> additional_vars,
                      std::array<double, N> additional_coeffs, const double rhs, std::string name = "") {
    return add_constraint<Op>(std::span(obj_coeffs_ptr, model_.lp_.col_cost_.size() - 6), additional_vars,
                              additional_coeffs, rhs, std::move(name));
  }

  void set_bounds(const Dimension var, const double lb = neg_infinity, const double ub = infinity) {
    LUCID_ASSERT(var >= 0, "Variable index must be non-negative.");
    LUCID_ASSERT(var < static_cast<Dimension>(model_.lp_.col_lower_.size()),
                 "Variable index must be less than the number of variables.");
    LUCID_ASSERT(var < static_cast<Dimension>(model_.lp_.col_upper_.size()),
                 "Variable index must be less than the number of variables.");
    LUCID_ASSERT(lb <= ub, "Lower bound must be less than or equal to upper bound.");
    model_.lp_.col_lower_[var] = lb;
    model_.lp_.col_upper_[var] = ub;
  }

  template <std::size_t N>
  void set_min_objective(const std::array<Dimension, N>& vars, std::array<double, N> coeffs,
                         const double offset = 0.0) {
    for (std::size_t i = 0; i < vars.size(); i++) {
      LUCID_ASSERT(vars[i] >= 0, "Variable index must be non-negative.");
      LUCID_ASSERT(vars[i] < static_cast<Dimension>(model_.lp_.col_cost_.size()),
                   "Variable index must be less than the number of variables.");
      model_.lp_.col_cost_[vars[i]] = coeffs[i];
    }
    model_.lp_.offset_ = offset;
  }

  void set_var_name(const Dimension var, std::string name) {
    LUCID_ASSERT(var >= 0, "Variable index must be non-negative.");
    LUCID_ASSERT(var < static_cast<Dimension>(model_.lp_.col_names_.size()),
                 "Variable index must be less than the number of variables.");
    model_.lp_.col_names_.at(var) = std::move(name);
  }

  void consolidate() {
    LUCID_DEBUG("Consolidating");
    // copy data from eigen sparse matrix into HiGHs sparse matrix
    A_.setFromTriplets(triplets_.begin(), triplets_.end());
    model_.lp_.a_matrix_.format_ = MatrixFormat::kColwise;
    model_.lp_.a_matrix_.start_.assign(A_.outerIndexPtr(), A_.outerIndexPtr() + A_.cols());
    model_.lp_.a_matrix_.start_.push_back(static_cast<int>(A_.nonZeros()));
    model_.lp_.a_matrix_.index_.assign(A_.innerIndexPtr(), A_.innerIndexPtr() + A_.nonZeros());
    model_.lp_.a_matrix_.value_.assign(A_.valuePtr(), A_.valuePtr() + A_.nonZeros());
  }

  HighsModel& model() { return model_; }

 private:
  std::vector<Eigen::Triplet<double>> triplets_;
  Eigen::SparseMatrix<double, Eigen::ColMajor> A_;
  HighsModel model_;
  const bool should_log_ = true;
};  // namespace
}  // namespace

bool HighsOptimiser::solve(ConstMatrixRef f0_lattice, ConstMatrixRef fu_lattice, ConstMatrixRef phi_mat,
                           ConstMatrixRef w_mat, const Dimension rkhs_dim, const Dimension num_frequencies_per_dim,
                           const Dimension num_frequency_samples_per_dim, const Dimension original_dim,
                           const SolutionCallback& cb) const {
  TimerGuard tg{Stats::Scoped::top() ? &Stats::Scoped::top()->value().optimiser_timer : nullptr};
  static_assert(Matrix::IsRowMajor, "Row major order is expected to avoid copy/eval");
  static_assert(std::remove_reference_t<ConstMatrixRef>::IsRowMajor, "Row major order is expected to avoid copy/eval");
  LUCID_CHECK_ARGUMENT_CMP(num_frequency_samples_per_dim, >, 0);
  constexpr double min_num = 1e-8;  // Minimum variable value for numerical stability
  constexpr double max_num = std::numeric_limits<double>::infinity();
  constexpr double min_eta = 0;
  const double C =
      std::pow(1 - C_coeff_ * 2.0 * num_frequencies_per_dim / num_frequency_samples_per_dim, -original_dim / 2.0);
  const Dimension num_vars = rkhs_dim + 2 + 4;  // b_1, ..., b_nBasis_x, c, eta, minX0, maxXU, maxXX, minDelta
  const Dimension num_constraints = 1 + 2 * (phi_mat.rows() + f0_lattice.rows() + fu_lattice.rows() + phi_mat.rows());
  // What if we make C as big as it can be?
  // const double C = pow((1 - 2.0 * num_freq_per_dim / (2.0 * num_freq_per_dim + 1)), -original_dim / 2.0);
  LUCID_DEBUG_FMT("C: {}", C);

  if (Stats::Scoped::top()) {
    Stats::Scoped::top()->value().num_variables = num_vars;
    Stats::Scoped::top()->value().num_constraints = num_constraints;
  }

  HighsLpProblem lp_problem{num_vars, num_constraints, should_log_problem()};

  // Specify constraints
  // Variables [b_1, ..., b_nBasis_x, c, eta, minX0, maxXU, maxXX, minDelta] in the verification case
  // Variables [b_1, ..., b_nBasis_x, c, eta, ...
  // SAT(x_1,u_1), ..., SAT(x_n_X,u1), SAT(x_1,u_n_USUpp), ..., SAT(x_n_X,u_n_USUpp), ...
  // SATOR(x_1), ..., SATOR(x_n_X)] in the control case
  const Dimension c = num_vars - 6;         // Index of the c variable
  const Dimension eta = num_vars - 5;       // Index of the eta variable
  const Dimension minX0 = num_vars - 4;     // Index of the minX0 variable
  const Dimension maxXU = num_vars - 3;     // Index of the maxXU variable
  const Dimension maxXX = num_vars - 2;     // Index of the maxXX variable
  const Dimension minDelta = num_vars - 1;  // Index of the minDelta variable

#ifndef NLOG
  if (should_log_problem()) {
    lp_problem.set_var_name(c, "c");
    lp_problem.set_var_name(eta, "eta");
    lp_problem.set_var_name(minX0, "minX0");
    lp_problem.set_var_name(maxXU, "maxXU");
    lp_problem.set_var_name(maxXX, "maxXX");
    lp_problem.set_var_name(minDelta, "minDelta");
  }
#endif

  // Variables related to the feature map
  for (int var = 0; var < rkhs_dim; ++var) lp_problem.set_bounds(var);
  lp_problem.set_bounds(c, 0, max_num);
  lp_problem.set_bounds(eta, min_eta, gamma_ - min_num);  // To enforce a strict inequality
  lp_problem.set_bounds(minDelta);
  for (const Dimension var : std::array{minX0, maxXU, maxXX}) {
    lp_problem.set_bounds(var, 0, max_num);
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
  lp_problem.add_constraint<'<'>(std::array{eta, c}, std::array{1.0, static_cast<double>(T_)}, gamma_,
                                 "eta+c*T<=gamma");

  LUCID_DEBUG_FMT(
      "Positive barrier - {} constraints\n"
      "for all x: [ B(x) >= hatxi ] AND [ B(x) <= maxXX ]\n"
      "hatxi = (C - 1) / (C + 1) * maxXX",
      phi_mat.rows() * 2);
  for (Index row = 0; row < phi_mat.rows(); ++row) {
    // B(x) >= hatxi
    lp_problem.add_constraint<'>'>(phi_mat.row(row).data(), std::array{maxXX}, std::array{-fctr2}, 0.0,
                                   LUCID_FORMAT_NAME(should_log_problem(), "B(x)>=hatxi[{}]", row));
    // B(x) <= maxXX
    lp_problem.add_constraint<'<'>(phi_mat.row(row).data(), std::array{maxXX}, std::array{-1.0}, 0.0,
                                   LUCID_FORMAT_NAME(should_log_problem(), "B(x)<=maxXX[{}]", row));
  }

  LUCID_DEBUG_FMT(
      "Initial constraints - {} constraints\n"
      "for all x_0: [ B(x_0) <= hateta ] AND [ B(x_0) >= minX0 ]\n"
      "hateta = 2 / (C + 1) * eta + (C - 1) / (C + 1) * minX0",
      f0_lattice.rows() * 2);
  for (Index row = 0; row < f0_lattice.rows(); ++row) {
    // B(x_0) <= hateta
    lp_problem.add_constraint<'<'>(f0_lattice.row(row).data(), std::array{eta, minX0}, std::array{-fctr1, -fctr2}, 0.0,
                                   LUCID_FORMAT_NAME(should_log_problem(), "B(x_0)<=hateta[{}]", row));
    // B(x_0) >= minX0
    lp_problem.add_constraint<'>'>(f0_lattice.row(row).data(), std::array{minX0}, std::array{-1.0}, 0.0,
                                   LUCID_FORMAT_NAME(should_log_problem(), "B(x_0)>=minX0[{}]", row));
  }

  LUCID_DEBUG_FMT(
      "Unsafe constraints - {} constraints\n"
      "for all x_u: [ B(x_u) >= hatgamma ] AND [ B(x_u) <= maxXU ]\n"
      "hatgamma = 2 / (C + 1) * gamma + (C - 1) / (C + 1) * maxXU",
      fu_lattice.rows() * 2);
  for (Index row = 0; row < fu_lattice.rows(); ++row) {
    // B(x_u) >= hatgamma
    lp_problem.add_constraint<'>'>(fu_lattice.row(row).data(), std::array{maxXU}, std::array{-fctr2}, unsafe_rhs,
                                   LUCID_FORMAT_NAME(should_log_problem(), "B(x_u)>=hatgamma[{}]", row));
    // B(x_u) <= maxXU
    lp_problem.add_constraint<'<'>(fu_lattice.row(row).data(), std::array{maxXU}, std::array{-1.0}, 0.0,
                                   LUCID_FORMAT_NAME(should_log_problem(), "B(x_u)<=maxXU[{}]", row));
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
                                   kushner_rhs,
                                   LUCID_FORMAT_NAME(should_log_problem(), "B(xp)-B(x)<=hatDelta[{}]", row));
    // B(x) >= minDelta
    lp_problem.add_constraint<'>'>(mult.row(row).data(), std::array{minDelta}, std::array{-1.0}, 0.0,
                                   LUCID_FORMAT_NAME(should_log_problem(), "B(xp)-B(x)>=minDelta[{}]", row));
  }

  // Objective function (cT + n)
  lp_problem.set_min_objective(std::array{c, eta}, std::array{static_cast<double>(T_) / gamma_, 1.0 / gamma_});
  lp_problem.consolidate();
  LUCID_INFO("Optimizing");

  Highs highs{};
#ifdef LUCID_PYTHON_BUILD
  highs.setCallback(interrupt_callback);
  highs.startCallback(HighsCallbackType::kCallbackSimplexInterrupt);
#endif
  highs.setOptionValue("time_limit", 10000);
  highs.setOptionValue("primal_feasibility_tolerance", 1e-9);
  highs.setOptionValue("log_to_console", LUCID_DEBUG_ENABLED);
  [[maybe_unused]] HighsStatus ret = highs.passModel(lp_problem.model());
  LUCID_ASSERT(ret != HighsStatus::kError, "Failed to pass the model to HiGHS");

  if (!problem_log_file_.empty()) highs.writeModel(problem_log_file_);

  // Solve the model
  ret = highs.run();
  LUCID_ASSERT(ret != HighsStatus::kError, "Failed to run the HiGHS model");

  // Get the model status
  const HighsModelStatus& model_status = highs.getModelStatus();
  if (model_status != HighsModelStatus::kOptimal) {
    LUCID_INFO_FMT("No solution found, optimization status = {}",
                   static_cast<std::underlying_type_t<HighsModelStatus>>(model_status));
    if (!iis_log_file_.empty()) {
      HighsIis iis;
      highs.getIis(iis);
      iis.report(iis_log_file_, highs.getLp());
    }
    cb(false, 0, Vector{}, 0, 0, 0);
    return false;
  }

  const HighsInfo& info = highs.getInfo();
  LUCID_INFO_FMT("Solution found, objective = {}", info.objective_function_value);
  LUCID_INFO_FMT("Satisfaction probability is {:.6f}%", (1 - info.objective_function_value) * 100);

  const HighsSolution& sol = highs.getSolution();
  const Vector solution{Vector::NullaryExpr(rkhs_dim, [&sol](const Index i) { return sol.col_value[i]; })};
  double actual_norm = solution.norm();
  LUCID_INFO_FMT("Actual norm: {}", actual_norm);
  if (actual_norm > b_norm_) {
    LUCID_WARN_FMT("Actual norm exceeds bound: {} > {} (diff: {})", actual_norm, b_norm_, actual_norm - b_norm_);
  }

  cb(true, info.objective_function_value, solution, sol.col_value[eta], sol.col_value[c], actual_norm);
  return true;
}

bool HighsOptimiser::solve_fourier_barrier_synthesis_impl(const FourierBarrierSynthesisParameters& params,
                                                          const SolutionCallback& cb) const {
  static_assert(Matrix::IsRowMajor, "Row major order is expected to avoid copy/eval");
  static_assert(std::remove_reference_t<ConstMatrixRef>::IsRowMajor, "Row major order is expected to avoid copy/eval");
  const auto& [num_vars, num_constraints, fx_lattice, fxp_lattice, fx0_lattice, fxu_lattice, T, gamma, C, b_kappa,
               fctr1, fctr2, unsafe_rhs, kushner_rhs] = params;
  constexpr double min_num = 1e-8;  // Minimum variable value for numerical stability
  constexpr double max_num = std::numeric_limits<double>::infinity();
  constexpr double min_eta = 0;

  HighsLpProblem lp_problem{num_vars, num_constraints, should_log_problem()};

  // Specify constraints
  // Variables [b_1, ..., b_nBasis_x, c, eta, minX0, maxXU, maxXX, minDelta] in the verification case
  // Variables [b_1, ..., b_nBasis_x, c, eta, ...
  // SAT(x_1,u_1), ..., SAT(x_n_X,u1), SAT(x_1,u_n_USUpp), ..., SAT(x_n_X,u_n_USUpp), ...
  // SATOR(x_1), ..., SATOR(x_n_X)] in the control case
  const Dimension c = num_vars - 6;         // Index of the c variable
  const Dimension eta = num_vars - 5;       // Index of the eta variable
  const Dimension minX0 = num_vars - 4;     // Index of the minX0 variable
  const Dimension maxXU = num_vars - 3;     // Index of the maxXU variable
  const Dimension maxXX = num_vars - 2;     // Index of the maxXX variable
  const Dimension minDelta = num_vars - 1;  // Index of the minDelta variable

#ifndef NLOG
  if (should_log_problem()) {
    lp_problem.set_var_name(c, "c");
    lp_problem.set_var_name(eta, "eta");
    lp_problem.set_var_name(minX0, "minX0");
    lp_problem.set_var_name(maxXU, "maxXU");
    lp_problem.set_var_name(maxXX, "maxXX");
    lp_problem.set_var_name(minDelta, "minDelta");
  }
#endif

  // Variables related to the feature map
  for (int var = 0; var < fx_lattice.cols(); ++var) lp_problem.set_bounds(var);
  lp_problem.set_bounds(c, 0, max_num);
  lp_problem.set_bounds(eta, min_eta, gamma - min_num);  // To enforce a strict inequality
  lp_problem.set_bounds(minDelta);
  for (const Dimension var : std::array{minX0, maxXU, maxXX}) {
    lp_problem.set_bounds(var, 0, max_num);
  }

  // To obtain only positive safety probabilities, restrict
  // eta + c*T in [0, gamma]
  // 1) eta + c*T >= 0 by design
  // 2) eta + c*T <= gamma
  LUCID_DEBUG("Restricting safety probabilities to be positive");
  lp_problem.add_constraint<'<'>(std::array{eta, c}, std::array{1.0, static_cast<double>(T)}, gamma, "eta+c*T<=gamma");

  LUCID_DEBUG_FMT(
      "Positive barrier - {} constraints\n"
      "for all x: [ B(x) >= hatxi ] AND [ B(x) <= maxXX ]\n"
      "hatxi = (C - 1) / (C + 1) * maxXX",
      fx_lattice.rows() * 2);
  for (Index row = 0; row < fx_lattice.rows(); ++row) {
    // B(x) >= hatxi
    lp_problem.add_constraint<'>'>(fx_lattice.row(row).data(), std::array{maxXX}, std::array{-fctr2}, 0.0,
                                   LUCID_FORMAT_NAME(should_log_problem(), "B(x)>=hatxi[{}]", row));
    // B(x) <= maxXX
    lp_problem.add_constraint<'<'>(fx_lattice.row(row).data(), std::array{maxXX}, std::array{-1.0}, 0.0,
                                   LUCID_FORMAT_NAME(should_log_problem(), "B(x)<=maxXX[{}]", row));
  }

  LUCID_DEBUG_FMT(
      "Initial constraints - {} constraints\n"
      "for all x_0: [ B(x_0) <= hateta ] AND [ B(x_0) >= minX0 ]\n"
      "hateta = 2 / (C + 1) * eta + (C - 1) / (C + 1) * minX0",
      fx0_lattice.rows() * 2);
  for (Index row = 0; row < fx0_lattice.rows(); ++row) {
    // B(x_0) <= hateta
    lp_problem.add_constraint<'<'>(fx0_lattice.row(row).data(), std::array{eta, minX0}, std::array{-fctr1, -fctr2}, 0.0,
                                   LUCID_FORMAT_NAME(should_log_problem(), "B(x_0)<=hateta[{}]", row));
    // B(x_0) >= minX0
    lp_problem.add_constraint<'>'>(fx0_lattice.row(row).data(), std::array{minX0}, std::array{-1.0}, 0.0,
                                   LUCID_FORMAT_NAME(should_log_problem(), "B(x_0)>=minX0[{}]", row));
  }

  LUCID_DEBUG_FMT(
      "Unsafe constraints - {} constraints\n"
      "for all x_u: [ B(x_u) >= hatgamma ] AND [ B(x_u) <= maxXU ]\n"
      "hatgamma = 2 / (C + 1) * gamma + (C - 1) / (C + 1) * maxXU",
      fxu_lattice.rows() * 2);
  for (Index row = 0; row < fxu_lattice.rows(); ++row) {
    // B(x_u) >= hatgamma
    lp_problem.add_constraint<'>'>(fxu_lattice.row(row).data(), std::array{maxXU}, std::array{-fctr2}, unsafe_rhs,
                                   LUCID_FORMAT_NAME(should_log_problem(), "B(x_u)>=hatgamma[{}]", row));
    // B(x_u) <= maxXU
    lp_problem.add_constraint<'<'>(fxu_lattice.row(row).data(), std::array{maxXU}, std::array{-1.0}, 0.0,
                                   LUCID_FORMAT_NAME(should_log_problem(), "B(x_u)<=maxXU[{}]", row));
  }

  LUCID_DEBUG_FMT(
      "Kushner constraints (verification case) - {} constraints\n"
      "for all x: [ B(xp) - B(x) <= hatDelta ] AND [ B(x) >= minDelta ]\n"
      "hatDelta = 2 / (C + 1) * (c - epsilon*Bnorm*kappa_x) + (C - 1) / (C + 1) * minDelta",
      fx_lattice.rows() * 2);
  const Matrix mult{fxp_lattice - b_kappa * fx_lattice};
  for (Index row = 0; row < mult.rows(); ++row) {
    // B(xp) - B(x) <= hatDelta
    lp_problem.add_constraint<'<'>(mult.row(row).data(), std::array{c, minDelta}, std::array{-fctr1, -fctr2},
                                   kushner_rhs,
                                   LUCID_FORMAT_NAME(should_log_problem(), "B(xp)-B(x)<=hatDelta[{}]", row));
    // B(x) >= minDelta
    lp_problem.add_constraint<'>'>(mult.row(row).data(), std::array{minDelta}, std::array{-1.0}, 0.0,
                                   LUCID_FORMAT_NAME(should_log_problem(), "B(xp)-B(x)>=minDelta[{}]", row));
  }

  // Objective function (cT + n)
  lp_problem.set_min_objective(std::array{c, eta}, std::array{static_cast<double>(T) / gamma, 1.0 / gamma});
  lp_problem.consolidate();
  LUCID_INFO("Optimizing");

  Highs highs{};
#ifdef LUCID_PYTHON_BUILD
  highs.setCallback(interrupt_callback);
  highs.startCallback(HighsCallbackType::kCallbackSimplexInterrupt);
#endif
  highs.setOptionValue("time_limit", 10000);
  highs.setOptionValue("primal_feasibility_tolerance", 1e-9);
  highs.setOptionValue("log_to_console", LUCID_DEBUG_ENABLED);
  [[maybe_unused]] HighsStatus ret = highs.passModel(lp_problem.model());
  LUCID_ASSERT(ret != HighsStatus::kError, "Failed to pass the model to HiGHS");

  if (!problem_log_file_.empty()) highs.writeModel(problem_log_file_);

  // Solve the model
  ret = highs.run();
  LUCID_ASSERT(ret != HighsStatus::kError, "Failed to run the HiGHS model");

  // Get the model status
  const HighsModelStatus& model_status = highs.getModelStatus();
  if (model_status != HighsModelStatus::kOptimal) {
    LUCID_INFO_FMT("No solution found, optimization status = {}",
                   static_cast<std::underlying_type_t<HighsModelStatus>>(model_status));
    if (!iis_log_file_.empty()) {
      HighsIis iis;
      highs.getIis(iis);
      iis.report(iis_log_file_, highs.getLp());
    }
    cb(false, 0, Vector{}, 0, 0, 0);
    return false;
  }

  const HighsInfo& info = highs.getInfo();
  const HighsSolution& sol = highs.getSolution();
  const Vector solution{Vector::NullaryExpr(fx_lattice.cols(), [&sol](const Index i) { return sol.col_value[i]; })};
  cb(true, info.objective_function_value, solution, sol.col_value[eta], sol.col_value[c], solution.norm());
  return true;
}
#else
bool HighsOptimiser::solve(ConstMatrixRef, ConstMatrixRef, ConstMatrixRef, ConstMatrixRef, Dimension, Dimension,
                           Dimension, Dimension, const SolutionCallback&) const {
  LUCID_NOT_SUPPORTED_MISSING_BUILD_DEPENDENCY("HighsOptimiser::solve", "HiGHS");
  return false;
}
bool HighsOptimiser::solve_fourier_barrier_synthesis(const FourierBarrierSynthesisParameters&,
                                                     const SolutionCallback&) const {
  LUCID_NOT_SUPPORTED_MISSING_BUILD_DEPENDENCY("HighsOptimiser::solve_fourier_barrier_synthesis", "HiGHS");
  return false;
}
#endif  // LUCID_HIGHS_BUILD

}  // namespace lucid
