/**
 * @author Room 6.030
 * @copyright 2025 lucid
 * @licence BSD 3-Clause License
 * @file
 * HighsOptimiser class.
 */
#include "lucid/verification/HighsOptimiser.h"

#include <limits>
#include <map>
#include <ostream>
#include <sstream>
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

HighsOptimiser::HighsOptimiser(std::map<std::string, std::string> options, std::string problem_log_file,
                               std::string iis_log_file)
    : Optimiser{std::move(problem_log_file), std::move(iis_log_file)}, options_{std::move(options)} {}

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
    LUCID_ASSERT(coeffs.size() == model_.lp_.col_cost_.size() - FourierBarrierSynthesisProblem::num_extra_vars,
                 "The number of coeffs must match the number of variables.");
    LUCID_ASSERT(coeffs.size() + additional_coeffs.size() <= model_.lp_.col_cost_.size(),
                 "The number of coeffs and additional coeffs must not exceed the tot number of variables.");
    LUCID_ASSERT(coeffs.size() + additional_coeffs.size() <= model_.lp_.col_cost_.size(),
                 "The number of coeffs and additional coeffs must not exceed the tot number of coefficients.");

    // Make sure the vars contain the original rkhs variables and the additional variables
    const Dimension new_row_idx = static_cast<Dimension>(model_.lp_.row_lower_.size());
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

    const Dimension new_row_idx = static_cast<Dimension>(model_.lp_.row_lower_.size());
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
    return add_constraint<Op>(
        std::span(obj_coeffs_ptr, model_.lp_.col_cost_.size() - FourierBarrierSynthesisProblem::num_extra_vars),
        additional_vars, additional_coeffs, rhs, std::move(name));
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

  template <char Op>
    requires(Op == '<' || Op == '>')
  void add_min_max_bounds(ConstMatrixRef lattice, const std::vector<Index>& mask, const Dimension var,
                          [[maybe_unused]] const std::string& set_name) {
    LUCID_DEBUG_FMT(
        "Xn/{} lattice constraints - {} constraints\n"
        "for all x in Xn/{}: [ B(x) {}= {}_Xn/{} ]",
        set_name, mask.size(), set_name, Op, Op == '<' ? "max" : "min", set_name);
    for (Index row : mask) {
      add_constraint<Op>(
          lattice.row(row).data(), std::array{var}, std::array{-1.0}, 0.0,
          should_log_ ? fmt::format("B(Xn/{0})>={1}_Xn/{0}[{2}]", set_name, Op == '<' ? "max" : "min", row) : "");
    }
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
};
}  // namespace

bool HighsOptimiser::solve_fourier_barrier_synthesis_impl(const FourierBarrierSynthesisProblem& problem,
                                                          const SolutionCallback& cb) const {
  static_assert(Matrix::IsRowMajor, "Row major order is expected to avoid copy/eval");
  static_assert(std::remove_reference_t<ConstMatrixRef>::IsRowMajor, "Row major order is expected to avoid copy/eval");

  constexpr double max_num = std::numeric_limits<double>::infinity();
  constexpr Dimension num_special_vars = 10;  // Additional variables, e.g., c, eta, minX0, maxXU, maxXX, minDelta
  const auto& [num_constraints, fxn_lattice, dn_lattice, x_include_mask, x_exclude_mask, x0_include_mask,
               x0_exclude_mask, xu_include_mask, xu_exclude_mask, T, gamma, eta_coeff, min_x0_coeff, diff_sx0_coeff,
               gamma_coeff, max_xu_coeff, diff_sxu_coeff, ebk, c_ebk_coeff, min_d_coeff, diff_d_sx_coeff, max_x_coeff,
               diff_sx_coeff] = problem;

#ifndef NLOG
  const bool should_log = should_log_problem();
#else
  constexpr bool should_log = false;
#endif

  HighsLpProblem lp_problem{static_cast<int>(fxn_lattice.cols() + num_special_vars), num_constraints,
                            should_log_problem()};
  std::array<Dimension, FourierBarrierSynthesisProblem::num_extra_vars> special_vars;
  for (std::size_t i = 0; i < special_vars.size(); ++i) {
    special_vars[i] = static_cast<Dimension>(fxn_lattice.cols() + i);
  }
  const auto [c, eta, min_x0, max_sx0, max_xu, min_sxu, max_x, min_sx, min_d, max_d_sx] = special_vars;

  if (should_log) {
    lp_problem.set_var_name(c, "c");
    lp_problem.set_var_name(eta, "eta");
    lp_problem.set_var_name(min_x0, "minX0");
    lp_problem.set_var_name(max_sx0, "maxSX0");
    lp_problem.set_var_name(max_xu, "maxXu");
    lp_problem.set_var_name(min_sxu, "minSXu");
    lp_problem.set_var_name(max_x, "maxX");
    lp_problem.set_var_name(min_sx, "minSX");
    lp_problem.set_var_name(min_d, "minDelta");
    lp_problem.set_var_name(max_d_sx, "maxDelta_sX");
  }

  // Variables related to the feature map
  // -inf <= b_i <= inf
  for (int var = 0; var < fxn_lattice.cols(); ++var) lp_problem.set_bounds(var);
  // 0 <= c <= inf
  lp_problem.set_bounds(c, 0, max_num);
  // 0 <= eta < gamma | To enforce a strict inequality, we sub a small number from gamma
  lp_problem.set_bounds(eta, 0, gamma - FourierBarrierSynthesisProblem::tolerance);  // To enforce a strict inequality
  // 0 <= var <= inf
  for (Dimension var : std::array{min_x0, max_xu, max_x}) lp_problem.set_bounds(var, 0, max_num);
  // -inf <= var <= inf
  for (Dimension var : std::array{min_d, max_sx0, min_sxu, min_sx, max_d_sx}) {
    lp_problem.set_bounds(var);
  }

  // To obtain only positive safety probabilities, restrict
  // eta + c*T in [0, gamma]
  // 1) eta + c*T >= 0 by design
  // 2) eta + c*T <= gamma
  LUCID_DEBUG("Restricting safety probabilities to be positive");
  lp_problem.add_constraint<'<'>(std::array{eta, c}, std::array{1.0, static_cast<double>(T)}, gamma, "eta+c*T<=gamma");

  LUCID_DEBUG_FMT(
      "X0 lattice constraints - {} constraints\n"
      "for all x_0: [ B(x_0) >= min_X0] AND [ B(x_0) <= hateta ]\n"
      "hateta = eta_coeff * eta + min_x0_coeff * min_X0 - diff_sx0_coeff * max_sx0\n"
      "hateta = {} * eta + {} * min_X0 - {} * max_sx0",
      x0_include_mask.size() * 2, eta_coeff, min_x0_coeff, diff_sx0_coeff);
  for (Index row : x0_include_mask) {
    // B(x) >= min_x0
    lp_problem.add_constraint<'>'>(fxn_lattice.row(row).data(), std::array{min_x0}, std::array{-1.0}, 0.0,
                                   LUCID_FORMAT_NAME(should_log_problem(), "B(x_0)>=min_X0[{}]", row));
    // B(x) <= hateta
    lp_problem.add_constraint<'<'>(fxn_lattice.row(row).data(), std::array{eta, min_x0, max_sx0},
                                   std::array{-eta_coeff, -min_x0_coeff, diff_sx0_coeff}, 0.0,
                                   LUCID_FORMAT_NAME(should_log_problem(), "B(x_0)<=hateta[{}]", row));
  }

  LUCID_DEBUG_FMT(
      "Xu lattice constraints - {} constraints\n"
      "for all x_u: [ B(x_u) <= max_Xu ] AND [ B(x_u) >= hatgamma ] \n"
      "hatgamma = gamma_coeff * gamma + max_Xu_coeff * max_Xu - diff_sxu_coeff * min_sxu\n"
      "hatgamma = {} + {} * max_Xu - {} * min_sxu",
      xu_include_mask.size() * 2, gamma_coeff * gamma, max_xu_coeff, diff_sxu_coeff);
  for (Index row : xu_include_mask) {
    // B(x_u) <= max_Xu
    lp_problem.add_constraint<'<'>(fxn_lattice.row(row).data(), std::array{max_xu}, std::array{-1.0}, 0.0,
                                   LUCID_FORMAT_NAME(should_log_problem(), "B(x_u)<=max_Xu[{}]", row));
    // B(x_u) >= hatgamma
    lp_problem.add_constraint<'>'>(fxn_lattice.row(row).data(), std::array{max_xu, min_sxu},
                                   std::array{-max_xu_coeff, diff_sxu_coeff}, gamma_coeff * gamma,
                                   LUCID_FORMAT_NAME(should_log_problem(), "B(x_u)>=hatgamma[{}]", row));
  }

  LUCID_DEBUG_FMT(
      "Kushner constraints (verification case) - {} constraints\n"
      "for all x: [ B(xp) - B(x) >= min_d ] AND [ B(xp) - B(x) <= hatDelta ] AND \n"
      "hatDelta = c_ebk_coeff * (c - ebk) + min_d_coeff * min_d - diff_d_sx_coeff * max_d_sx\n"
      "hatDelta = {} * (c - {}) + {} * min_d - {} * max_d_sx",
      x_include_mask.size() * 2, c_ebk_coeff, ebk, min_d_coeff, diff_d_sx_coeff);
  for (Index row : x_include_mask) {
    // B(x) >= min_d
    lp_problem.add_constraint<'>'>(dn_lattice.row(row).data(), std::array{min_d}, std::array{-1.0}, 0.0,
                                   LUCID_FORMAT_NAME(should_log_problem(), "B(xp)-B(x)>=min_d[{}]", row));
    // B(xp) - B(x) <= hatDelta
    lp_problem.add_constraint<'<'>(dn_lattice.row(row).data(), std::array{c, min_d, max_d_sx},
                                   std::array{-c_ebk_coeff, -min_d_coeff, diff_d_sx_coeff}, -c_ebk_coeff * ebk,
                                   LUCID_FORMAT_NAME(should_log_problem(), "B(xp)-B(x)<=hatDelta[{}]", row));
  }

  LUCID_DEBUG_FMT(
      "Positive barrier - {} constraints\n"
      "for all x: [ B(x) <= max_X ] AND [ B(x) >= hatxi ]\n"
      "hatxi = max_x_coeff * max_X - diff_sx_coeff * min_sx\n"
      "hatxi = {} * max_X - {} * min_sx",
      x_include_mask.size() * 2, max_x_coeff, diff_sx_coeff);
  for (Index row : x_include_mask) {
    // B(x) <= max_x
    lp_problem.add_constraint<'<'>(fxn_lattice.row(row).data(), std::array{max_x}, std::array{-1.0}, 0.0,
                                   LUCID_FORMAT_NAME(should_log_problem(), "B(x)<=max_x[{}]", row));
    // B(x) >= hatxi
    lp_problem.add_constraint<'>'>(fxn_lattice.row(row).data(), std::array{max_x, min_sx},
                                   std::array{-max_x_coeff, diff_sx_coeff}, 0.0,
                                   LUCID_FORMAT_NAME(should_log_problem(), "B(x)>=hatxi[{}]", row));
  }

  lp_problem.add_min_max_bounds<'<'>(fxn_lattice, x0_exclude_mask, max_sx0, "X0");
  lp_problem.add_min_max_bounds<'>'>(fxn_lattice, xu_exclude_mask, min_sxu, "Xu");
  lp_problem.add_min_max_bounds<'>'>(fxn_lattice, x_exclude_mask, min_sx, "X");
  lp_problem.add_min_max_bounds<'<'>(dn_lattice, x_exclude_mask, max_d_sx, "dX");

  // Objective function (cT + n)
  lp_problem.set_min_objective(std::array{c, eta}, std::array{static_cast<double>(T) / gamma, 1.0 / gamma});
  lp_problem.consolidate();
  LUCID_INFO("Optimizing");

  Highs highs{};
#ifdef LUCID_PYTHON_BUILD
  highs.setCallback(interrupt_callback);
  highs.startCallback(HighsCallbackType::kCallbackSimplexInterrupt);
#endif
  [[maybe_unused]] HighsStatus ret = highs.passModel(lp_problem.model());
  LUCID_ASSERT(ret != HighsStatus::kError, "Failed to pass the model to HiGHS");
  ret = highs.setOptionValue("time_limit", 10000);
  LUCID_ASSERT(ret != HighsStatus::kError, "Failed to set the time limit option in HiGHS");
  ret = highs.setOptionValue("primal_feasibility_tolerance", 1e-9);
  LUCID_ASSERT(ret != HighsStatus::kError, "Failed to set the primal feasibility tolerance option in HiGHS");
  ret = highs.setOptionValue("log_to_console", LUCID_DEBUG_ENABLED);
  LUCID_ASSERT(ret != HighsStatus::kError, "Failed to set the log to console option in HiGHS");
  for (const auto& [key, value] : options_) highs.setOptionValue(key, value);

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
  const Vector solution{Vector::NullaryExpr(fxn_lattice.cols(), [&sol](const Index i) { return sol.col_value[i]; })};
  cb(true, info.objective_function_value, solution, sol.col_value[eta], sol.col_value[c], solution.norm());
  return true;
}
#else
bool HighsOptimiser::solve_fourier_barrier_synthesis_impl(const FourierBarrierSynthesisProblem&,
                                                          const SolutionCallback&) const {
  LUCID_NOT_SUPPORTED_MISSING_BUILD_DEPENDENCY("HighsOptimiser::solve_fourier_barrier_synthesis_impl", "HiGHS");
}
#endif  // LUCID_HIGHS_BUILD

std::string HighsOptimiser::to_string() const {
  return fmt::format("HighsOptimiser(problem_log_file( {} ) iis_log_file( {} ) )", problem_log_file_, iis_log_file_);
}

std::ostream& operator<<(std::ostream& os, const HighsOptimiser& optimiser) { return os << optimiser.to_string(); }

}  // namespace lucid
