/**
 * @file main.cpp
 * @author Room 6.030
 * @copyright 2025 lucid
 * @licence Apache-2.0 license
 * @file
 */
#include <fmt/core.h>
#include <gurobi_c++.h>

#include <chrono>
#include <iostream>
#include <span>
#include <string>
#include <utility>
#include <vector>

#include "lucid/lucid.h"
#include "lucid/util/logging.h"
#include "lucid/util/math.h"
#include "lucid/util/matplotlibcpp.h"
#include "util/error.h"

namespace plt = matplotlibcpp;

using namespace lucid;  // NOLINT

#pragma GCC diagnostic ignored "-Wunused-variable"
#pragma GCC diagnostic ignored "-Wunused-parameter"
#pragma GCC diagnostic ignored "-Wunused-but-set-variable"
#pragma GCC diagnostic ignored "-Wunused-function"

namespace {

using Basis = std::function<Matrix(ConstVectorRef x, ConstMatrixRef x_limits)>;

void test_surface() {
  Matrix X, Y;
  meshgrid(arange(-5, 5, 0.5), arange(-5, 5, 0.5), X, Y);
  Matrix Z = peaks(X, Y);
  plt::plot_wireframe(X, Y, Z);
}

void plot_points(const Matrix& points, const std::string& color = "blue", const double size = 10) {
  if (points.cols() != 2) throw std::invalid_argument("points must have 2 rows");
  const Vector x = points.col(0);
  const Vector y = points.col(1);
  plt::scatter(x, y, {.s = size, .c = color});
}

void test_barrier_3_old() {
  // Generic test
  const benchmark::InitBarr3Scenario scenario;
  scenario.plot();
  Matrix inputs, outputs;
  scenario.sample_transition(1000, inputs, outputs);
  plot_points(inputs, "blue");
  plot_points(outputs, "magenta");

  constexpr double regularization_constant = 0.0000001;
  const GaussianKernel kernel{1, Vector::Constant(scenario.dimension(), 0.3)};
  const KernelRidgeRegression regression{kernel, inputs, outputs, regularization_constant};

  // const Matrix samples = scenario.sample_element(10);
  // plot_points(samples, "green");
  Matrix res = regression(inputs);
  std::cout << "RMS: " << rms(res - outputs) << std::endl;
  plot_points(res, "cyan", 2);

  plt::figure(2);
  scenario.sample_transition(1000, inputs, outputs);
  scenario.plot();
  plot_points(inputs, "blue");
  plot_points(outputs, "magenta");

  res = regression(inputs);
  std::cout << res.transpose() << std::endl;
  std::cout << "RMS: " << rms(res - outputs) << std::endl;
  plot_points(res, "cyan", 2);

  plt::figure(3);
  plt::scatter(inputs.col(0), inputs.col(1), outputs.col(0));

  plt::figure(4);
  plt::scatter(static_cast<Vector>(inputs.col(0)), static_cast<Vector>(inputs.col(1)), static_cast<Vector>(res.col(0)));
  plt::show();
}

#if 0
Matrix wavelengths(const Dimension dimension, const int num_frequencies_per_dimension) {
  if (dimension < 1) throw std::invalid_argument("dimension must be at least 1");
  // Compute the total number of columns of the output matrix
  const Vector frequencies{arange(0, num_frequencies_per_dimension)};

  Matrix frequency_combination = frequencies.transpose();
  for (Dimension i = 1; i < dimension; i++) {
    frequency_combination = combvec(frequency_combination, frequencies.transpose());
  }
  return (frequency_combination.rightCols(frequency_combination.cols() - 1) * 2 * M_PI).transpose();
}

Basis generate_basis(ConstMatrixRef omega_T, const Dimension dimension, Scalar sigma_f, ConstVectorRef sigma_l,
                     const int num_frequencies_per_dimension) {
  if (sigma_l.size() < 1) throw std::invalid_argument("sigma_l must have at least one element");

  const Vector omega_dim_wise_lb = (2 * M_PI * arange(0, num_frequencies_per_dimension)).array() - M_PI;
  const Vector omega_dim_wise_ub = omega_dim_wise_lb.array() + 2 * M_PI;

  Matrix prob_dim_wise{dimension, num_frequencies_per_dimension};
  for (Dimension i = 0; i < dimension; i++) {
    prob_dim_wise.row(i) = normal_cdf(omega_dim_wise_ub, 0, sigma_l(i)) - normal_cdf(omega_dim_wise_lb, 0, sigma_l(i));
    prob_dim_wise.row(i).rightCols(prob_dim_wise.cols() - 1) *= 2;
  }

  auto prod_ND = combvec(prob_dim_wise);
  auto prod = prod_ND.colwise().prod().transpose();
  if (Scalar sum = prod.sum(); sum > 0.9)
    LUCID_INFO_FMT("Probability captured by Fourier expansion is {:.3f} percent", sum);
  else
    LUCID_WARN_FMT("Probability captured by Fourier expansion is only {:.3f} percent", sum);
  auto temp = prod.transpose().cwiseSqrt();

  Vector weights{temp.size() * 2 - 1};
  // Repeat each column twice, except the first one
  for (Index i = 0; i < temp.size(); i++) {
    weights(2 * i) = temp(i);
    if (i != 0) weights(2 * i - 1) = temp(i);
  }

  return [weights = std::move(weights), omega_T = omega_T.eval(), sigma_f](ConstVectorRef x, ConstMatrixRef x_limits) {
    // TODO(tend): this only works for rectangles sets
    // Project the element onto the unit hypercube
    auto z = (x.transpose() - x_limits.row(0)).cwiseQuotient(x_limits.row(1) - x_limits.row(0));

    Vector z_proj = omega_T * z.transpose();
    Vector trig{2 * z_proj.size() + 1};
    trig(0) = 1;
    for (Index i = 0; i < z_proj.size(); i++) {
      trig(2 * i + 1) = std::cos(z_proj(i));
      trig(2 * i + 2) = std::sin(z_proj(i));
    }

    auto basis = sigma_f * weights.cwiseProduct(trig);
    if (Scalar checksum = (basis.cwiseProduct(basis).colwise().sum().array().sqrt() - sigma_f).abs().maxCoeff();
        checksum > 1e-3) {
      LUCID_WARN_FMT("Checksum failed: Fourier basis frequency bands don't add up: {} > 1e-3", checksum);
    }
    // LUCID_ASSERT((basis.cwiseProduct(basis).colwise().sum().array().sqrt() - sigma_f).abs().maxCoeff() <= 1e-3,
    //              "Checksum failed: Fourier basis frequency bands don't add up");
    return Matrix{basis};
  };
}

Basis generate_basis(ConstMatrixRef omega_T, const Dimension dimension, const Scalar sigma_f, const Scalar sigma_l,
                     const int num_frequencies_per_dimension) {
  return generate_basis(omega_T, dimension, sigma_f, Vector::Constant(dimension, sigma_l),
                        num_frequencies_per_dimension);
}
#endif

Matrix lattice(const Matrix& x_limits, const Index samples_per_dim, bool include_endpoints = false) {
  Matrix x_lattice{1, samples_per_dim};
  if (include_endpoints) {
    x_lattice.row(0) = Vector::LinSpaced(samples_per_dim, x_limits(0, 0), x_limits(1, 0));
    for (Dimension i = 1; i < x_limits.cols(); ++i) {
      x_lattice = combvec(x_lattice, Vector::LinSpaced(samples_per_dim, x_limits(0, i), x_limits(1, i)).transpose());
    }
  } else {
    const Matrix delta_per_dim = (x_limits.row(1) - x_limits.row(0)).array() / static_cast<double>(samples_per_dim);
    x_lattice.row(0) = arange(x_limits(0, 0), x_limits(1, 0), delta_per_dim(0));
    for (Dimension i = 1; i < x_limits.cols(); ++i) {
      x_lattice = combvec(x_lattice, arange(x_limits(0, i), x_limits(1, i), delta_per_dim(i)).transpose());
    }
  }
  x_lattice.transposeInPlace();
  return x_lattice;
}

void plotting_fcn(const Matrix& XX_Fourier, const Index x_samples, const Index f_sampling, const Matrix& w_vec_Fourier,
                  const Matrix& X_limits, const Matrix& w_vec) {
  // For 2D case
  const Matrix x1_original = XX_Fourier.col(0).reshaped(x_samples, x_samples);
  const Matrix x2_original = XX_Fourier.col(1).reshaped(x_samples, x_samples);

  const Matrix XX = lattice(X_limits, f_sampling);
  const Matrix x1_reconstr = XX.col(0).reshaped(f_sampling, f_sampling);
  const Matrix x2_reconstr = XX.col(1).reshaped(f_sampling, f_sampling);

  static int fig_number = 0;
  plt::plot_wireframe(x1_original, x2_original, w_vec_Fourier, {.fig_number = fig_number});
  plt::plot_surface(x1_reconstr, x2_reconstr, w_vec.transpose(), {.fig_number = fig_number++});
}

Vector project(const Matrix& f, const Dimension dim, const Index n_per_dim, const Index samples_per_dim) {
  // TODO(tend): this only works for 2 dimensions
  const Eigen::MatrixXcd f_fft{fft2(f)};
  // std::cout << "f" << std::endl << f << std::endl;

  const int n_pad = floor((n_per_dim / 2 - samples_per_dim / 2));
  LUCID_DEBUG_FMT("n_pad: {}", n_pad);

  Eigen::MatrixXcd padded_ft{pad(fftshift(f_fft), n_pad, std::complex<double>{})};
  LUCID_DEBUG_FMT("padded_ft: {}x{}", padded_ft.rows(), padded_ft.cols());
  // std::cout << "padded_ft\n" << padded_ft << std::endl;
  // std::cout << "ifft2\n" << ifft2(ifftshift(padded_ft)) << std::endl;

  Matrix f_interp = ifft2(ifftshift(padded_ft)).array() * lucid::pow(n_per_dim / samples_per_dim, dim);
  // std::cout << "f_interp\n" << f_interp << std::endl;
  return f_interp.reshaped(Eigen::AutoSize, 1);
}

void cme_2_fourier(const benchmark::Scenario& scenario, const TruncatedFourierFeatureMap& tfm, const Kernel& kernel,
                   Matrix& w_mat, Matrix& phi_mat) {
  const auto& init_barr3 = dynamic_cast<const benchmark::InitBarr3Scenario&>(scenario);
  const auto& gaussian_kernel = dynamic_cast<const GaussianKernel&>(kernel);
  // TODO(tend): double check this. Number of num_freq... and actual number of samples are not the same in matlab
  const Dimension dim = init_barr3.dimension();
  if (dim < 1) throw std::invalid_argument("dimension must be at least 1");
  const int samples_per_dim = 2 * init_barr3.num_freq_per_dim();  // Should probably add + 1
  const Matrix x_lattice{lattice(init_barr3.x_limits(), samples_per_dim)};
  // fmt::println("Lattice: \n{}", x_fourier);

  Matrix x_lattice_fourier{tfm(x_lattice)};
  Matrix xp_fourier{tfm(init_barr3.xp_samples())};
  // std::cout << x_lattice_fourier.transpose() << std::endl;

  // fmt::println("x_lattice_fourier: {}x{}\n{}", x_lattice_fourier.rows(), x_lattice_fourier.cols(),
  // x_lattice_fourier);
  // fmt::println("xp_fourier: {}x{}\n{}", xp_fourier.rows(), xp_fourier.cols(), xp_fourier);

  const KernelRidgeRegression regression{gaussian_kernel, init_barr3.x_samples(), xp_fourier, init_barr3.lambda()};
  Matrix w_vec_fourier = regression(x_lattice);
  LUCID_DEBUG_FMT("w_vec_fourier size: {}x{}", w_vec_fourier.rows(), w_vec_fourier.cols());

  int factor = std::ceil(init_barr3.num_supp_per_dim() / static_cast<double>(samples_per_dim)) + 1;
  LUCID_DEBUG_FMT("factor: {}", factor);
  int n_per_dim = factor * samples_per_dim;
  LUCID_DEBUG_FMT("n_per_dim: {}", n_per_dim);

  w_mat = Matrix::Zero(lucid::pow(n_per_dim, dim), xp_fourier.cols());
  phi_mat = Matrix::Zero(lucid::pow(n_per_dim, dim), xp_fourier.cols());
  LUCID_DEBUG_FMT("w_mat size: {}x{}", w_mat.rows(), w_mat.cols());
  LUCID_DEBUG_FMT("phi_mat size: {}x{}", phi_mat.rows(), phi_mat.cols());

  for (Index i = 0; i < w_mat.cols(); ++i) {
    LUCID_INFO_FMT("Progress {}/{}", i + 1, w_mat.cols());
    // TODO(tend): this only works for 2 dimensions
    const Matrix w{w_vec_fourier.col(i).reshaped(samples_per_dim, samples_per_dim).transpose()};

    w_mat.col(i) = project(w, dim, n_per_dim, samples_per_dim);
    // std::cout << "w_vec\n" << w_vec.col(i).transpose() << std::endl;

    const Matrix phi{x_lattice_fourier.col(i).reshaped(samples_per_dim, samples_per_dim).transpose()};
    phi_mat.col(i) = project(phi, dim, n_per_dim, samples_per_dim);
    // std::cout << "phi_vec\n" << phi_vec << std::endl;

#if 0
    Matrix a = w_vec_fourier.col(i).reshaped(samples_per_dim, samples_per_dim).transpose();
    Matrix p =
        project(w, dim, n_per_dim, samples_per_dim).reshaped(samples_per_dim * 2, samples_per_dim * 2).transpose();
    plotting_fcn(x_lattice, samples_per_dim, samples_per_dim * 2, a, init_barr3.x_limits(), p);
#endif
  }
}

Matrix lattice_rect_multiset(const MultiSet& multi_set, const Dimension dim, const int num_samples) {
  Matrix rect_multiset_lattice{0, dim};
  for (const auto& initial_set : multi_set.sets()) {
    const RectSet& rect_set = dynamic_cast<const RectSet&>(*initial_set);
    const Matrix initial_lattice{lattice(rect_set, num_samples, true)};
    rect_multiset_lattice.conservativeResize(rect_multiset_lattice.rows() + initial_lattice.rows(),
                                             initial_lattice.cols());
    rect_multiset_lattice.bottomRows(initial_lattice.rows()) = initial_lattice;
  }
  return rect_multiset_lattice;
}

void plot_solution_2d(const benchmark::InitBarr3Scenario& scenario, const Vector& b, double eta, double gamma,
                      const TruncatedFourierFeatureMap& genBasis_x) {
  constexpr Dimension res = 100;
  Matrix X, Y;
  meshgrid(Vector::LinSpaced(res, scenario.x_limits()(0, 0), scenario.x_limits()(1, 0)),
           Vector::LinSpaced(res, scenario.x_limits()(0, 1), scenario.x_limits()(1, 1)), X, Y);
  Matrix XY{X.reshaped(Eigen::AutoSize, 1)};
  XY.conservativeResize(XY.rows(), XY.cols() + 1);
  XY.col(1) = Y.reshaped(Eigen::AutoSize, 1);
  Matrix f_XY = genBasis_x(XY);
  auto b_XY = f_XY * b;
  plt::plot_surface(X, Y, b_XY.reshaped(res, res));
}

void KBCLP(const benchmark::Scenario& scenario, const TruncatedFourierFeatureMap& tffm, const Kernel& kernel) {
  const benchmark::InitBarr3Scenario& init_barr3 = dynamic_cast<const benchmark::InitBarr3Scenario&>(scenario);
  const int maxNumFreqPerDim = init_barr3.num_freq_per_dim() - 1;  // Deducting the zero frequency level
  Matrix w_mat, phi_mat;
  cme_2_fourier(scenario, tffm, kernel, w_mat, phi_mat);

  // Sample initial regions
  Matrix x_initial_lattice{lattice_rect_multiset(dynamic_cast<const MultiSet&>(init_barr3.initial_set()),
                                                 init_barr3.dimension(), 2 * init_barr3.num_supp_per_dim() - 1)};
  LUCID_DEBUG_FMT("x_initial_lattice {}x{}", x_initial_lattice.rows(), x_initial_lattice.cols());
  // std::cout << "initial_lattice\n" << x_initial_lattice << std::endl;
  Matrix x_unsafe_lattice{lattice_rect_multiset(dynamic_cast<const MultiSet&>(init_barr3.unsafe_set()),
                                                init_barr3.dimension(), 2 * init_barr3.num_supp_per_dim() - 1)};
  LUCID_DEBUG_FMT("x_unsafe_lattice {}x{}", x_unsafe_lattice.rows(), x_unsafe_lattice.cols());
  // std::cout << "unsafe_lattice\n" << x_unsafe_lattice << std::endl;

  // Construct bases
  Matrix f_initial_lattice{tffm(x_initial_lattice)};
  LUCID_DEBUG_FMT("f_initial_lattice {}x{}", f_initial_lattice.rows(), f_initial_lattice.cols());
  // std::cout << "f_initial_lattice\n" << f_initial_lattice << std::endl;
  Matrix f_unsafe_lattice{tffm(x_unsafe_lattice)};
  LUCID_DEBUG_FMT("f_unsafe_lattice {}x{}", f_unsafe_lattice.rows(), f_unsafe_lattice.cols());
  // std::cout << "f_unsafe_lattice\n" << f_unsafe_lattice << std::endl;

  // Fix variables
  double min_num = 0;                                        // %1e-13; % Minimum variable value for numerical stability
  double max_num = std::numeric_limits<double>::infinity();  // %1e13; % Maximum variable value for numerical stability
  double min_eta = 0;
  double C = pow((1 - 2.0 * maxNumFreqPerDim / (2.0 * init_barr3.num_supp_per_dim())), -init_barr3.dimension() / 2.0);
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
  Index nVars = tffm.dimension() + 2 + 4;
  std::unique_ptr<GRBVar> vars_{model.addVars(nVars)};
  const std::span<GRBVar> vars{vars_.get(), static_cast<std::size_t>(nVars)};
  GRBVar& c = vars[vars.size() - 6];
  GRBVar& eta = vars[vars.size() - 5];
  GRBVar& minX0 = vars[vars.size() - 4];
  GRBVar& maxXU = vars[vars.size() - 3];
  GRBVar& maxXX = vars[vars.size() - 2];
  GRBVar& minDelta = vars[vars.size() - 1];

  // Variables related to the feature map [0, 71)
  for (GRBVar& var : vars.subspan(0, tffm.dimension())) {
    var.set(GRB_DoubleAttr_LB, -max_num);
    var.set(GRB_DoubleAttr_UB, max_num);
  }
  c.set(GRB_DoubleAttr_LB, 0);
  c.set(GRB_DoubleAttr_UB, max_num);
  eta.set(GRB_DoubleAttr_LB, min_eta);
  eta.set(GRB_DoubleAttr_UB, init_barr3.gamma() - min_num);
  for (GRBVar& var : std::array{minX0, maxXU, maxXX, minDelta}) {
    var.set(GRB_DoubleAttr_LB, 0);
    var.set(GRB_DoubleAttr_UB, max_num);
  }

  const double maxXX_coeff = -(C - 1) / (C + 1);
  const double fctr1 = 2 / (C + 1);
  const double fctr2 = (C - 1) / (C + 1);
  const double unsafe_rhs = fctr1 * init_barr3.gamma();
  const double kushner_rhs = -fctr1 * init_barr3.epsilon() * init_barr3.b_norm() * std::abs(init_barr3.sigma_f());

  // To obtain only positive safety probabilities, restrict
  // eta + c*T in [0, gamma]
  // 1) eta + c*T >= 0 by design
  // 2) eta + c*T <= gamma
  LUCID_DEBUG("Restricting safety probabilities to be positive");
  model.addConstr(eta + init_barr3.T() * c, GRB_LESS_EQUAL, init_barr3.gamma());

  LUCID_DEBUG(
      "Positive barrier\n"
      "for all x: [ B(x) >= hatxi ] AND [ B(x) <= maxXX ]\n"
      "hatxi = (C - 1) / (C + 1) * maxXX");
  for (Index row = 0; row < phi_mat.rows(); ++row) {
    GRBLinExpr expr{};
    expr.addTerms(phi_mat.row(row).eval().data(), vars_.get(), phi_mat.cols());
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
  for (Index row = 0; row < f_initial_lattice.rows(); ++row) {
    GRBLinExpr expr{};
    expr.addTerms(f_initial_lattice.row(row).eval().data(), vars_.get(), f_initial_lattice.cols());
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
  for (Index row = 0; row < f_unsafe_lattice.rows(); ++row) {
    GRBLinExpr expr{};
    expr.addTerms(f_unsafe_lattice.row(row).eval().data(), vars_.get(), f_unsafe_lattice.cols());
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
  auto mult = w_mat - init_barr3.kappa_b() * phi_mat;
  for (Index row = 0; row < w_mat.rows(); ++row) {
    GRBLinExpr expr{};
    expr.addTerms(mult.row(row).eval().data(), vars_.get(), mult.cols());
    expr += -fctr1 * c - fctr2 * minDelta;
    model.addConstr(expr, GRB_LESS_EQUAL, kushner_rhs);
  }
  for (Index row = 0; row < phi_mat.rows(); ++row) {
    GRBLinExpr expr{-minDelta};
    expr.addTerms(phi_mat.row(row).eval().data(), vars_.get(), phi_mat.cols());
    model.addConstr(expr, GRB_GREATER_EQUAL, 0);
  }

  // Objective function
  model.setObjective(GRBLinExpr{init_barr3.T() / init_barr3.gamma() * c + 1 / init_barr3.gamma() * eta});

  model.optimize();
  if (model.get(GRB_IntAttr_SolCount) == 0) {
    LUCID_WARN_FMT("No solution found, optimization status = {}", model.get(GRB_IntAttr_Status));
  } else {
    LUCID_INFO_FMT("Solution found, objective = {}", model.get(GRB_DoubleAttr_ObjVal));
    LUCID_INFO_FMT("Satisfaction probability is {:.6f}% percent", 1 - model.get(GRB_DoubleAttr_ObjVal));
  }

  auto solution{Vector::NullaryExpr(tffm.dimension(), [&vars](Index i) { return vars[i].get(GRB_DoubleAttr_X); })};
  double actual_norm = solution.norm();
  LUCID_INFO_FMT("Actual norm: {}", actual_norm);
  if (actual_norm > init_barr3.b_norm()) {
    LUCID_WARN_FMT("Actual norm exceeds bound: {} > {} (diff: {})", actual_norm, init_barr3.b_norm(),
                   actual_norm - init_barr3.b_norm());
  }
  double eta_result = eta.get(GRB_DoubleAttr_X);
  double c_result = c.get(GRB_DoubleAttr_X);
  LUCID_INFO_FMT("eta: {}", eta_result);
  LUCID_INFO_FMT("c: {}", c_result);

  plot_solution_2d(init_barr3, solution, eta_result, c_result, tffm);
}

}  // namespace

/**
 * Main function.
 * @param argc Number of arguments.
 * @param argv Arguments.
 * @return Execution status.
 */
int main(int, char**) {
  LUCID_LOG_INIT_VERBOSITY(4);
  plt::backend("WebAgg");
  // Seeded randomness
  std::srand(1);

#if 1
  benchmark::InitBarr3Scenario scenario;

  LUCID_DEBUG_FMT("x_samples: {}x{}", scenario.x_samples().rows(), scenario.x_samples().cols());
  LUCID_DEBUG_FMT("xp_samples: {}x{}", scenario.xp_samples().rows(), scenario.xp_samples().cols());
  GaussianKernel kernel{scenario.sigma_f(), scenario.sigma_l()};

  TruncatedFourierFeatureMap tfm{scenario.num_freq_per_dim(), scenario.dimension(), scenario.sigma_l(),
                                 scenario.sigma_f(), scenario.x_limits()};
  LUCID_DEBUG_FMT("omega_T: {}x{}", tfm.omega().rows(), tfm.omega().cols());
  LUCID_DEBUG_FMT("weights: {}x{}", tfm.weights().rows(), tfm.weights().cols());

  KBCLP(scenario, tfm, kernel);

  plt::show();
#endif
}

#pragma GCC diagnostic pop
