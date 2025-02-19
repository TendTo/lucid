/**
 * @file main.cpp
 * @author Room 6.030
 * @copyright 2025 lucid
 * @licence Apache-2.0 license
 * @file
 */
#include <fmt/core.h>

#include <chrono>
#include <cmath>
#include <iostream>
#include <vector>

#include "lucid/lucid.h"
#include "lucid/util/logging.h"
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

void cme_2_fourier(const benchmark::Scenario& scenario, const Basis& basis, GramMatrix& gram_matrix) {
  const benchmark::InitBarr3Scenario& init_barr3 = dynamic_cast<const benchmark::InitBarr3Scenario&>(scenario);
  // TODO(tend): double check this. Number of num_freq... and actual number of samples are not the same in matlab
  const Dimension dim = init_barr3.dimension();
  if (dim < 1) throw std::invalid_argument("dimension must be at least 1");
  const int samples_per_dim = 2 * (init_barr3.num_freq_per_dim_x() - 1) + 2;
  Matrix grid{1, samples_per_dim};

#if 1
  const Matrix delta_per_dim =
      (init_barr3.x_limits().row(1) - init_barr3.x_limits().row(0)).array() / static_cast<double>(samples_per_dim);
  grid.row(0) = arange(init_barr3.x_limits()(0, 0), init_barr3.x_limits()(1, 0), delta_per_dim(0));
  for (Dimension i = 1; i < dim; ++i) {
    grid =
        combvec(grid, arange(init_barr3.x_limits()(0, i), init_barr3.x_limits()(1, i), delta_per_dim(i)).transpose());
  }
#else
  // TODO(tend): must we exclude the endpoint?
  grid.row(0) = Vector::LinSpaced(samples_per_dim, init_barr3.x_limits()(0, 0), init_barr3.x_limits()(1, 0));
  for (Dimension i = 1; i < dim; ++i) {
    grid = combvec(
        grid, Vector::LinSpaced(samples_per_dim, init_barr3.x_limits()(0, i), init_barr3.x_limits()(1, i)).transpose());
  }
#endif
  grid.transposeInPlace();
  fmt::println("GRID: \n{}", grid);

  Matrix x_basis_fourier{grid.rows(), init_barr3.num_freq_per_dim_x() * init_barr3.num_freq_per_dim_x() * 2 - 1};
  Matrix xp_basis_fourier{grid.rows(), init_barr3.num_freq_per_dim_x() * init_barr3.num_freq_per_dim_x() * 2 - 1};
  for (Index row = 0; row < grid.rows(); row++) {
    x_basis_fourier.row(row) = basis(grid.row(row), init_barr3.x_limits()).transpose();
    xp_basis_fourier.row(row) = basis(init_barr3.xp_samples().row(row), init_barr3.x_limits()).transpose();
  }

  gram_matrix.compute_coefficients(xp_basis_fourier);
  auto res = gram_matrix(x_basis_fourier);
  fmt::println("res:\n{}", res);
}

void KBCLP(const benchmark::Scenario& scenario, const Basis& basis, const Kernel& kernel, GramMatrix& gram_matrix) {
  const benchmark::InitBarr3Scenario& init_barr3 = dynamic_cast<const benchmark::InitBarr3Scenario&>(scenario);
  const int maxNumFreqPerDim = init_barr3.num_freq_per_dim_x() - 1;  // Deducting the zero frequency level
  cme_2_fourier(scenario, basis, gram_matrix);
}

std::vector<std::vector<Scalar>> to_vector(const Matrix& matrix) {
  std::vector<std::vector<Scalar>> res(matrix.rows(), std::vector<Scalar>(matrix.cols()));
  for (Index i = 0; i < matrix.rows(); i++) {
    for (Index j = 0; j < matrix.cols(); j++) {
      res[i][j] = matrix(i, j);
    }
  }
  return res;
}

}  // namespace

/**
 * @brief Main function.
 * @param argc Number of arguments.
 * @param argv Arguments.
 * @return Execution status.
 */
int main(int, char**) {
  LUCID_LOG_INIT_VERBOSITY(3);
  plt::backend("WebAgg");
  // Seeded randomness
  std::srand(1);

  benchmark::InitBarr3Scenario scenario;

  fmt::println("fake x sampling: {}x{}", scenario.x_samples().rows(), scenario.x_samples().cols());
  fmt::println("fake xp sampling: {}x{}", scenario.xp_samples().rows(), scenario.xp_samples().cols());
  GaussianKernel kernel{scenario.sigma_f(), scenario.sigma_l()};
  GramMatrix gram_matrix{kernel, scenario.x_samples(), scenario.lambda() * scenario.N()};
  fmt::print("fake gram matrix: {}x{}\n", gram_matrix.gram_matrix().rows(), gram_matrix.gram_matrix().cols());

  Matrix omega_T = wavelengths(scenario.dimension(), scenario.num_freq_per_dim_x());
  fmt::println("fake omega_T: {}x{}", omega_T.rows(), omega_T.cols());
  fmt::println("fake omega_T: {}", omega_T);

  const Basis basis = generate_basis(omega_T, scenario.dimension(), scenario.sigma_f(), scenario.sigma_l(),
                                     scenario.num_freq_per_dim_x());

  KBCLP(scenario, basis, kernel, gram_matrix);

  // Matrix omega_T = wavelengths(dim, frequencies).transpose();
  // auto basis = generate_basis(omega_T, dim, sigma_f, sigma_l, frequencies);
  // auto to_basis = basis(x, x_limits);
  // std::cout << "OUT" << to_basis << std::endl;

#if 0
  // Generic test
  const benchmark::InitBarr3Scenario scenario;
  scenario.plot();
  Matrix inputs, outputs;
  scenario.sample_transition(1000, inputs, outputs);
  plot_points(inputs, "blue");
  plot_points(outputs, "magenta");

  GaussianKernel kernel{1, Vector::Constant(scenario.dimension(), 0.3)};
  GramMatrix gram_matrix{kernel, inputs, outputs, 0.0001};

  // const Matrix samples = scenario.sample_element(10);
  // plot_points(samples, "green");
  Matrix res = gram_matrix(inputs);
  std::cout << "RMS: " << rms(res - outputs) << std::endl;
  plot_points(res, "cyan", 2);

  plt::figure(2);
  scenario.sample_transition(1000, inputs, outputs);
  scenario.plot();
  plot_points(inputs, "blue");
  plot_points(outputs, "magenta");

  res = gram_matrix(inputs);
  std::cout << "RMS: " << rms(res - outputs) << std::endl;
  plot_points(res, "cyan", 2);

  plt::figure(3);
  plt::scatter(inputs.col(0), inputs.col(1), outputs.col(0));

  plt::figure(4);
  plt::scatter(static_cast<Vector>(inputs.col(0)), static_cast<Vector>(inputs.col(1)), static_cast<Vector>(res.col(0)));

#endif
  // plt::show();
}

#pragma GCC diagnostic pop
