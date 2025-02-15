/**
 * @file main.cpp
 * @author Room 6.030
 * @copyright 2025 lucid
 * @licence Apache-2.0 license
 * @file
 */
#include <fmt/core.h>

#include <chrono>
#include <iostream>
#include <vector>

#include "lucid/lucid.h"
#include "lucid/util/logging.h"
#include "lucid/util/matplotlibcpp.h"

namespace plt = matplotlibcpp;

using namespace lucid;  // NOLINT

namespace {

[[maybe_unused]] void test_surface() {
  Matrix X, Y;
  meshgrid(arange(-5, 5, 0.5), arange(-5, 5, 0.5), X, Y);
  Matrix Z = peaks(X, Y);
  plt::plot_wireframe(X, Y, Z);
}

[[maybe_unused]] void plot_points(const Matrix& points, const std::string& color = "blue", const double size = 10) {
  if (points.cols() != 2) throw std::invalid_argument("points must have 2 rows");
  const Vector x = points.col(0);
  const Vector y = points.col(1);
  plt::scatter(x, y, {.s = size, .c = color});
}

[[maybe_unused]] Matrix wavelengths(const Dimension dimension, const int num_frequencies_per_dimension) {
  if (dimension < 1) throw std::invalid_argument("dimension must be at least 1");
  // Compute the total number of columns of the output matrix
  Dimension cols = num_frequencies_per_dimension;
  for (Dimension i = 1; i < dimension; i++) cols *= num_frequencies_per_dimension;
  const Vector frequencies{arange(0, num_frequencies_per_dimension)};

  Matrix frequency_combination = frequencies.transpose();
  for (Dimension i = 1; i < dimension; i++) {
    frequency_combination = combvec(frequency_combination, frequencies.transpose());
  }
  return frequency_combination * 2 * M_PI;
}

[[maybe_unused]] void generate_basis([[maybe_unused]] ConstMatrixRef omega_T, [[maybe_unused]] Dimension dimension,
                                     [[maybe_unused]] Scalar sigma_f, const Vector& sigma_l,
                                     const int num_frequencies_per_dimension) {
  if (sigma_l.size() < 1) throw std::invalid_argument("sigma_l must have at least one element");

  // const Dimension M = omega_T.rows();
  const Vector omega_dim_wise_lb = (2 * M_PI * arange(0, num_frequencies_per_dimension)).array() - M_PI;
  const Vector omega_dim_wise_ub = omega_dim_wise_lb.array() + 2 * M_PI;

  // fmt::println("M: {}", M);
  // fmt::println("omega_dim_wise_lb: {}", omega_dim_wise_lb);
  // fmt::println("omega_dim_wise_ub: {}", omega_dim_wise_ub);
}

[[maybe_unused]] void generate_basis(ConstMatrixRef omega_T, Dimension dimension, Scalar sigma_f, Scalar sigma_l,
                                     const int num_frequencies_per_dimension) {
  return generate_basis(omega_T, dimension, sigma_f, Vector::Constant(dimension, sigma_l),
                        num_frequencies_per_dimension);
}

[[maybe_unused]] std::vector<std::vector<Scalar>> to_vector(const Matrix& matrix) {
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
  plt::backend("WebAgg");
  // Seeded randomness
  std::srand(1);

#if 0
  Matrix omega_T = wavelengths(2, 10).transpose();
  generate_basis(omega_T, 2, 1, 1, 10);
#endif

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
  plt::show();
}
