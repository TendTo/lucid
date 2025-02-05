/**
 * @file main.cpp
 * @author Room 6.030
 * @copyright 2025 lucid
 * @licence Apache-2.0 license
 * @file
 */
#include <fmt/core.h>

#include <iostream>
#include <vector>

#include "benchmark/InitBarr3Generator.h"
#include "lucid/math/GaussianKernel.h"
#include "lucid/math/GramMatrix.h"
#include "lucid/util/logging.h"
#include "lucid/util/matplotlibcpp.h"
#include "util/error.h"

namespace plt = matplotlibcpp;

using namespace lucid;  // NOLINT

namespace {

void plot_points(const Matrix& points, const std::string& color = "blue", const int size = 10) {
  if (points.rows() != 2) throw std::invalid_argument("points must have 2 rows");
  Vector x = points.row(0);
  Vector y = points.row(1);
  plt::scatter(x, y, size, {{"color", color}});
}

[[maybe_unused]] void wavelengths(const Dimension dimension, const int num_frequencies_per_dimension) {
  if (dimension < 1) throw std::invalid_argument("dimension must be at least 1");
  Dimension cols = 1;
  Dimension divided_cols = 1;
  for (Dimension i = 0; i < dimension; i++) {
    if (i != 0) divided_cols *= num_frequencies_per_dimension;
    cols *= num_frequencies_per_dimension;
  }
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

#if 1

  const benchmark::InitBarr3Generator generator;
  generator.plot();
  Matrix inputs, outputs;
  generator.sample_transition(1000, inputs, outputs);
  plot_points(inputs, "blue");
  plot_points(outputs, "magenta");

  GaussianKernel kernel{1, Vector::Constant(generator.dimension(), 0.3)};
  GramMatrix gram_matrix{kernel, inputs, outputs, 0.0001};

  // const Matrix samples = generator.sample_element(10);
  // plot_points(samples, "green");
  Matrix res = gram_matrix(inputs);
  std::cout << "RMS: " << rms(res - outputs) << std::endl;
  plot_points(res, "cyan", 2);

  plt::figure(2);
  generator.sample_transition(1000, inputs, outputs);
  generator.plot();
  plot_points(inputs, "blue");
  plot_points(outputs, "magenta");

  res = gram_matrix(inputs);
  std::cout << "RMS: " << rms(res - outputs) << std::endl;
  plot_points(res, "cyan", 2);

  plt::figure(3);
  plt::scatter(static_cast<Vector>(inputs.row(0)), static_cast<Vector>(inputs.row(1)),
               static_cast<Vector>(outputs.row(0)));

  plt::figure(4);
  plt::scatter(static_cast<Vector>(inputs.row(0)), static_cast<Vector>(inputs.row(1)), static_cast<Vector>(res.row(0)));

#endif

#if 0
  std::vector<Scalar> x;
  std::vector<Scalar> y;
  constexpr int size = 2;
  auto v = mvnrnd(Vector::Zero(size), Matrix::Identity(size, size) * 0.01);
  fmt::print("v: {}\n", v);
  for (int i = 0; i < 100; i++) {
    auto internal = mvnrnd(Vector::Zero(size), Matrix::Identity(size, size) * 0.01);
    x.push_back(internal(0));
    y.push_back(internal(1));
  }

  plt::scatter(x, y, 10);
#endif
#if 0

  // constexpr int n = 7;  // Similar to MATLAB default peaks(49)
  const Vector x = arange(-4, 4, 0.01, true);
  const Vector z = peaks(x, x);

  std::vector<Scalar> x_remaining, z_remaining;
  std::vector<Index> missing_indices, remaining_indices;
  x_remaining.reserve(x.size());
  z_remaining.reserve(x.size());
  missing_indices.reserve(x.size());
  remaining_indices.reserve(x.size());
  assert(x.size() == z.size());
  // for (int i = 0; i < x.size(); i++) {
  //   if (std::rand() < RAND_MAX * 0.01 * loss_percentage) {
  //     missing_indices.push_back(i);
  //     continue;
  //   }
  //   z_remaining.push_back(z(i));
  //   x_remaining.push_back(x(i));
  //   remaining_indices.push_back(i);
  // }
  for (int i = 0; i < x.size(); i += 50) {
    for (int j = 1; j < 100; j++) {
      if (i + j >= x.size()) break;
      missing_indices.push_back(i + j);
    }
    z_remaining.push_back(z(i));
    x_remaining.push_back(x(i));
    remaining_indices.push_back(i);
  }
  const Vector x_remaining_v = Eigen::Map<Vector>(x_remaining.data(), static_cast<Index>(x_remaining.size()));
  const Vector z_remaining_v = Eigen::Map<Vector>(z_remaining.data(), static_cast<Index>(z_remaining.size()));

  Vector sigma_l = Vector::Ones(1) * 0.3;

  GaussianKernel kernel{1.0, sigma_l};
  fmt::println("Kernel {}", kernel);

  GramMatrix gram_matrix{kernel, x_remaining_v, z_remaining_v};
  fmt::println("matrix: {}", gram_matrix);
  fmt::println("{}/{} entries missing. Only {} remaining", missing_indices.size(), x.size(), remaining_indices.size());

  Vector estimated_z{z.size()};
  estimated_z(remaining_indices) = z_remaining_v;
  for (const auto& idx : missing_indices) {
    auto res = gram_matrix(x.row(idx))(0);
    estimated_z(idx) = res;
  }

  plt::scatter(x_remaining_v, z_remaining_v, 10);
  plt::plot(x, z);
  plt::plot(x, estimated_z);
#endif
  plt::show();

#if 0
  Vector sigma_l(1), target(1);
  sigma_l << 1.0;
  target << 1.5;
  Matrix initial_states{3, 1};
  initial_states << 1.0, 2.0, 3.0;
  Matrix transition_states{3, 1};
  transition_states << 2.0, 3.0, 2.5;

  GaussianKernel kernel{1.0, sigma_l};
  fmt::println("Kernel {}", kernel);

  GramMatrix gram_matrix{kernel, initial_states, transition_states};
  fmt::println("matrix: {}", gram_matrix);
  auto res = gram_matrix(target);
  fmt::println("Solution: {}", res);

  std::vector<int> test_data;
  for (int i = 0; i < 20; i++) {
    test_data.push_back(i);
  }
  Vector test_data2{test_data.size()};
  for (std::size_t i = 0; i < test_data.size(); i++) {
    test_data2(i) = test_data[i];
  }
#endif
}
