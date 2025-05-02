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
#include <span>
#include <string>
#include <utility>
#include <vector>

#include "lucid/lucid.h"
#include "lucid/util/logging.h"
#include "lucid/util/math.h"

#ifdef LUCID_MATPLOTLIB_BUILD
#include "lucid/util/matplotlib.h"
#endif
#include "util/error.h"

using namespace lucid;  // NOLINT

// #pragma GCC diagnostic ignored "-Wunused-variable"
// #pragma GCC diagnostic ignored "-Wunused-parameter"
// #pragma GCC diagnostic ignored "-Wunused-but-set-variable"
// #pragma GCC diagnostic ignored "-Wunused-function"

namespace {

#if 0
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

Matrix wavelengths(const Dimension dimension, const int num_frequencies_per_dimension) {
  if (dimension < 1) throw std::invalid_argument("dimension must be at least 1");
  // Compute the total number of columns of the output matrix
  const Vector frequencies{arange(0, num_frequencies_per_dimension)};

  Matrix frequency_combination = frequencies.transpose();
  for (Dimension i = 1; i < dimension; i++) {
    frequency_combination = combvec(frequency_combination, frequencies.transpose());
  }
  return (frequency_combination.rightCols(frequency_combination.cols() - 1) * 2 * std::numbers::pi).transpose();
}

Basis generate_basis(ConstMatrixRef omega_T, const Dimension dimension, Scalar sigma_f, ConstVectorRef sigma_l,
                     const int num_frequencies_per_dimension) {
  if (sigma_l.size() < 1) throw std::invalid_argument("sigma_l must have at least one element");

  const Vector omega_dim_wise_lb =
      (2 * std::numbers::pi * arange(0, num_frequencies_per_dimension)).array() - std::numbers::pi;
  const Vector omega_dim_wise_ub = omega_dim_wise_lb.array() + 2 * std::numbers::pi;

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

}  // namespace

/**
 * Main function.
 * @param argc Number of arguments.
 * @param argv Arguments.
 * @return Execution status.
 */
int main(int, char**) {
  LUCID_LOG_INIT_VERBOSITY(4);
#ifdef LUCID_MATPLOTLIB_BUILD
  plt::backend("WebAgg");
  // Seeded randomness
  std::srand(1);
#endif

#if 0


  // KBCLP(scenario, tfm, kernel);

  plt::show();
#else
  const RectSet set{Vector2{-1, -1}, Vector2{1, 1}, 0};
  std::cout << set.lattice(3) << std::endl;
  std::cout << set.lattice(3, true) << std::endl;
#ifdef LUCID_MATPLOTLIB_BUILD
  plt::plot(Vector::LinSpaced(100, -1, 1), Vector::Random(100), {.fmt = "r--"});
#endif

  Tensor<double> t{{1, 2, 3, 4, 5, 6}, {2, 3}};
  std::cout << t << std::endl;

  for (const auto& val : t) {
    std::cout << val << std::endl;
  }

  std::cout << t.permute(1, 0) << std::endl;

  for (const auto& val : t) {
    std::cout << val << std::endl;
  }

  std::vector<double> v(t.begin(), t.end());
  std::cout << Matrix::Map(v.data(), static_cast<Index>(t.dimensions()[0]), static_cast<Index>(t.dimensions()[1]))
            << std::endl;

#endif
}

// #pragma GCC diagnostic pop
