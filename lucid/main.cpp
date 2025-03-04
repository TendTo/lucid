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

Matrix lattice(const Matrix& x_limits, const Index samples_per_dim) {
  Matrix x_lattice{1, samples_per_dim};
#if 1
  const Matrix delta_per_dim = (x_limits.row(1) - x_limits.row(0)).array() / static_cast<double>(samples_per_dim);
  x_lattice.row(0) = arange(x_limits(0, 0), x_limits(1, 0), delta_per_dim(0));
  for (Dimension i = 1; i < x_limits.cols(); ++i) {
    x_lattice = combvec(x_lattice, arange(x_limits(0, i), x_limits(1, i), delta_per_dim(i)).transpose());
  }
#else
  // TODO(tend): must we exclude the endpoint?
  grid.row(0) = Vector::LinSpaced(samples_per_dim, x_limits(0, 0), init_barr3(1, 0));
  for (Dimension i = 1; i < x_limits.cols(); ++i) {
    grid = combvec(grid, Vector::LinSpaced(samples_per_dim, x_limits(0, i), x_limits(1, i)).transpose());
  }
#endif
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
  const Eigen::MatrixXcd f_fft{fftn(f)};
  // std::cout << "f" << std::endl << f << std::endl;

  const int n_pad = floor((n_per_dim / 2 - samples_per_dim / 2));
  fmt::println("n_pad: {}", n_pad);

  Eigen::MatrixXcd padded_ft{pad(fftshift(f_fft), n_pad, std::complex<double>{})};
  fmt::println("padded_ft: {}x{}", padded_ft.rows(), padded_ft.cols());
  // std::cout << "padded_ft\n" << padded_ft << std::endl;
  // std::cout << "ifftn\n" << ifftn(ifftshift(padded_ft)) << std::endl;

  Matrix f_interp = ifftn(ifftshift(padded_ft)).array() * lucid::pow(n_per_dim / samples_per_dim, dim);
  // std::cout << "f_interp\n" << f_interp << std::endl;
  return f_interp.reshaped(Eigen::AutoSize, 1);
}

void cme_2_fourier(const benchmark::Scenario& scenario, const TruncatedFourierFeatureMap& tfm, const Kernel& kernel) {
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

  // fmt::println("x_lattice_fourier: {}x{}\n{}", x_lattice_fourier.rows(), x_lattice_fourier.cols(),
  // x_lattice_fourier);
  // fmt::println("xp_fourier: {}x{}\n{}", xp_fourier.rows(), xp_fourier.cols(), xp_fourier);

  const KernelRidgeRegression regression{gaussian_kernel, init_barr3.x_samples(), xp_fourier, init_barr3.lambda()};
  Matrix w_vec_fourier = regression(x_lattice);
  fmt::println("res: {}x{}\n{}", w_vec_fourier.rows(), w_vec_fourier.cols(), w_vec_fourier);

  int factor = std::ceil(init_barr3.num_supp_per_dim() / static_cast<double>(samples_per_dim)) + 1;
  fmt::println("factor: {}", factor);
  int n_per_dim = factor * samples_per_dim;
  fmt::println("n_per_dim: {}", n_per_dim);

  Matrix w_vec{Matrix::Zero(lucid::pow(n_per_dim, dim), xp_fourier.cols())};
  Matrix phi_vec{Matrix::Zero(lucid::pow(n_per_dim, dim), xp_fourier.cols())};
  fmt::println("w_vec size: {}x{}", w_vec.rows(), w_vec.cols());
  fmt::println("phi_vec size: {}x{}", phi_vec.rows(), phi_vec.cols());
  std::cout << std::endl;

  for (Index i = 0; i < w_vec.cols() && i < 3; ++i) {
    fmt::println("Progress {}/{}", i + 1, w_vec.cols());
    // TODO(tend): this only works for 2 dimensions
    const Matrix w{w_vec_fourier.col(i).reshaped(samples_per_dim, samples_per_dim).transpose()};

    std::cout << std::endl;
    w_vec.col(i) = project(w, dim, n_per_dim, samples_per_dim);
    std::cout << "w_vec\n" << w_vec.col(i).transpose() << std::endl;

    const Matrix phi{x_lattice_fourier.col(i).reshaped(samples_per_dim, samples_per_dim).transpose()};
    phi_vec.col(i) = project(phi, dim, n_per_dim, samples_per_dim);
    // std::cout << "phi_vec\n" << phi_vec << std::endl;

    Matrix a = w_vec_fourier.col(i).reshaped(samples_per_dim, samples_per_dim).transpose();
    Matrix p =
        project(w, dim, n_per_dim, samples_per_dim).reshaped(samples_per_dim * 2, samples_per_dim * 2).transpose();
    plotting_fcn(x_lattice, samples_per_dim, samples_per_dim * 2, a, init_barr3.x_limits(), p);
  }
}

void KBCLP(const benchmark::Scenario& scenario, const TruncatedFourierFeatureMap& tfm, const Kernel& kernel) {
  const benchmark::InitBarr3Scenario& init_barr3 = dynamic_cast<const benchmark::InitBarr3Scenario&>(scenario);
  const int maxNumFreqPerDim = init_barr3.num_freq_per_dim() - 1;  // Deducting the zero frequency level
  cme_2_fourier(scenario, tfm, kernel);
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
 * Main function.
 * @param argc Number of arguments.
 * @param argv Arguments.
 * @return Execution status.
 */
int main(int, char**) {
  LUCID_LOG_INIT_VERBOSITY(1);
  plt::backend("WebAgg");
  // Seeded randomness
  std::srand(1);

#if 1
  benchmark::InitBarr3Scenario scenario;

  fmt::println("fake x sampling: {}x{}", scenario.x_samples().rows(), scenario.x_samples().cols());
  fmt::println("fake xp sampling: {}x{}", scenario.xp_samples().rows(), scenario.xp_samples().cols());
  GaussianKernel kernel{scenario.sigma_f(), scenario.sigma_l()};

  // Matrix omega_T = wavelengths(scenario.dimension(), scenario.num_freq_per_dim());
  // // fmt::println("fake omega_T: {}x{}", omega_T.rows(), omega_T.cols());
  // // fmt::println("fake omega_T: {}", omega_T);
  //
  // const Basis basis = generate_basis(omega_T, scenario.dimension(), scenario.sigma_f(), scenario.sigma_l(),
  //                                    scenario.num_freq_per_dim());

  // KBCLP(scenario, basis, kernel);
  TruncatedFourierFeatureMap tfm{scenario.num_freq_per_dim(), scenario.dimension(), scenario.sigma_l(),
                                 scenario.sigma_f(), scenario.x_limits()};
  fmt::println("fake omega_T: {}x{}", tfm.omega().rows(), tfm.omega().cols());
  fmt::println("fake omega_T: {}", tfm.omega());
  fmt::println("fake weights: {}x{}", tfm.weights().rows(), tfm.weights().cols());
  fmt::println("fake weights: {}", tfm.weights());
  fmt::println("scenario.x_samples: {}x{}", scenario.x_samples().rows(), scenario.x_samples().cols());

  // Matrix exp = basis(scenario.x_samples().row(0), scenario.x_limits());
  // fmt::println("fake exp: {}x{}", exp.rows(), exp.cols());
  // fmt::println("fake exp: {}", exp);
  std::cout << std::endl;

  KBCLP(scenario, tfm, kernel);

  plt::show();
  // Matrix omega_T = wavelengths(dim, frequencies).transpose();
  // auto basis = generate_basis(omega_T, dim, sigma_f, sigma_l, frequencies);
  // auto to_basis = basis(x, x_limits);
  // std::cout << "OUT" << to_basis << std::endl;

#endif
}

#pragma GCC diagnostic pop
