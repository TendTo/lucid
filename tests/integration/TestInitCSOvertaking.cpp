/**
 * @author Room 6.030
 * @copyright 2025 lucid
 * @licence BSD 3-Clause License
 */
#include <gtest/gtest.h>

#include <numbers>

#include "lucid/lucid.h"
#include "lucid/util/error.h"
#include "lucid/util/math.h"

using namespace lucid;

constexpr double tolerance = 1e-3;
constexpr Dimension dimension = 3;
constexpr int num_supp_per_dim = 31;
constexpr int num_freq_per_dim = 4;
constexpr double sigma_f = 7;
constexpr double b_norm = 2.1;
constexpr double kappa_b = 1.0;
constexpr double gmma = 0.5;
constexpr int T = 50;
constexpr double lambda = 1e-5;
constexpr double epsilon = .1 / 30;
[[maybe_unused]] constexpr int N = 1000;
[[maybe_unused]] constexpr bool autonomous = true;
const RectSet limit_set{Vector3{1, -7, -std::numbers::pi}, Vector3{90, 19, std::numbers::pi}};  ///< Set.
const MultiSet initial_set{RectSet{Vector3{1, -0.5, -0.005}, Vector3{2, 0.5, 0.005}}};          ///< Initial set.
const MultiSet unsafe_set{
    RectSet{Vector3{1, -7, -std::numbers::pi}, Vector3{90, -6, std::numbers::pi}},
    RectSet{Vector3{1, 18, -std::numbers::pi}, Vector3{90, 19, std::numbers::pi}},
    RectSet{Vector3{40, -6, -std::numbers::pi}, Vector3{45, 6, std::numbers::pi}}};  ///< Unsafe set.

class TestInitCSOvertaking : public ::testing::Test {
 protected:
  TestInitCSOvertaking() {
    x_samples = read_matrix<double>("tests/integration/init_cs_overtaking/x_samples.matrix");
    xp_samples = read_matrix<double>("tests/integration/init_cs_overtaking/xp_samples.matrix");
    sigma_l = Vector{dimension};
    sigma_l << 10., 7., 5.;

    expected_f_lattice = read_matrix<double>("tests/integration/init_cs_overtaking/f_lattice.matrix");
    expected_fp_samples = read_matrix<double>("tests/integration/init_cs_overtaking/fp_samples.matrix");
    expected_if_lattice = read_matrix<double>("tests/integration/init_cs_overtaking/if_lattice.matrix");

    expected_x_lattice = read_matrix<double>("tests/integration/init_cs_overtaking/x_lattice.matrix");
    expected_x0_lattice = read_matrix<double>("tests/integration/init_cs_overtaking/x0_lattice.matrix");
    expected_xu_lattice = read_matrix<double>("tests/integration/init_cs_overtaking/xu_lattice.matrix");

    EXPECT_GT(x_samples.size(), 0l);
    EXPECT_GT(xp_samples.size(), 0l);
    EXPECT_GT(sigma_l.size(), 0l);
    EXPECT_GT(expected_f_lattice.size(), 0l);
    EXPECT_GT(expected_fp_samples.size(), 0l);
    EXPECT_GT(expected_if_lattice.size(), 0l);
    EXPECT_GT(expected_x_lattice.size(), 0l);
    EXPECT_GT(expected_x0_lattice.size(), 0l);
    EXPECT_GT(expected_xu_lattice.size(), 0l);
  }

  Matrix x_samples;
  Matrix xp_samples;
  Vector sigma_l;

  Matrix expected_f_lattice;
  Matrix expected_fp_samples;
  Matrix expected_if_lattice;

  Matrix expected_x_lattice;
  Matrix expected_x0_lattice;
  Matrix expected_xu_lattice;
};

inline Vector project(ConstMatrixRef f, const Index n_per_dim, const Index samples_per_dim) {
  if constexpr (dimension <= 1) throw std::runtime_error("Dimension must be greater than 1");

  const int n_pad = static_cast<int>(floor(static_cast<double>(n_per_dim / 2 - samples_per_dim / 2)));
  // Get a view of the input data
  const TensorView<double> in_view{std::span<const double>{f.data(), static_cast<std::size_t>(f.size())},
                                   std::vector<std::size_t>(dimension, samples_per_dim)};
  // Permute the last two axes and create a complex tensor
  Tensor<std::complex<double>> fft_in{in_view.dimensions()};
  std::vector<std::size_t> axes{fft_in.axes()};
  std::swap(axes[axes.size() - 2], axes[axes.size() - 1]);
  in_view.permute(fft_in.m_view(), axes);
  // Perform FFT upsampling on the data and return the result
  return static_cast<Eigen::Map<const Vector>>(
      fft_in.fft_upsample(std::vector<std::size_t>(dimension, samples_per_dim + 2 * n_pad)));
}

TEST_F(TestInitCSOvertaking, InitCSOvertaking) {
  const GaussianKernel kernel{sigma_l, sigma_f};
  const TruncatedFourierFeatureMap tffm{num_freq_per_dim, sigma_l, sigma_f, limit_set};

  // With n frequencies, the highest frequency is n-1 (they go from 0 to n-1).
  // So, by Shannon's theorem, we need 2n - 1 samples to avoid aliasing. 2n will do.
  constexpr int samples_per_dim = 2 * num_freq_per_dim;

  const Matrix x_lattice{limit_set.lattice(samples_per_dim)};
  ASSERT_TRUE(x_lattice.isApprox(expected_x_lattice, tolerance));

  Matrix f_lattice{tffm(x_lattice)};
  ASSERT_TRUE(f_lattice.isApprox(expected_f_lattice, tolerance));
  Matrix fp_samples{tffm(xp_samples)};
  ASSERT_TRUE(fp_samples.isApprox(expected_fp_samples, tolerance));

  const KernelRidgeRegressor regressor{kernel, x_samples, fp_samples, lambda};
  const Matrix if_lattice = regressor(x_lattice);
  ASSERT_TRUE(if_lattice.isApprox(expected_if_lattice, tolerance));

  const int factor = static_cast<int>(std::ceil(num_supp_per_dim / static_cast<double>(samples_per_dim)) + 1);
  const int n_per_dim = factor * samples_per_dim;

  Matrix w_mat = Matrix::Zero(lucid::pow(n_per_dim, dimension), fp_samples.cols());
  Matrix phi_mat = Matrix::Zero(lucid::pow(n_per_dim, dimension), fp_samples.cols());
  for (Index i = 0; i < w_mat.cols(); ++i) {
    w_mat.col(i) = project(if_lattice.col(i), n_per_dim, samples_per_dim);
    phi_mat.col(i) = project(f_lattice.col(i), n_per_dim, samples_per_dim);
  }

  // Sample initial regions
  const Matrix x0_lattice{initial_set.lattice(n_per_dim - 1, true)};
  ASSERT_TRUE(x0_lattice.isApprox(expected_x0_lattice, tolerance));
  const Matrix xu_lattice{unsafe_set.lattice(n_per_dim - 1, true)};
  ASSERT_TRUE(xu_lattice.isApprox(expected_xu_lattice, tolerance));

  // Construct bases
  const Matrix f0_lattice{tffm(x0_lattice)};
  const Matrix fu_lattice{tffm(xu_lattice)};

  [[maybe_unused]] GurobiOptimiser optimiser{T, gmma, epsilon, b_norm, kappa_b, sigma_f};
#ifdef LUCID_GUROBI_BUILD
  const bool res = optimiser.solve(f0_lattice, fu_lattice, phi_mat, w_mat, tffm.dimension(), num_freq_per_dim - 1,
                                   n_per_dim, dimension,
                                   [](const bool success, const double obj_val, const Vector& sol, const double eta,
                                      const double c, const double norm) {
                                     EXPECT_TRUE(success);
                                     EXPECT_NEAR(obj_val, 0.77774607136635343, tolerance);
                                     EXPECT_NEAR(eta, 0.38887303568317672, tolerance);
                                     EXPECT_NEAR(c, 0.0, tolerance);
                                     EXPECT_NEAR(norm, 0.58449853272166907, tolerance);
                                     EXPECT_EQ(sol.size(), 127);
                                   });
  EXPECT_TRUE(res);
#endif
}
