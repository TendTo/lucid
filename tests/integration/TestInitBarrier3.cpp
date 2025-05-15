/**
 * @author Room 6.030
 * @copyright 2025 lucid
 * @licence BSD 3-Clause License
 */
#include <gtest/gtest.h>

#include "lucid/lucid.h"
#include "lucid/math/TensorView.h"
#include "lucid/util/math.h"

using namespace lucid;

constexpr double tolerance = 1e-3;
constexpr int num_supp_per_dim = 12;
constexpr Dimension dimension = 2;
constexpr int num_freq_per_dim = 6;
constexpr double sigma_f = 19.456;
constexpr double b_norm = 25;
constexpr double kappa_b = 1.0;
constexpr double gmma = 18.312;
constexpr int T = 10;
constexpr double lambda = 1e-5;
constexpr double epsilon = 1e-3;
[[maybe_unused]] constexpr int N = 1000;
[[maybe_unused]] constexpr bool autonomous = true;
const RectSet limit_set{Vector2{-3, -2}, Vector2{2.5, 1}};                     ///< Set.
const MultiSet initial_set{RectSet{Vector2{1, -0.5}, Vector2{2, 0.5}},         //
                           RectSet{Vector2{-1.8, -0.1}, Vector2{-1.2, 0.1}},   //
                           RectSet{Vector2{-1.4, -0.5}, Vector2{-1.2, 0.1}}};  ///< Initial set.
const MultiSet unsafe_set{RectSet{Vector2{0.4, 0.1}, Vector2{0.6, 0.5}},       //
                          RectSet{Vector2{0.4, 0.1}, Vector2{0.8, 0.3}}};      ///< Unsafe set.

class TestInitBarrier3 : public ::testing::Test {
 protected:
  TestInitBarrier3() {
    x_samples = read_matrix<double>("tests/integration/init_barrier_3/x_samples.matrix");
    xp_samples = read_matrix<double>("tests/integration/init_barrier_3/xp_samples.matrix");
    sigma_l = Vector{dimension};
    sigma_l << 30, 23.568;

    expected_f0_lattice = read_matrix<double>("tests/integration/init_barrier_3/f0_lattice.matrix");
    expected_f_lattice = read_matrix<double>("tests/integration/init_barrier_3/f_lattice.matrix");
    expected_fp_samples = read_matrix<double>("tests/integration/init_barrier_3/fp_samples.matrix");
    expected_fu_lattice = read_matrix<double>("tests/integration/init_barrier_3/fu_lattice.matrix");
    expected_if_lattice = read_matrix<double>("tests/integration/init_barrier_3/if_lattice.matrix");

    expected_phi_mat = read_matrix<double>("tests/integration/init_barrier_3/phi_mat.matrix");
    expected_w_mat = read_matrix<double>("tests/integration/init_barrier_3/w_mat.matrix");

    expected_x_lattice = read_matrix<double>("tests/integration/init_barrier_3/x_lattice.matrix");
    expected_x0_lattice = read_matrix<double>("tests/integration/init_barrier_3/x0_lattice.matrix");
    expected_xu_lattice = read_matrix<double>("tests/integration/init_barrier_3/xu_lattice.matrix");

    EXPECT_GT(x_samples.size(), 0l);
    EXPECT_GT(xp_samples.size(), 0l);
    EXPECT_GT(sigma_l.size(), 0l);
    EXPECT_GT(expected_f0_lattice.size(), 0l);
    EXPECT_GT(expected_f_lattice.size(), 0l);
    EXPECT_GT(expected_fp_samples.size(), 0l);
    EXPECT_GT(expected_fu_lattice.size(), 0l);
    EXPECT_GT(expected_if_lattice.size(), 0l);
    EXPECT_GT(expected_phi_mat.size(), 0l);
    EXPECT_GT(expected_w_mat.size(), 0l);
    EXPECT_GT(expected_x_lattice.size(), 0l);
    EXPECT_GT(expected_x0_lattice.size(), 0l);
    EXPECT_GT(expected_xu_lattice.size(), 0l);
  }

  Matrix x_samples;
  Matrix xp_samples;
  Vector sigma_l;

  Matrix expected_f0_lattice;
  Matrix expected_f_lattice;
  Matrix expected_fp_samples;
  Matrix expected_fu_lattice;
  Matrix expected_if_lattice;

  Matrix expected_phi_mat;
  Matrix expected_w_mat;

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

TEST_F(TestInitBarrier3, InitBarrier3) {
  const GaussianKernel kernel{sigma_l, sigma_f};
  const ConstantTruncatedFourierFeatureMap tffm{num_freq_per_dim, sigma_l, sigma_f, limit_set};

  // With n frequencies, the highest frequency is n-1 (they go from 0 to n-1).
  // So, by Shannon's theorem, we need 2n - 1 samples to avoid aliasing. 2n will do.
  constexpr int samples_per_dim = 2 * num_freq_per_dim;

  const Matrix x_lattice{limit_set.lattice(samples_per_dim)};
  ASSERT_TRUE(x_lattice.isApprox(expected_x_lattice, tolerance));

  // Construct bases
  Matrix f_lattice{tffm(x_lattice)};
  ASSERT_TRUE(f_lattice.isApprox(expected_f_lattice, tolerance));
  Matrix fp_samples{tffm(xp_samples)};
  ASSERT_TRUE(fp_samples.isApprox(expected_fp_samples, tolerance));

  // Build the regressor to interpolate the basis for any point
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
  ASSERT_TRUE(w_mat.isApprox(expected_w_mat, tolerance));
  ASSERT_TRUE(phi_mat.isApprox(expected_phi_mat, tolerance));

  // Sample initial regions
  const Matrix x0_lattice{initial_set.lattice(n_per_dim - 1, true)};
  ASSERT_TRUE(x0_lattice.isApprox(expected_x0_lattice, tolerance));
  const Matrix xu_lattice{unsafe_set.lattice(n_per_dim - 1, true)};
  ASSERT_TRUE(xu_lattice.isApprox(expected_xu_lattice, tolerance));

  // Construct bases
  const Matrix f0_lattice{tffm(x0_lattice)};
  ASSERT_TRUE(f0_lattice.isApprox(expected_f0_lattice, tolerance));
  const Matrix fu_lattice{tffm(xu_lattice)};
  ASSERT_TRUE(fu_lattice.isApprox(expected_fu_lattice, tolerance));

  [[maybe_unused]] GurobiLinearOptimiser optimiser{T, gmma, epsilon, b_norm, kappa_b, sigma_f};
#ifdef LUCID_GUROBI_BUILD
  const bool res = optimiser.solve(f0_lattice, fu_lattice, phi_mat, w_mat, tffm.dimension(), num_freq_per_dim - 1,
                                   n_per_dim, dimension,
                                   [](const bool success, const double obj_val, const Vector &sol, const double eta,
                                      const double c, const double norm) {
                                     EXPECT_TRUE(success);
                                     EXPECT_NEAR(obj_val, 0.83752674401056304, tolerance);
                                     EXPECT_NEAR(eta, 15.336789736321432, tolerance);
                                     EXPECT_NEAR(c, 0.0, tolerance);
                                     EXPECT_NEAR(norm, 10.393929781427465, tolerance);
                                     EXPECT_EQ(sol.size(), 71);
                                   });
  EXPECT_TRUE(res);
#endif
}
