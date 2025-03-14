/**
 * @author Room 6.030
 * @copyright 2025 lucid
 * @licence BSD 3-Clause License
 */
#include <gtest/gtest.h>

#include "lucid/lucid.h"
#include "lucid/util/error.h"
#include "lucid/util/math.h"

using namespace lucid;

using Vector3 = Eigen::Matrix<double, 3, 1>;

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
constexpr int N = 1000;
constexpr double epsilon = .1 / 30;
constexpr bool autonomous = true;
const RectSet limit_set{Vector3{1, -7, -M_PI}, Vector3{90, 19, M_PI}};                  ///< Set.
const MultiSet initial_set{RectSet{Vector3{1, -0.5, -0.005}, Vector3{2, 0.5, 0.005}}};  ///< Initial set.
const MultiSet unsafe_set{RectSet{Vector3{1, -7, -M_PI}, Vector3{90, -6, M_PI}},
                          RectSet{Vector3{1, 18, -M_PI}, Vector3{90, 19, M_PI}},
                          RectSet{Vector3{40, -6, -M_PI}, Vector3{45, 6, M_PI}}};  ///< Unsafe set.

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

template <class Derived, int NumIndices>
Eigen::MatrixXcd fftn(const Eigen::MatrixBase<Derived>& x) {
  Eigen::Tensor<Scalar, NumIndices> t{
      Eigen::TensorMap<Eigen::Tensor<Scalar, NumIndices>>{x.data(), std::array{x.rows(), x.cols()}}};
  Eigen::Tensor<std::complex<double>, 2> res = t.template fft<Eigen::BothParts, Eigen::FFT_FORWARD>(std::array{0, 1});
  return MatrixCast(res, x.rows(), x.cols());
}



template <class Derived>
Vector project(const Eigen::MatrixBase<Derived>& f, const Index n_per_dim, const Index samples_per_dim) {
  // TODO(tend): this only works for 2 dimensions
  const int n_pad = floor((n_per_dim / 2 - samples_per_dim / 2));
  const double coeff = lucid::pow(n_per_dim / samples_per_dim, dimension);
  if (dimension == 1 || dimension == 2) {
    const Eigen::MatrixXcd f_fft{fft2(f.derived().reshaped(samples_per_dim, samples_per_dim).transpose())};
    // We do, in order:
    // 1. Shift the zero frequency to the center
    // 2. Pad the frequencies to increase the resolution
    // 3. Unshift the zero frequency to the corner
    // 4. Inverse FFT to get the interpolated function
    // 5. Scale the function by the ratio of the number of samples to the number of frequencies
    // 6. Reshape the matrix to a vector
    return (ifft2(ifftshift(pad(fftshift(f_fft), n_pad, std::complex<double>{}))).array() *
            lucid::pow(n_per_dim / samples_per_dim, dimension))
        .reshaped(Eigen::AutoSize, 1);
  }
  if (dimension == 3) {
    const Eigen::Tensor<std::complex<double>, 3> t{Eigen::TensorMap<Eigen::Tensor<const std::complex<double>, 3>>{
        f.template cast<std::complex<double>>().eval().data(),
        std::array{samples_per_dim, samples_per_dim, samples_per_dim}}};
    Eigen::Tensor<std::complex<double>, 3> temp =
        t.template fft<Eigen::BothParts, Eigen::FFT_FORWARD>(std::array{1, 0, 2})
            .pad(std::array{std::pair<Index, Index>{0, 2 * n_pad}, std::pair<Index, Index>{0, 2 * n_pad},
                            std::pair<Index, Index>{0, 2 * n_pad}});

    // Dim 0
    const Index total_dim = samples_per_dim + 2 * n_pad;
    temp.slice(std::array<Index, 3>{samples_per_dim / 2 + 2 * n_pad, 0, 0},
               std::array<Index, 3>{samples_per_dim / 2, total_dim, total_dim}) =
        temp.slice(std::array<Index, 3>{samples_per_dim / 2, 0, 0},
                   std::array<Index, 3>{samples_per_dim / 2, total_dim, total_dim})
            .eval();
    temp.slice(std::array<Index, 3>{samples_per_dim / 2, 0, 0},
               std::array<Index, 3>{samples_per_dim / 2, total_dim, total_dim})
        .setZero();
    // Dim 1
    temp.slice(std::array<Index, 3>{0, samples_per_dim / 2 + n_pad, 0},
               std::array<Index, 3>{total_dim, samples_per_dim / 2, total_dim}) =
        temp.slice(std::array<Index, 3>{0, samples_per_dim / 2, 0},
                   std::array<Index, 3>{total_dim, samples_per_dim / 2, total_dim})
            .eval();
    temp.slice(std::array<Index, 3>{0, samples_per_dim / 2, 0},
               std::array<Index, 3>{total_dim, samples_per_dim / 2, total_dim})
        .setZero();
    // Dim 2
    temp.slice(std::array<Index, 3>{0, 0, samples_per_dim / 2 + n_pad},
               std::array<Index, 3>{total_dim, total_dim, samples_per_dim / 2}) =
        temp.slice(std::array<Index, 3>{0, 0, samples_per_dim / 2},
                   std::array<Index, 3>{total_dim, total_dim, samples_per_dim / 2})
            .eval();
    temp.slice(std::array<Index, 3>{0, 0, samples_per_dim / 2},
               std::array<Index, 3>{total_dim, total_dim, samples_per_dim / 2})
        .setZero();
    const Eigen::Tensor<double, 3> temp2 =
        temp.template fft<Eigen::BothParts, Eigen::FFT_REVERSE>(std::array{0, 1, 2}).real() * coeff;
    return Vector::Map(temp2.data(), temp2.size());
  }
  LUCID_NOT_SUPPORTED("Only 2D and 3D are supported");
}

TEST_F(TestInitCSOvertaking, InitCSOvertaking) {
  const GaussianKernel kernel{sigma_f, sigma_l};
  const TruncatedFourierFeatureMap tffm{num_freq_per_dim, dimension, sigma_l, sigma_f, limit_set};

  // With n frequencies, the highest frequency is n-1 (they go from 0 to n-1).
  // So, by Shannon's theorem, we need 2n - 1 samples to avoid aliasing. 2n will do.
  constexpr int samples_per_dim = 2 * num_freq_per_dim;

  const Matrix x_lattice{limit_set.lattice(samples_per_dim)};
  ASSERT_TRUE(x_lattice.isApprox(expected_x_lattice, tolerance));

  Matrix f_lattice{tffm(x_lattice)};
  ASSERT_TRUE(f_lattice.isApprox(expected_f_lattice, tolerance));
  Matrix fp_samples{tffm(xp_samples)};
  ASSERT_TRUE(fp_samples.isApprox(expected_fp_samples, tolerance));

  const KernelRidgeRegression regression{kernel, x_samples, fp_samples, lambda};
  const Matrix if_lattice = regression(x_lattice);
  ASSERT_TRUE(if_lattice.isApprox(expected_if_lattice, tolerance));

  const int factor = std::ceil(num_supp_per_dim / static_cast<double>(samples_per_dim)) + 1;
  const int n_per_dim = factor * samples_per_dim;

  Matrix w_mat = Matrix::Zero(lucid::pow(n_per_dim, dimension), fp_samples.cols());
  Matrix phi_mat = Matrix::Zero(lucid::pow(n_per_dim, dimension), fp_samples.cols());
  for (Index i = 0; i < w_mat.cols(); ++i) {
    LUCID_INFO_FMT("Progress {}/{}", i + 1, w_mat.cols());
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

  GurobiLinearOptimiser optimiser{T, gmma, epsilon, b_norm, kappa_b, sigma_f};
  optimiser.solve(f0_lattice, fu_lattice, phi_mat, w_mat, tffm.dimension(), num_freq_per_dim - 1, n_per_dim, dimension,
                  [](const bool success, const double obj_val, const double eta, const double c, const double norm) {
                    EXPECT_TRUE(success);
                    EXPECT_NEAR(obj_val, 0.76609867952476407, tolerance);
                    EXPECT_NEAR(eta, 0.38304933976238204, tolerance);
                    EXPECT_NEAR(c, 0.0, tolerance);
                    EXPECT_NEAR(norm, 0.52365992867299227, tolerance);
                  });
}
