/**
 * @author Room 6.030
 * @copyright 2025 lucid
 * @licence BSD 3-Clause License
 */
#include <gmock/gmock.h>
#include <gtest/gtest.h>

#include "lucid/lucid.h"

using namespace lucid;

constexpr int seed = 42;
constexpr int N = 1000;
constexpr int time_horizon = 5;
constexpr int num_frequencies = 4;
constexpr double tolerance = 1e-3;
constexpr double sigma_f = 15.0;
constexpr double sigma_l = 1.75555556;
constexpr double b_norm = 1.0;
constexpr double b_kappa = 1.0;
constexpr double gmma = 1.0;
constexpr double lambda = 1e-3;
constexpr double epsilon = 0;
constexpr double c_coefficient = 1.0;
constexpr double oversample_factor = 32.0;
constexpr double noise_scale = 0.01;
const RectSet X_bounds{{{-1, 1}}};
const RectSet X_init{{{-0.5, 0.5}}};
const MultiSet X_unsafe{RectSet{{-1, -0.9}}, RectSet{{0.9, 1}}};

template <class T>
class TestLinear : public testing::Test {
 public:
  TestLinear() : optimiser_{time_horizon, gmma, epsilon, b_norm, b_kappa, sigma_f, c_coefficient} { random::seed(42); }

  // Linear function: f(x) = 0.5 * x
  static Matrix f_det(const Matrix& x) { return 0.5 * x; }

  // Function with noise: f(x) = 0.5 * x + noise
  static Matrix f(const Matrix& x) {
    std::normal_distribution d{0.0, noise_scale};
    // Add noise to the linear function
    const Matrix y{f_det(x)};
    return y + Matrix::NullaryExpr(y.rows(), y.cols(), [&d](Index, Index) { return d(random::gen); });
  }

  T optimiser_;
};

using OptimizerTypes = ::testing::Types<
#ifdef LUCID_GUROBI_BUILD
    AlglibOptimiser
#endif
#if defined(LUCID_ALGLIB_BUILD) && defined(LUCID_GUROBI_BUILD)
    ,
#endif
#ifdef LUCID_ALGLIB_BUILD
    GurobiOptimiser
#endif
#if (defined(LUCID_ALGLIB_BUILD) || defined(LUCID_GUROBI_BUILD)) && defined(LUCID_HIGHS_BUILD)
    ,
#endif
#ifdef LUCID_HIGHS_BUILD
    HighsOptimiser
#endif
    >;
TYPED_TEST_SUITE(TestLinear, OptimizerTypes);

TYPED_TEST(TestLinear, TestLinear) {
  const Matrix x_samples{X_bounds.sample(N)};
  const Matrix xp_samples{this->f(x_samples)};

  KernelRidgeRegressor estimator{std::make_unique<GaussianKernel>(sigma_l, sigma_f), lambda};
  LinearTruncatedFourierFeatureMap feature_map{num_frequencies, sigma_l, sigma_f, X_bounds};

  const Matrix f_xp_samples{feature_map(xp_samples)};

  const int n_per_dim = static_cast<int>(std::ceil((2 * num_frequencies + 1) * oversample_factor));
  ASSERT_GT(n_per_dim, 2 * num_frequencies);
  ASSERT_EQ(n_per_dim, 288);

  estimator.fit(x_samples, f_xp_samples);
  ASSERT_DOUBLE_EQ(scorer::rmse_score(estimator, x_samples, f_xp_samples), -0.55940545363305361);
  ASSERT_DOUBLE_EQ(estimator.score(x_samples, f_xp_samples), 0.9901976271473962);

  const Matrix x_evaluation = X_bounds.sample(x_samples.rows() / 2);
  const Matrix f_xp_evaluation = feature_map(this->f_det(x_evaluation));
  ASSERT_DOUBLE_EQ(scorer::rmse_score(estimator, x_evaluation, f_xp_evaluation), -0.5265806387684655);
  ASSERT_DOUBLE_EQ(estimator.score(x_evaluation, f_xp_evaluation), 0.9913180698330606);

  const Matrix x_lattice = X_bounds.lattice(n_per_dim, true);
  const Matrix u_f_x_lattice = feature_map(x_lattice);
  Matrix u_f_xp_lattice_via_regressor = estimator(x_lattice);
  // We are fixing the zero frequency to the constant value we computed in the feature map
  // If we don't, the regressor has a hard time learning it on the extreme left and right points, because it tends to 0
  u_f_xp_lattice_via_regressor.col(0).array() = feature_map.weights()[0] * sigma_f;

  const Matrix x0_lattice = X_init.lattice(n_per_dim, true);
  const Matrix f_x0_lattice = feature_map(x0_lattice);

  const Matrix xu_lattice = X_unsafe.lattice(n_per_dim, true);
  const Matrix f_xu_lattice = feature_map(xu_lattice);

  auto check_cb = [](const bool success, const float obj_val, [[maybe_unused]] const Vector& sol,
                     [[maybe_unused]] const float eta, [[maybe_unused]] const float c,
                     [[maybe_unused]] const float norm) {
    ASSERT_TRUE(success);
    ASSERT_NEAR(obj_val, 0.0959243, tolerance);
    ASSERT_NEAR(eta, 0.05833359, tolerance);
    ASSERT_NEAR(c, 0.007518151, tolerance);
    ASSERT_NEAR(norm, 0.07567549, tolerance);
    constexpr std::array expected{0.0353493, 0.0468235, -6.16032e-05, 0.0404232, -0.000184679, 0.0255075, -0.000246124};
    for (std::size_t i = 0; i < expected.size(); ++i) ASSERT_NEAR(sol[i], expected[i], tolerance);
  };

  ASSERT_TRUE(this->optimiser_.solve(f_x0_lattice, f_xu_lattice, u_f_x_lattice, u_f_xp_lattice_via_regressor,
                                     feature_map.dimension(), num_frequencies - 1, n_per_dim, X_bounds.dimension(),
                                     check_cb));
}
