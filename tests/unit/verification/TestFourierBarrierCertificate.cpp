/**
 * @author Room 6.030
 * @copyright 2025 lucid
 * @licence BSD 3-Clause License
 */
#include <gmock/gmock.h>
#include <gtest/gtest.h>

#include <numbers>

#include "lucid/model/ConstantTruncatedFourierFeatureMap.h"
#include "lucid/model/RectSet.h"
#include "lucid/verification/FourierBarrierCertificate.h"

using lucid::ConstantTruncatedFourierFeatureMap;
using lucid::ConstMatrixRef;
using lucid::FourierBarrierCertificate;
using lucid::Matrix;
using lucid::RectSet;
using lucid::Vector;

class TestFourierBarrierCertificate : public ::testing::Test {
 protected:
  TestFourierBarrierCertificate()
      : T_{10},
        gamma_{1.0},
        eta_{0.1},
        c_{0.05},
        barrier_{T_, gamma_, eta_, c_},
        X_bounds_{std::vector<std::pair<double, double>>(2, {-1.0, 1.0})} {}

  const int T_;
  const double gamma_;
  const double eta_;
  const double c_;
  FourierBarrierCertificate barrier_;
  const RectSet X_bounds_;
};

TEST_F(TestFourierBarrierCertificate, ConstructorInitialization) {
  EXPECT_EQ(barrier_.T(), T_);
  EXPECT_EQ(barrier_.gamma(), gamma_);
  EXPECT_EQ(barrier_.eta(), eta_);
  EXPECT_EQ(barrier_.c(), c_);
  EXPECT_FALSE(barrier_.is_synthesized());
  EXPECT_EQ(barrier_.norm(), 0.0);
  EXPECT_EQ(barrier_.safety(), 0.0);
}

TEST_F(TestFourierBarrierCertificate, IsNotSynthesizedByDefault) { EXPECT_FALSE(barrier_.is_synthesized()); }

TEST_F(TestFourierBarrierCertificate, ComputeObjective) {
  const RectSet& X_tilde{{-1.0, 1.0}, {-1.0, 1.0}};
  const RectSet& X{{-0.5, 0.5}, {-0.5, 0.5}};
  constexpr double increase = 0.1;
  constexpr int lattice_resolution = 25;
  constexpr int f_max = 3;

  double result = 0.0;
  EXPECT_NO_THROW(result = barrier_.compute_A(lattice_resolution, f_max, X_tilde, X, {.increase = increase}));
  EXPECT_GT(result, 0);
}
