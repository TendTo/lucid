/**
 * @author Room 6.030
 * @copyright 2025 lucid
 * @licence BSD 3-Clause License
 */
#include <gtest/gtest.h>

#include <atomic>
#include <functional>

#include "lucid/lib/eigen.h"
#include "lucid/model/MultiSet.h"
#include "lucid/model/RectSet.h"
#include "lucid/util/exception.h"
#include "lucid/util/random.h"
#include "lucid/verification/MontecarloSimulation.h"

using lucid::ConstVectorRef;
using lucid::Dimension;
using lucid::Index;
using lucid::Matrix;
using lucid::MontecarloSimulation;
using lucid::MultiSet;
using lucid::RectSet;
using lucid::Vector;

constexpr double tolerance = 1e-8;

class TestMontecarloSimulation : public ::testing::Test {
 protected:
  TestMontecarloSimulation() { lucid::random::seed(42); }

  const std::size_t time_horizon_ = 15;
  const Dimension num_samples_ = 1000;
  const double confidence_level_ = 0.9;
  const double noise_scale_ = 0.1;
  const RectSet X_bounds_{{-1}, {1}};
  const RectSet X_init_{{-0.5}, {0.5}};
  const MultiSet X_unsafe_{RectSet{{-1}, {-0.9}}, RectSet{{0.9}, {1}}};
  const MontecarloSimulation sim_;
  const std::function<Vector(ConstVectorRef)> dynamics_ = [](ConstVectorRef x) { return x; };
};

TEST_F(TestMontecarloSimulation, DeterministicAllSamplesSafe) {
  // f(x) = x/2, so all samples in X_init will remain in [-0.5, 0.5] for any time horizon
  const auto dynamics = [](ConstVectorRef x) -> Vector { return x.array() / 2; };

  const auto [lb, ub] =
      sim_.safety_probability(X_bounds_, X_init_, X_unsafe_, dynamics, time_horizon_, confidence_level_, num_samples_);

  EXPECT_GE(ub, lb);
  EXPECT_EQ(ub, 1);
  EXPECT_GT(lb, 0);
  EXPECT_LE(ub - lb, std::sqrt(1 / (4 * static_cast<double>(num_samples_) * (1 - confidence_level_))) * 2);
}

TEST_F(TestMontecarloSimulation, StochasticAllSamplesSafe) {
  std::normal_distribution d{0.0, noise_scale_};
  // f(x) = x/2, so all samples in X_init will remain in [-0.5, 0.5] for any time horizon
  const auto dynamics = [&d](ConstVectorRef x) -> Vector {
    return (x.array() / 2) +
           Vector::NullaryExpr(x.rows(), [&d](Index, Index) { return d(lucid::random::gen); }).array();
  };

  const auto [lb, ub] =
      sim_.safety_probability(X_bounds_, X_init_, X_unsafe_, dynamics, time_horizon_, confidence_level_, num_samples_);

  EXPECT_GE(ub, lb);
  EXPECT_EQ(ub, 1);
  EXPECT_GT(lb, 0);
  EXPECT_LE(ub - lb, std::sqrt(1 / (4 * static_cast<double>(num_samples_) * (1 - confidence_level_))) * 2);
}

TEST_F(TestMontecarloSimulation, StochasticSamplesSafe) {
  std::normal_distribution d{0.0, 0.7};
  // f(x) = x/2, so all samples in X_init will remain in [-0.5, 0.5] for any time horizon
  const auto dynamics = [&d](ConstVectorRef x) -> Vector {
    return (x.array() / 2) +
           Vector::NullaryExpr(x.rows(), [&d](Index, Index) { return d(lucid::random::gen); }).array();
  };

  const auto [lb, ub] =
      sim_.safety_probability(X_bounds_, X_init_, X_unsafe_, dynamics, time_horizon_, confidence_level_, num_samples_);

  EXPECT_GE(ub, lb);
  EXPECT_LT(ub, 1);
  EXPECT_GT(lb, 0);
  EXPECT_LE(ub - lb, std::sqrt(1 / (4 * static_cast<double>(num_samples_) * (1 - confidence_level_))) * 2 + tolerance);
}

TEST_F(TestMontecarloSimulation, SamplesAllUnsafe) {
  // f(x) = x/2, so all samples in X_init will remain in [-0.5, 0.5] for any time horizon
  const auto dynamics = [](ConstVectorRef x) -> Vector { return x.array() / 2; };

  const auto [lb, ub] =
      sim_.safety_probability(X_bounds_, X_unsafe_, X_init_, dynamics, time_horizon_, confidence_level_, num_samples_);

  EXPECT_GE(ub, lb);
  EXPECT_LT(ub, 1);
  EXPECT_EQ(lb, 0);
  EXPECT_LE(ub - lb, std::sqrt(1 / (4 * static_cast<double>(num_samples_) * (1 - confidence_level_))) * 2 + tolerance);
}

TEST_F(TestMontecarloSimulation, InvalidConfidence) {
  // Confidence outside [0,1]
  EXPECT_THROW(static_cast<void>(sim_.safety_probability(X_bounds_, X_init_, X_unsafe_, dynamics_, time_horizon_, -0.1,
                                                         num_samples_)),
               lucid::exception::LucidInvalidArgumentException);
  EXPECT_THROW(static_cast<void>(
                   sim_.safety_probability(X_bounds_, X_init_, X_unsafe_, dynamics_, time_horizon_, 1.1, num_samples_)),
               lucid::exception::LucidInvalidArgumentException);
}

TEST_F(TestMontecarloSimulation, InvalidNumSamples) {
  // Number of samples <= 0
  EXPECT_THROW(static_cast<void>(sim_.safety_probability(X_bounds_, X_init_, X_unsafe_, dynamics_, time_horizon_,
                                                         confidence_level_, 0)),
               lucid::exception::LucidInvalidArgumentException);
}

TEST_F(TestMontecarloSimulation, MissmatchDimensions) {
  const RectSet X_bounds{{-1, -1}, {1, 1}};
  const RectSet X_init{{-0.5, -0.5}, {0.5, 0.5}};
  const RectSet X_unsafe{{-1, -1}, {-0.9, -0.9}};
  // Number of samples <= 0
  EXPECT_THROW(static_cast<void>(sim_.safety_probability(X_bounds, X_init_, X_unsafe_, dynamics_, time_horizon_,
                                                         confidence_level_, num_samples_)),
               lucid::exception::LucidInvalidArgumentException);
  EXPECT_THROW(static_cast<void>(sim_.safety_probability(X_bounds_, X_init, X_unsafe_, dynamics_, time_horizon_,
                                                         confidence_level_, num_samples_)),
               lucid::exception::LucidInvalidArgumentException);
  EXPECT_THROW(static_cast<void>(sim_.safety_probability(X_bounds_, X_init_, X_unsafe, dynamics_, time_horizon_,
                                                         confidence_level_, num_samples_)),
               lucid::exception::LucidInvalidArgumentException);
}
