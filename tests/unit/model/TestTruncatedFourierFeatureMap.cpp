/**
 * @author Room 6.030
 * @copyright 2025 lucid
 * @licence BSD 3-Clause License
 */
#include <gtest/gtest.h>

#include "lucid/model/ConstantTruncatedFourierFeatureMap.h"
#include "lucid/model/LinearTruncatedFourierFeatureMap.h"
#include "lucid/model/LogTruncatedFourierFeatureMap.h"
#include "lucid/model/RectSet.h"
#include "lucid/util/math.h"

using lucid::ConstantTruncatedFourierFeatureMap;
using lucid::Dimension;
using lucid::Index;
using lucid::LinearTruncatedFourierFeatureMap;
using lucid::LogTruncatedFourierFeatureMap;
using lucid::Matrix;
using lucid::normal_cdf;
using lucid::RectSet;
using lucid::Scalar;
using lucid::TruncatedFourierFeatureMap;
using lucid::Vector;
using lucid::Vector2;

constexpr double sigma_f = 1.0;
constexpr double sigma_l = 1.0;
constexpr int num_frequencies = 8;
const RectSet x_limits{Vector2{-1, -1}, Vector2{1, 1}};

template <typename T>
class TestTruncatedFourierFeatureMap : public testing::Test {
 protected:
  T feature_map_{num_frequencies, sigma_l, sigma_f, x_limits};
  Dimension feature_map_dimension_{static_cast<Dimension>(lucid::pow(num_frequencies, x_limits.dimension()))};
};

using TestTypes = ::testing::Types<ConstantTruncatedFourierFeatureMap, LinearTruncatedFourierFeatureMap,
                                   LogTruncatedFourierFeatureMap>;
TYPED_TEST_SUITE(TestTruncatedFourierFeatureMap, TestTypes);

TYPED_TEST(TestTruncatedFourierFeatureMap, TruncatedFourierFeatureMapConstructor) {
  EXPECT_EQ(this->feature_map_.num_frequencies(), 8);
  EXPECT_EQ(this->feature_map_.dimension(), this->feature_map_dimension_ * 2 - 1);
  EXPECT_EQ(this->feature_map_.weights().size(), this->feature_map_dimension_ * 2 - 1);
  EXPECT_EQ(this->feature_map_.omega().rows(), this->feature_map_dimension_);
  EXPECT_EQ(this->feature_map_.omega().cols(), x_limits.dimension());
  EXPECT_GT(this->feature_map_.captured_probability(), 0.7);
  EXPECT_LE(this->feature_map_.captured_probability(), 1.0);
}

TYPED_TEST(TestTruncatedFourierFeatureMap, TruncatedFourierFeatureMapApply) {
  const Vector x{Vector::Random(x_limits.dimension())};
  const Matrix features{this->feature_map_(x)};
  EXPECT_EQ(features.rows(), 1);
  EXPECT_EQ(features.cols(), this->feature_map_.dimension());
  EXPECT_FALSE(std::isnan(features(0, 0)));
}

TYPED_TEST(TestTruncatedFourierFeatureMap, TruncatedFourierFeatureMapApplyMatrix) {
  constexpr Dimension samples = 15;

  const Matrix x{Matrix::Random(samples, x_limits.dimension())};
  const Matrix features{this->feature_map_(x)};
  EXPECT_EQ(features.rows(), samples);
  EXPECT_EQ(features.cols(), this->feature_map_.dimension());
  EXPECT_FALSE(std::isnan(features(0, 0)));
}

TYPED_TEST(TestTruncatedFourierFeatureMap, TruncatedFourierFeatureMapApplyVector) {
  const Vector x{Vector::Random(x_limits.dimension())};
  const Vector features{this->feature_map_.map_vector(x)};
  EXPECT_EQ(features.size(), this->feature_map_.dimension());
  EXPECT_FALSE(std::isnan(features(0)));
}

TYPED_TEST(TestTruncatedFourierFeatureMap, TruncatedFourierFeatureMapApply1D) {
  RectSet x_limits_1d{std::vector<std::pair<Scalar, Scalar>>{{-1, 1}}};
  TypeParam feature_map{num_frequencies, sigma_l, sigma_f, x_limits_1d};
  const Vector x{Vector::Random(x_limits_1d.dimension())};
  const Matrix features{feature_map(x.transpose())};
  EXPECT_EQ(features.rows(), 1);
  EXPECT_EQ(features.cols(), feature_map.dimension());
  EXPECT_FALSE(std::isnan(features(0, 0)));
}

TEST(TestTruncatedFourierFeatureMap, CorrectSpread) {
  // We are using sigma_f = 1.0 and sigma_l = 3 (so a normal distribution with mean 0.0 and stddev 3.0)
  // We divide the normal distribution into 4 intervals on each side of the mean
  // ┌────────────────────────────────────────────────────┬────────────┬────────────┬────────────┬────────────┬┐
  // │                                                ▗▄▄▄▄▄▄▄▖-0.27   │    -0.16   │    -0.05   │    -0.01   ││
  // │                                        ▄▄▄▞▀▀▀▀▘   │   ▝▀▀▀▀▚▄▄▄│            │            │            ││
  // │                                  ▄▄▄▀▀▀            │            ▀▀▀▄▄▄       │            │            ││
  // │                            ▗▄▄▞▀▀                  │            │     ▀▀▚▄▄▖ │            │            ││
  // │                    ▄▄▄▄▞▀▀▀▘                       │            │          ▝▀▀▀▚▄▄▄▄      │            ││
  // │▄▄▄▄▄▄▄▄▄▄▄▞▀▀▀▀▀▀▀▀                                │            │            │      ▀▀▀▀▀▀▀▀▚▄▄▄▄▄▄▄▄▄▄▄│
  // └┬─────────────────────────┬─────────────────────────┴────────────┴────────────┴────────────┴────────────┴┘
  //-9.0                      -4.5                       0.0                       4.5                      9.0
  constexpr int exp_num_frequencies = 4;
  constexpr double exp_sigma_l = 3.0;
  const RectSet exp_x_limits{std::vector<std::pair<Scalar, Scalar>>{{-1, 1}}};
  const LinearTruncatedFourierFeatureMap feature_map{exp_num_frequencies, exp_sigma_l, 1.0, exp_x_limits};

  // Let's divide the interval [0, 3 * exp_sigma_l] into 7 intervals: 1 for the 0th frequency and 2 for each of the
  // remaining frequencies.
  constexpr double offset = 3 * exp_sigma_l / (2 * exp_num_frequencies - 1);
  // The intervals are [0, offset], [offset, 3 * offset], ..., [(2 * exp_num_frequencies - 2) * offset, 3 * exp_sigma_l]
  Vector intervals{exp_num_frequencies + 1};
  intervals(0) = 0.0;
  intervals(1) = offset;
  for (Index i = 2; i < intervals.size(); ++i) intervals(i) = intervals(i - 1) + offset * 2;

  Vector expected_values{normal_cdf(intervals.tail(exp_num_frequencies), 0, exp_sigma_l) -
                         normal_cdf(intervals.head(exp_num_frequencies), 0, exp_sigma_l)};
  // We need to consider the left side of the normal distribution as well, so we multiply by 2.
  expected_values *= 2;

  std::cout << "Intervals: " << intervals << std::endl;
  std::cout << "Expected values: " << expected_values.transpose() << std::endl;

  const TruncatedFourierFeatureMap expected_feature_map{exp_num_frequencies, expected_values, 1.0, exp_x_limits};

  EXPECT_TRUE(feature_map.omega().isApprox(expected_feature_map.omega()));
  EXPECT_TRUE(feature_map.weights().isApprox(expected_feature_map.weights()));
}
