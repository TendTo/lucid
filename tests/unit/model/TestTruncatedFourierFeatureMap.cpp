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
using lucid::RectSet;
using lucid::Scalar;
using lucid::Vector;
using lucid::Vector2;

constexpr double sigma_f = 1.0;
constexpr double sigma_l = 1.0;
constexpr int num_frequencies = 8;

template <typename T>
class TestTruncatedFourierFeatureMap : public testing::Test {
 protected:
  RectSet x_limits_{Vector2{-1, -1}, Vector2{1, 1}};
  T feature_map_{num_frequencies, sigma_l, sigma_f, x_limits_};
  Dimension feature_map_dimension_{static_cast<Dimension>(lucid::pow(num_frequencies, x_limits_.dimension()))};
};

using TestTypes = ::testing::Types<ConstantTruncatedFourierFeatureMap, LinearTruncatedFourierFeatureMap,
                                   LogTruncatedFourierFeatureMap>;
TYPED_TEST_SUITE(TestTruncatedFourierFeatureMap, TestTypes);

TYPED_TEST(TestTruncatedFourierFeatureMap, TruncatedFourierFeatureMapConstructor) {
  EXPECT_EQ(this->feature_map_.num_frequencies(), 8);
  EXPECT_EQ(this->feature_map_.dimension(), this->feature_map_dimension_ * 2 - 1);
  EXPECT_EQ(this->feature_map_.weights().size(), this->feature_map_dimension_ * 2 - 1);
  EXPECT_EQ(this->feature_map_.omega().rows(), this->feature_map_dimension_);
  EXPECT_EQ(this->feature_map_.omega().cols(), this->x_limits_.dimension());
  EXPECT_GT(this->feature_map_.captured_probability(), 0.7);
  EXPECT_LE(this->feature_map_.captured_probability(), 1.0);
}

TYPED_TEST(TestTruncatedFourierFeatureMap, TruncatedFourierFeatureMapApply) {
  const Vector x{Vector::Random(this->x_limits_.dimension())};
  const Matrix features{this->feature_map_(x.transpose())};
  EXPECT_EQ(features.rows(), 1);
  EXPECT_EQ(features.cols(), this->feature_map_.dimension());
  EXPECT_FALSE(std::isnan(features(0, 0)));
}

TYPED_TEST(TestTruncatedFourierFeatureMap, TruncatedFourierFeatureMapApplyMatrix) {
  constexpr Dimension samples = 15;

  const Matrix x{Matrix::Random(samples, this->x_limits_.dimension())};
  const Matrix features{this->feature_map_(x)};
  EXPECT_EQ(features.rows(), samples);
  EXPECT_EQ(features.cols(), this->feature_map_.dimension());
  EXPECT_FALSE(std::isnan(features(0, 0)));
}

TYPED_TEST(TestTruncatedFourierFeatureMap, TruncatedFourierFeatureMapApplyVector) {
  const Vector x{Vector::Random(this->x_limits_.dimension())};
  const Vector features{this->feature_map_.map_vector(x)};
  EXPECT_EQ(features.size(), this->feature_map_.dimension());
  EXPECT_FALSE(std::isnan(features(0)));
}

TYPED_TEST(TestTruncatedFourierFeatureMap, TruncatedFourierFeatureMapApply1D) {
  RectSet x_limits{std::vector<std::pair<Scalar, Scalar>>{{-1, 1}}};
  TypeParam feature_map{num_frequencies, sigma_l, sigma_f, x_limits};
  const Vector x{Vector::Random(x_limits.dimension())};
  const Matrix features{feature_map(x.transpose())};
  EXPECT_EQ(features.rows(), 1);
  EXPECT_EQ(features.cols(), feature_map.dimension());
  EXPECT_FALSE(std::isnan(features(0, 0)));
}
