/**
 * @author Room 6.030
 * @copyright 2025 lucid
 * @licence BSD 3-Clause License
 */
#include <gmock/gmock.h>
#include <gtest/gtest.h>

#include "lucid/math/Tensor.h"
#include "lucid/util/exception.h"

using lucid::Index;
using lucid::Matrix;
using lucid::Tensor;
using lucid::Vector;
using lucid::Vector2;
using lucid::exception::LucidInvalidArgumentException;
using lucid::exception::LucidNotSupportedException;

#define EXPECT_COMPLEX_DOUBLE_EQ(a, b)        \
  do {                                        \
    EXPECT_DOUBLE_EQ((a).real(), (b).real()); \
    EXPECT_DOUBLE_EQ((a).imag(), (b).imag()); \
  } while (0)

#define EXPECT_VECTOR_NEAR_P(a, b, p)              \
  do {                                             \
    ASSERT_EQ((a).size(), (b).size());             \
    for (std::size_t i = 0; i < (a).size(); ++i) { \
      EXPECT_NEAR((a)[i], (b)[i], p);              \
    }                                              \
  } while (0)

#define EXPECT_VECTOR_NEAR(a, b)                   \
  do {                                             \
    ASSERT_EQ((a).size(), (b).size());             \
    for (std::size_t i = 0; i < (a).size(); ++i) { \
      EXPECT_NEAR((a)[i], (b)[i], 1e-14);          \
    }                                              \
  } while (0)

template <int Dim>
Tensor<double> trigonometric_tensor(std::size_t n_samples);
template <>
inline Tensor<double> trigonometric_tensor<1>(const std::size_t n_samples) {
  constexpr double frequency = 2.0;
  constexpr double max = 2 * std::numbers::pi;
  Tensor<double> out(std::vector<std::size_t>{n_samples});
  const double coeff = max / static_cast<double>(n_samples);
  for (std::size_t i = 0; i < n_samples; ++i) {
    const double x = frequency * static_cast<double>(i) * coeff;
    out(i) = std::sin(x) + 0.5 * std::cos(x);
  }
  return out;
}
template <>
inline Tensor<double> trigonometric_tensor<2>(const std::size_t n_samples) {
  constexpr double frequency_x = 2.0;
  constexpr double frequency_y = 1.0;
  constexpr double max = 2 * std::numbers::pi;
  Tensor<double> out(std::vector<std::size_t>{n_samples, n_samples});
  const double coeff = max / static_cast<double>(n_samples);
  for (std::size_t i = 0; i < n_samples; ++i) {
    for (std::size_t j = 0; j < n_samples; ++j) {
      const double x = frequency_x * static_cast<double>(i) * coeff;
      const double y = frequency_y * static_cast<double>(j) * coeff;
      out(i, j) = 0.5 * std::sin(x) + std::cos(y);
    }
  }
  return out;
}
template <>
inline Tensor<double> trigonometric_tensor<3>(const std::size_t n_samples) {
  constexpr double frequency_x = 2.0;
  constexpr double frequency_y = 1.0;
  constexpr double frequency_z = 3.0;
  const double max = 2 * std::numbers::pi;
  Tensor<double> out(std::vector<std::size_t>{n_samples, n_samples, n_samples});
  const double coeff = max / static_cast<double>(n_samples);
  for (std::size_t i = 0; i < n_samples; ++i) {
    for (std::size_t j = 0; j < n_samples; ++j) {
      for (std::size_t k = 0; k < n_samples; ++k) {
        const double x = frequency_x * static_cast<double>(i) * coeff;
        const double y = frequency_y * static_cast<double>(j) * coeff;
        const double z = frequency_z * static_cast<double>(k) * coeff;
        out(i, j, k) = 0.5 * std::sin(x) + std::cos(y) + 2 * std::sin(z) + 0.1 * std::cos(z);
      }
    }
  }
  return out;
}

TEST(TestTensor, Constructor0D) {
  const Tensor t{std::vector<double>{}, std::vector<std::size_t>{}};
  EXPECT_EQ(t.size(), 0u);
}

TEST(TestTensor, Constructor0DInvalidElements) {
  EXPECT_THROW(Tensor(std::vector{1.0}, std::vector<std::size_t>{2}), LucidInvalidArgumentException);
}

TEST(TestTensor, Pick0D) {
  const Tensor t{std::vector<double>{}, std::vector<std::size_t>{}};
  EXPECT_THROW(t(std::vector{0l}), LucidInvalidArgumentException);
}

TEST(TestTensor, Constructor1D) {
  const Tensor t{std::vector{1.0, 2.0}, std::vector{2ul}};
  EXPECT_DOUBLE_EQ(t(std::vector{0ul}), 1.0);
  EXPECT_DOUBLE_EQ(t(std::vector{1ul}), 2.0);
}

TEST(TestTensor, Constructor1DInvalidElements) {
  EXPECT_THROW(Tensor(std::vector{1.0}, std::vector{2ul}), LucidInvalidArgumentException);
}

TEST(TestTensor, Pick1D) {
  const Tensor t{std::vector{1.0, 2.0}, std::vector{2ul}};
  EXPECT_THROW(t(std::vector{0ul, 0ul}), LucidInvalidArgumentException);
}

TEST(TestTensor, Constructor2D) {
  const Tensor t{std::vector{1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0}, std::vector{2ul, 4ul}};
  EXPECT_EQ(t.size(), 8u);
}

TEST(TestTensor, Constructor2DInvalidElements) {
  EXPECT_THROW(Tensor(std::vector{1.0, 2.0, 3.0, 4.0, 5.0, 6.0}, std::vector{2ul, 4ul}), LucidInvalidArgumentException);
}

TEST(TestTensor, Pick2D) {
  const Tensor t{std::vector{1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0}, std::vector{2ul, 4ul}};
  EXPECT_DOUBLE_EQ(t(std::vector{0ul, 0ul}), 1.0);
  EXPECT_DOUBLE_EQ(t(std::vector{0ul, 1ul}), 2.0);
  EXPECT_DOUBLE_EQ(t(std::vector{0ul, 2ul}), 3.0);
  EXPECT_DOUBLE_EQ(t(std::vector{0ul, 3ul}), 4.0);
  EXPECT_DOUBLE_EQ(t(std::vector{1ul, 0ul}), 5.0);
  EXPECT_DOUBLE_EQ(t(std::vector{1ul, 1ul}), 6.0);
  EXPECT_DOUBLE_EQ(t(std::vector{1ul, 2ul}), 7.0);
  EXPECT_DOUBLE_EQ(t(std::vector{1ul, 3ul}), 8.0);
}

TEST(TestTensor, Constructor3D) {
  const Tensor t{std::vector{1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 11.0, 12.0},
                 std::vector{2ul, 3ul, 2ul}};
  EXPECT_EQ(t.size(), 12u);
}

TEST(TestTensor, Constructor3DInvalidElements) {
  EXPECT_THROW(Tensor(std::vector{1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 11.0}, std::vector{2ul, 3ul, 2ul}),
               LucidInvalidArgumentException);
}

TEST(TestTensor, Pick3D) {
  const Tensor t{std::vector{1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 11.0, 12.0},
                 std::vector{2ul, 3ul, 2ul}};
  EXPECT_DOUBLE_EQ(t(std::vector{0ul, 0ul, 0ul}), 1.0);
  EXPECT_DOUBLE_EQ(t(std::vector{0ul, 0ul, 1ul}), 2.0);
  EXPECT_DOUBLE_EQ(t(std::vector{0ul, 1ul, 0ul}), 3.0);
  EXPECT_DOUBLE_EQ(t(std::vector{0ul, 1ul, 1ul}), 4.0);
  EXPECT_DOUBLE_EQ(t(std::vector{0ul, 2ul, 0ul}), 5.0);
  EXPECT_DOUBLE_EQ(t(std::vector{0ul, 2ul, 1ul}), 6.0);
  EXPECT_DOUBLE_EQ(t(std::vector{1ul, 0ul, 0ul}), 7.0);
  EXPECT_DOUBLE_EQ(t(std::vector{1ul, 0ul, 1ul}), 8.0);
  EXPECT_DOUBLE_EQ(t(std::vector{1ul, 1ul, 0ul}), 9.0);
  EXPECT_DOUBLE_EQ(t(std::vector{1ul, 1ul, 1ul}), 10.0);
  EXPECT_DOUBLE_EQ(t(std::vector{1ul, 2ul, 0ul}), 11.0);
  EXPECT_DOUBLE_EQ(t(std::vector{1ul, 2ul, 1ul}), 12.0);
}

TEST(TestTensor, Pick3DTemplate) {
  const Tensor t{std::vector{1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 11.0, 12.0},
                 std::vector{2ul, 3ul, 2ul}};
  EXPECT_DOUBLE_EQ(t(0ul, 0ul, 0l), 1.0);
  EXPECT_DOUBLE_EQ(t(0ul, 0ul, 1l), 2.0);
  EXPECT_DOUBLE_EQ(t(0ul, 1ul, 0l), 3.0);
  EXPECT_DOUBLE_EQ(t(0ul, 1ul, 1l), 4.0);
  EXPECT_DOUBLE_EQ(t(0ul, 2ul, 0l), 5.0);
  EXPECT_DOUBLE_EQ(t(0ul, 2ul, 1l), 6.0);
  EXPECT_DOUBLE_EQ(t(1ul, 0ul, 0l), 7.0);
  EXPECT_DOUBLE_EQ(t(1ul, 0ul, 1l), 8.0);
  EXPECT_DOUBLE_EQ(t(1ul, 1ul, 0l), 9.0);
  EXPECT_DOUBLE_EQ(t(1ul, 1ul, 1l), 10.0);
  EXPECT_DOUBLE_EQ(t(1ul, 2ul, 0l), 11.0);
  EXPECT_DOUBLE_EQ(t(1ul, 2ul, 1l), 12.0);
}

TEST(TestTensor, Reshape) {
  Tensor<double> tensor{{1.0, 2.0, 3.0, 4.0, 5.0, 6.0}, {2ul, 3ul}};
  const std::vector<std::size_t> new_dims{3ul, 2ul};
  tensor.reshape(new_dims);
  EXPECT_EQ(tensor.dimensions(), new_dims);
  EXPECT_DOUBLE_EQ(tensor(0ul, 0ul), 1.0);
  EXPECT_DOUBLE_EQ(tensor(0ul, 1ul), 2.0);
  EXPECT_DOUBLE_EQ(tensor(1ul, 0ul), 3.0);
  EXPECT_DOUBLE_EQ(tensor(1ul, 1ul), 4.0);
  EXPECT_DOUBLE_EQ(tensor(2ul, 0ul), 5.0);
  EXPECT_DOUBLE_EQ(tensor(2ul, 1ul), 6.0);
}

TEST(TestTensor, ReshapeReduceRank) {
  Tensor<double> tensor{{1.0, 2.0, 3.0, 4.0, 5.0, 6.0}, {2ul, 3ul}};
  const std::vector<std::size_t> new_dims{6ul};
  tensor.reshape(new_dims);
  EXPECT_EQ(tensor.dimensions(), new_dims);
  EXPECT_DOUBLE_EQ(tensor(0ul), 1.0);
  EXPECT_DOUBLE_EQ(tensor(1ul), 2.0);
  EXPECT_DOUBLE_EQ(tensor(2ul), 3.0);
  EXPECT_DOUBLE_EQ(tensor(3ul), 4.0);
  EXPECT_DOUBLE_EQ(tensor(4ul), 5.0);
  EXPECT_DOUBLE_EQ(tensor(5ul), 6.0);
}

TEST(TestTensor, ReshapeIncreaseRank) {
  Tensor<double> tensor{{1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 11.0, 12.0}, {3ul, 4ul}};
  const std::vector<std::size_t> new_dims{2ul, 3ul, 2ul, 1ul, 1ul};
  tensor.reshape(new_dims);
  EXPECT_EQ(tensor.dimensions(), new_dims);
  EXPECT_DOUBLE_EQ(tensor(0ul, 0ul, 0ul, 0ul, 0ul), 1.0);
  EXPECT_DOUBLE_EQ(tensor(0ul, 0ul, 1ul, 0ul, 0ul), 2.0);
  EXPECT_DOUBLE_EQ(tensor(0ul, 1ul, 0ul, 0ul, 0ul), 3.0);
  EXPECT_DOUBLE_EQ(tensor(0ul, 1ul, 1ul, 0ul, 0ul), 4.0);
  EXPECT_DOUBLE_EQ(tensor(0ul, 2ul, 0ul, 0ul, 0ul), 5.0);
  EXPECT_DOUBLE_EQ(tensor(0ul, 2ul, 1ul, 0ul, 0ul), 6.0);
  EXPECT_DOUBLE_EQ(tensor(1ul, 0ul, 0ul, 0ul, 0ul), 7.0);
  EXPECT_DOUBLE_EQ(tensor(1ul, 0ul, 1ul, 0ul, 0ul), 8.0);
  EXPECT_DOUBLE_EQ(tensor(1ul, 1ul, 0ul, 0ul, 0ul), 9.0);
  EXPECT_DOUBLE_EQ(tensor(1ul, 1ul, 1ul, 0ul, 0ul), 10.0);
  EXPECT_DOUBLE_EQ(tensor(1ul, 2ul, 0ul, 0ul, 0ul), 11.0);
  EXPECT_DOUBLE_EQ(tensor(1ul, 2ul, 1ul, 0ul, 0ul), 12.0);
}

TEST(TestTensor, ReshapeInvalidSize) {
  Tensor<double> tensor{{1.0, 2.0, 3.0, 4.0, 5.0, 6.0}, {2ul, 3ul}};
  const std::vector<std::size_t> new_dims{3ul};
  EXPECT_THROW(tensor.reshape(new_dims), LucidInvalidArgumentException);
}

TEST(TestTensor, PermuteIdentity1D) {
  Tensor<double> tensor{Tensor<double>{std::vector{1.0, 2.0, 3.0}, std::vector{3ul}}.permute(0ul)};
  EXPECT_EQ(tensor.dimensions(), std::vector<std::size_t>({3ul}));
  EXPECT_DOUBLE_EQ(tensor(0ul), 1.0);
  EXPECT_DOUBLE_EQ(tensor(1ul), 2.0);
  EXPECT_DOUBLE_EQ(tensor(2ul), 3.0);
}
TEST(TestTensor, Permute2D) {
  Tensor<double> tensor{
      Tensor<double>{std::vector{1.0, 2.0, 3.0, 4.0, 5.0, 6.0}, std::vector{2ul, 3ul}}.permute(1ul, 0ul)};
  EXPECT_EQ(tensor.dimensions(), std::vector<std::size_t>({3ul, 2ul}));
  EXPECT_DOUBLE_EQ(tensor(0ul, 0ul), 1.0);
  EXPECT_DOUBLE_EQ(tensor(0ul, 1ul), 4.0);
  EXPECT_DOUBLE_EQ(tensor(1ul, 0ul), 2.0);
  EXPECT_DOUBLE_EQ(tensor(1ul, 1ul), 5.0);
  EXPECT_DOUBLE_EQ(tensor(2ul, 0ul), 3.0);
  EXPECT_DOUBLE_EQ(tensor(2ul, 1ul), 6.0);
}
TEST(TestTensor, PermuteVector2D) {
  Tensor<double> tensor{
      Tensor<double>{std::vector{1.0, 2.0, 3.0, 4.0, 5.0, 6.0}, std::vector{2ul, 3ul}}.permute({1ul, 0ul})};
  EXPECT_EQ(tensor.dimensions(), std::vector<std::size_t>({3ul, 2ul}));
  EXPECT_DOUBLE_EQ(tensor(0ul, 0ul), 1.0);
  EXPECT_DOUBLE_EQ(tensor(0ul, 1ul), 4.0);
  EXPECT_DOUBLE_EQ(tensor(1ul, 0ul), 2.0);
  EXPECT_DOUBLE_EQ(tensor(1ul, 1ul), 5.0);
  EXPECT_DOUBLE_EQ(tensor(2ul, 0ul), 3.0);
  EXPECT_DOUBLE_EQ(tensor(2ul, 1ul), 6.0);
}
// 2D tensor, identity permutation
TEST(TestTensor, PermuteIdentity2D) {
  Tensor<double> tensor{
      Tensor<double>{std::vector{1.0, 2.0, 3.0, 4.0, 5.0, 6.0}, std::vector{2ul, 3ul}}.permute(0ul, 1ul)};
  EXPECT_EQ(tensor.dimensions(), std::vector<std::size_t>({2ul, 3ul}));
  EXPECT_DOUBLE_EQ(tensor(0ul, 0ul), 1.0);
  EXPECT_DOUBLE_EQ(tensor(0ul, 1ul), 2.0);
  EXPECT_DOUBLE_EQ(tensor(0ul, 2ul), 3.0);
  EXPECT_DOUBLE_EQ(tensor(1ul, 0ul), 4.0);
  EXPECT_DOUBLE_EQ(tensor(1ul, 1ul), 5.0);
  EXPECT_DOUBLE_EQ(tensor(1ul, 2ul), 6.0);
}

TEST(TestTensor, PermuteVector3D) {
  Tensor<int> tensor{
      Tensor<int>{std::vector<int>{1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12}, std::vector{2ul, 2ul, 3ul}}.permute(2ul, 0ul,
                                                                                                               1ul)};
  EXPECT_EQ(tensor.dimensions(), std::vector<std::size_t>({3ul, 2ul, 2ul}));
  EXPECT_EQ(tensor(0, 0, 0), 1);
  EXPECT_EQ(tensor(0, 0, 1), 4);
  EXPECT_EQ(tensor(0, 1, 0), 7);
  EXPECT_EQ(tensor(0, 1, 1), 10);
  EXPECT_EQ(tensor(1, 0, 0), 2);
  EXPECT_EQ(tensor(1, 0, 1), 5);
  EXPECT_EQ(tensor(1, 1, 0), 8);
  EXPECT_EQ(tensor(1, 1, 1), 11);
  EXPECT_EQ(tensor(2, 0, 0), 3);
  EXPECT_EQ(tensor(2, 0, 1), 6);
  EXPECT_EQ(tensor(2, 1, 0), 9);
  EXPECT_EQ(tensor(2, 1, 1), 12);
}
TEST(TestTensor, PermuteAllDifferent3D) {
  Tensor<int> tensor{Tensor<int>{
      std::vector<int>{1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24},
      std::vector{2ul, 3ul, 4ul}}
                         .permute(2ul, 0ul, 1ul)};
  EXPECT_EQ(tensor.dimensions(), std::vector<std::size_t>({4ul, 2ul, 3ul}));
  EXPECT_EQ(tensor(0, 0, 0), 1);
  EXPECT_EQ(tensor(0, 0, 1), 5);
  EXPECT_EQ(tensor(0, 0, 2), 9);
  EXPECT_EQ(tensor(0, 1, 0), 13);
  EXPECT_EQ(tensor(0, 1, 1), 17);
  EXPECT_EQ(tensor(0, 1, 2), 21);
  EXPECT_EQ(tensor(1, 0, 0), 2);
  EXPECT_EQ(tensor(1, 0, 1), 6);
  EXPECT_EQ(tensor(1, 0, 2), 10);
  EXPECT_EQ(tensor(1, 1, 0), 14);
  EXPECT_EQ(tensor(1, 1, 1), 18);
  EXPECT_EQ(tensor(1, 1, 2), 22);
  EXPECT_EQ(tensor(2, 0, 0), 3);
  EXPECT_EQ(tensor(2, 0, 1), 7);
  EXPECT_EQ(tensor(2, 0, 2), 11);
  EXPECT_EQ(tensor(2, 1, 0), 15);
  EXPECT_EQ(tensor(2, 1, 1), 19);
  EXPECT_EQ(tensor(2, 1, 2), 23);
  EXPECT_EQ(tensor(3, 0, 0), 4);
  EXPECT_EQ(tensor(3, 0, 1), 8);
  EXPECT_EQ(tensor(3, 0, 2), 12);
  EXPECT_EQ(tensor(3, 1, 0), 16);
  EXPECT_EQ(tensor(3, 1, 1), 20);
  EXPECT_EQ(tensor(3, 1, 2), 24);
}
TEST(TestTensor, PermutePartial3D) {
  Tensor<int> tensor{Tensor<int>{
      std::vector<int>{1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24},
      std::vector{2ul, 3ul, 4ul}}
                         .permute(0, 2, 1)};
  EXPECT_EQ(tensor.dimensions(), std::vector<std::size_t>({2ul, 4ul, 3ul}));
  EXPECT_THAT(tensor.data(), ::testing::ElementsAre(1, 5, 9, 2, 6, 10, 3, 7, 11, 4, 8, 12, 13, 17, 21, 14, 18, 22, 15,
                                                    19, 23, 16, 20, 24));
}
TEST(TestTensor, PermutePartial4D) {
  Tensor<int> tensor{Tensor<int>{
      std::vector<int>{1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24},
      std::vector{2ul, 3ul, 4ul, 1ul}}
                         .permute(1ul, 0ul)};
  EXPECT_EQ(tensor.dimensions(), std::vector<std::size_t>({3ul, 2ul, 4ul, 1ul}));
  EXPECT_EQ(tensor(0, 0, 0, 0), 1);
  EXPECT_EQ(tensor(0, 0, 1, 0), 2);
  EXPECT_EQ(tensor(0, 0, 2, 0), 3);
  EXPECT_EQ(tensor(0, 0, 3, 0), 4);
  EXPECT_EQ(tensor(0, 1, 0, 0), 13);
  EXPECT_EQ(tensor(0, 1, 1, 0), 14);
  EXPECT_EQ(tensor(0, 1, 2, 0), 15);
  EXPECT_EQ(tensor(0, 1, 3, 0), 16);
  EXPECT_EQ(tensor(1, 0, 0, 0), 5);
  EXPECT_EQ(tensor(1, 0, 1, 0), 6);
  EXPECT_EQ(tensor(1, 0, 2, 0), 7);
  EXPECT_EQ(tensor(1, 0, 3, 0), 8);
  EXPECT_EQ(tensor(1, 1, 0, 0), 17);
  EXPECT_EQ(tensor(1, 1, 1, 0), 18);
  EXPECT_EQ(tensor(1, 1, 2, 0), 19);
  EXPECT_EQ(tensor(1, 1, 3, 0), 20);
  EXPECT_EQ(tensor(2, 0, 0, 0), 9);
  EXPECT_EQ(tensor(2, 0, 1, 0), 10);
  EXPECT_EQ(tensor(2, 0, 2, 0), 11);
  EXPECT_EQ(tensor(2, 0, 3, 0), 12);
  EXPECT_EQ(tensor(2, 1, 0, 0), 21);
  EXPECT_EQ(tensor(2, 1, 1, 0), 22);
  EXPECT_EQ(tensor(2, 1, 2, 0), 23);
  EXPECT_EQ(tensor(2, 1, 3, 0), 24);
}

// Permute with repeated application (round-trip)
TEST(TestTensor, PermuteRoundTrip3D) {
  Tensor<int> tensor{Tensor<int>{std::vector<int>{1, 2, 3, 4, 5, 6, 7, 8}, std::vector{2ul, 2ul, 2ul}}
                         .permute(2ul, 0ul, 1ul)
                         .permute(1ul, 2ul, 0ul)};
  EXPECT_EQ(tensor.dimensions(), std::vector<std::size_t>({2ul, 2ul, 2ul}));

  EXPECT_EQ(tensor(0, 0, 0), 1);
  EXPECT_EQ(tensor(0, 0, 1), 2);
  EXPECT_EQ(tensor(0, 1, 0), 3);
  EXPECT_EQ(tensor(0, 1, 1), 4);
  EXPECT_EQ(tensor(1, 0, 0), 5);
  EXPECT_EQ(tensor(1, 0, 1), 6);
  EXPECT_EQ(tensor(1, 1, 0), 7);
  EXPECT_EQ(tensor(1, 1, 1), 8);
}
TEST(TestTensor, PermuteEmpty) {
  Tensor<float> tensor{Tensor<float>{std::vector<float>{}, std::vector<std::size_t>{2ul, 0ul}}.permute(1ul, 0ul)};
  EXPECT_EQ(tensor.dimensions(), std::vector<std::size_t>({0ul, 2ul}));
}
TEST(TestTensor, PermuteHighDimemsion) {
  Tensor<float> tensor{
      Tensor<float>{std::vector<float>{}, std::vector<std::size_t>{2ul, 0ul, 5ul}}.permute(2ul, 0ul, 1ul)};
  EXPECT_EQ(tensor.dimensions(), std::vector<std::size_t>({5ul, 2ul, 0ul}));
}

TEST(TestTensor, FFT1) {
  const Tensor t{std::vector{1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0}, std::vector{10ul}};
  const Tensor<std::complex<double>> fft = t.fft();
  EXPECT_EQ(fft.size(), 10u);
  EXPECT_COMPLEX_DOUBLE_EQ(fft(0ul), std::complex<double>(55.0, 0.0));
  EXPECT_COMPLEX_DOUBLE_EQ(fft(1ul), std::complex<double>(-5.0, 15.388417685876265));
  EXPECT_COMPLEX_DOUBLE_EQ(fft(2ul), std::complex<double>(-5.0, 6.8819096023558677));
  EXPECT_COMPLEX_DOUBLE_EQ(fft(3ul), std::complex<double>(-5.0, 3.6327126400268037));
  EXPECT_COMPLEX_DOUBLE_EQ(fft(4ul), std::complex<double>(-5.0, 1.6245984811645311));
  EXPECT_COMPLEX_DOUBLE_EQ(fft(5ul), std::complex<double>(-5.0, 0.0));
}

TEST(TestTensor, IFFT1) {
  const Tensor t{std::vector{1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0}, std::vector{10ul}};
  const Tensor<double> ifft = t.fft().ifft();
  EXPECT_EQ(ifft.size(), 10u);
  EXPECT_DOUBLE_EQ(ifft(0ul), 1.0);
  EXPECT_DOUBLE_EQ(ifft(1ul), 2.0);
  EXPECT_DOUBLE_EQ(ifft(2ul), 3.0);
  EXPECT_DOUBLE_EQ(ifft(3ul), 4.0);
  EXPECT_DOUBLE_EQ(ifft(4ul), 5.0);
  EXPECT_DOUBLE_EQ(ifft(5ul), 6.0);
  EXPECT_DOUBLE_EQ(ifft(6ul), 7.0);
  EXPECT_DOUBLE_EQ(ifft(7ul), 8.0);
  EXPECT_DOUBLE_EQ(ifft(8ul), 9.0);
  EXPECT_DOUBLE_EQ(ifft(9ul), 10.0);
}

TEST(TestTensor, FFT2) {
  const Tensor t{std::vector{1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0}, std::vector{2ul, 5ul}};
  const Tensor<std::complex<double>> fft = t.fft();
  EXPECT_EQ(fft.size(), 10u);
  EXPECT_COMPLEX_DOUBLE_EQ(fft(0ul, 0ul), std::complex<double>(55.0, 0.0));
  EXPECT_COMPLEX_DOUBLE_EQ(fft(0ul, 1ul), std::complex<double>(-5.0, 6.8819096023558677));
  EXPECT_COMPLEX_DOUBLE_EQ(fft(0ul, 2ul), std::complex<double>(-5.0, 1.6245984811645311));
  EXPECT_COMPLEX_DOUBLE_EQ(fft(0ul, 3ul), std::complex<double>(-5.0, -1.6245984811645311));
  EXPECT_COMPLEX_DOUBLE_EQ(fft(0ul, 4ul), std::complex<double>(-5.0, -6.8819096023558677));
  EXPECT_COMPLEX_DOUBLE_EQ(fft(1ul, 0ul), std::complex<double>(-25.0, 0.0));
  EXPECT_COMPLEX_DOUBLE_EQ(fft(1ul, 1ul), std::complex<double>(0.0, 0.0));
  EXPECT_COMPLEX_DOUBLE_EQ(fft(1ul, 2ul), std::complex<double>(0, 0.0));
  EXPECT_COMPLEX_DOUBLE_EQ(fft(1ul, 3ul), std::complex<double>(0, 0.0));
  EXPECT_COMPLEX_DOUBLE_EQ(fft(1ul, 4ul), std::complex<double>(0.0, 0.0));
}

TEST(TestTensor, IFFT2) {
  const Tensor t{std::vector{1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0}, std::vector{2ul, 5ul}};
  const Tensor<double> fft = t.fft().ifft();
  EXPECT_EQ(fft.size(), 10u);
  EXPECT_DOUBLE_EQ(fft(0ul, 0ul), 1.0);
  EXPECT_DOUBLE_EQ(fft(0ul, 1ul), 2.0);
  EXPECT_DOUBLE_EQ(fft(0ul, 2ul), 3.0);
  EXPECT_DOUBLE_EQ(fft(0ul, 3ul), 4.0);
  EXPECT_DOUBLE_EQ(fft(0ul, 4ul), 5.0);
  EXPECT_DOUBLE_EQ(fft(1ul, 0ul), 6.0);
  EXPECT_DOUBLE_EQ(fft(1ul, 1ul), 7.0);
  EXPECT_DOUBLE_EQ(fft(1ul, 2ul), 8.0);
  EXPECT_DOUBLE_EQ(fft(1ul, 3ul), 9.0);
  EXPECT_DOUBLE_EQ(fft(1ul, 4ul), 10.0);
}

TEST(TestTensor, FFT3) {
  const Tensor t{std::vector{1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 11.0, 12.0},
                 std::vector{2ul, 2ul, 3ul}};
  const Tensor<std::complex<double>> fft = t.fft();
  EXPECT_EQ(fft.size(), 12u);
  EXPECT_COMPLEX_DOUBLE_EQ(fft(0ul, 0ul, 0ul), std::complex<double>(78.0, 0.0));
  EXPECT_COMPLEX_DOUBLE_EQ(fft(0ul, 0ul, 1ul), std::complex<double>(-6.0, 3.4641016151377544));
  EXPECT_COMPLEX_DOUBLE_EQ(fft(0ul, 0ul, 2ul), std::complex<double>(-6.0, -3.4641016151377544));
  EXPECT_COMPLEX_DOUBLE_EQ(fft(0ul, 1ul, 0ul), std::complex<double>(-18.0, 0));
  EXPECT_COMPLEX_DOUBLE_EQ(fft(0ul, 1ul, 1ul), std::complex<double>(0, 0));
  EXPECT_COMPLEX_DOUBLE_EQ(fft(0ul, 1ul, 2ul), std::complex<double>(0, 0));
  EXPECT_COMPLEX_DOUBLE_EQ(fft(1ul, 0ul, 0ul), std::complex<double>(-36.0, 0));
  EXPECT_COMPLEX_DOUBLE_EQ(fft(1ul, 0ul, 1ul), std::complex<double>(0, 0));
  EXPECT_COMPLEX_DOUBLE_EQ(fft(1ul, 0ul, 2ul), std::complex<double>(0, 0));
  EXPECT_COMPLEX_DOUBLE_EQ(fft(1ul, 1ul, 0ul), std::complex<double>(0, 0));
  EXPECT_COMPLEX_DOUBLE_EQ(fft(1ul, 1ul, 1ul), std::complex<double>(0, 0));
}

TEST(TestTensor, IFFT3) {
  const Tensor t{std::vector{1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 11.0, 12.0},
                 std::vector{2ul, 2ul, 3ul}};
  const Tensor<double> fft = t.fft().ifft();
  EXPECT_EQ(fft.size(), 12u);
  EXPECT_DOUBLE_EQ(fft(0ul, 0ul, 0ul), 1.0);
  EXPECT_DOUBLE_EQ(fft(0ul, 0ul, 1ul), 2.0);
  EXPECT_DOUBLE_EQ(fft(0ul, 0ul, 2ul), 3.0);
  EXPECT_DOUBLE_EQ(fft(0ul, 1ul, 0ul), 4.0);
  EXPECT_DOUBLE_EQ(fft(0ul, 1ul, 1ul), 5.0);
  EXPECT_DOUBLE_EQ(fft(0ul, 1ul, 2ul), 6.0);
  EXPECT_DOUBLE_EQ(fft(1ul, 0ul, 0ul), 7.0);
  EXPECT_DOUBLE_EQ(fft(1ul, 0ul, 1ul), 8.0);
  EXPECT_DOUBLE_EQ(fft(1ul, 0ul, 2ul), 9.0);
  EXPECT_DOUBLE_EQ(fft(1ul, 1ul, 0ul), 10.0);
  EXPECT_DOUBLE_EQ(fft(1ul, 1ul, 1ul), 11.0);
  EXPECT_DOUBLE_EQ(fft(1ul, 1ul, 2ul), 12.0);
}

TEST(TestTensor, ToVector) {
  const Tensor t{std::vector{1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 11.0, 12.0},
                 std::vector{2ul, 2ul, 3ul}};
  const Eigen::Map<const Eigen::VectorX<double>> v{t};
  EXPECT_EQ(v.size(), 12);
  EXPECT_DOUBLE_EQ(v(0), 1.0);
  EXPECT_DOUBLE_EQ(v(1), 2.0);
  EXPECT_DOUBLE_EQ(v(2), 3.0);
  EXPECT_DOUBLE_EQ(v(3), 4.0);
  EXPECT_DOUBLE_EQ(v(4), 5.0);
  EXPECT_DOUBLE_EQ(v(5), 6.0);
  EXPECT_DOUBLE_EQ(v(6), 7.0);
  EXPECT_DOUBLE_EQ(v(7), 8.0);
  EXPECT_DOUBLE_EQ(v(8), 9.0);
  EXPECT_DOUBLE_EQ(v(9), 10.0);
  EXPECT_DOUBLE_EQ(v(10), 11.0);
  EXPECT_DOUBLE_EQ(v(11), 12.0);
}

TEST(TestTensor, ToMatrix) {
  const Tensor t{std::vector{1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 11.0, 12.0}, std::vector{2ul, 6ul}};
  const Eigen::Map<const Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>> m{t};
  EXPECT_EQ(m.size(), 12);
  EXPECT_DOUBLE_EQ(m(0, 0), 1.0);
  EXPECT_DOUBLE_EQ(m(0, 1), 2.0);
  EXPECT_DOUBLE_EQ(m(0, 2), 3.0);
  EXPECT_DOUBLE_EQ(m(0, 3), 4.0);
  EXPECT_DOUBLE_EQ(m(0, 4), 5.0);
  EXPECT_DOUBLE_EQ(m(0, 5), 6.0);
  EXPECT_DOUBLE_EQ(m(1, 0), 7.0);
  EXPECT_DOUBLE_EQ(m(1, 1), 8.0);
  EXPECT_DOUBLE_EQ(m(1, 2), 9.0);
  EXPECT_DOUBLE_EQ(m(1, 3), 10.0);
  EXPECT_DOUBLE_EQ(m(1, 4), 11.0);
  EXPECT_DOUBLE_EQ(m(1, 5), 12.0);
}

TEST(TestTensor, PadWithZeros) {
  Tensor<double> tensor({1.0, 2.0, 3.0, 4.0}, {2ul, 2ul});
  Tensor<double> padded_tensor = tensor.pad({{1, 1}, {1, 1}}, 0.0);
  EXPECT_THAT(padded_tensor.data(),
              ::testing::ElementsAre(0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 2.0, 0.0, 0.0, 3.0, 4.0, 0.0, 0.0, 0.0, 0.0, 0.0));
  EXPECT_EQ(padded_tensor.dimensions(), (std::vector<std::size_t>{4, 4}));
}

TEST(TestTensor, PadWithConstantValue) {
  Tensor<double> tensor({1.0, 2.0, 3.0, 4.0}, {2ul, 2ul});
  Tensor<double> padded_tensor = tensor.pad({{1, 1}, {1, 1}}, 5.0);
  EXPECT_THAT(padded_tensor.data(),
              ::testing::ElementsAre(5.0, 5.0, 5.0, 5.0, 5.0, 1.0, 2.0, 5.0, 5.0, 3.0, 4.0, 5.0, 5.0, 5.0, 5.0, 5.0));
  EXPECT_EQ(padded_tensor.dimensions(), (std::vector<std::size_t>{4, 4}));
}

TEST(TestTensor, PadWithDifferentPadding) {
  Tensor<double> tensor({1.0, 2.0, 3.0, 4.0}, {2ul, 2ul});
  Tensor<double> padded_tensor = tensor.pad({{1, 2}, {2, 1}}, 0.0);
  EXPECT_THAT(tensor.pad({{1, 2}, {2, 1}}).data(),
              ::testing::ElementsAre(0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 2.0, 0.0, 0.0, 0.0, 3.0, 4.0, 0.0, 0.0,
                                     0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0));
  EXPECT_EQ(padded_tensor.dimensions(), (std::vector<std::size_t>{5, 5}));
}

TEST(TestTensor, Pad3D) {
  Tensor<double> tensor({1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 11.0, 12.0}, {2ul, 2ul, 3ul});
  EXPECT_THAT(tensor.pad({{1, 0}, {0, 2}, {2, 1}}).data(),
              ::testing::ElementsAre(0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 2,
                                     3, 0, 0, 0, 4, 5, 6, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 7, 8, 9, 0, 0, 0,
                                     10, 11, 12, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0));
}

TEST(TestTensor, PadMiddle) {
  Tensor<double> t{std::vector<double>{1, 2, 3, 4, 5, 6}, std::vector<std::size_t>{3, 2}};
  EXPECT_THAT(t.pad({2, 3}, {2, 1}).data(),
              ::testing::ElementsAre(1, 0, 0, 0, 2, 3, 0, 0, 0, 4, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 5, 0, 0, 0, 6));
}

TEST(TestTensor, PadCross) {
  Tensor<double> t{std::vector<double>{1, 2, 3, 4, 5, 6}, std::vector<std::size_t>{3, 2}};
  EXPECT_THAT(t.pad({1, 1}, {1, 1}).data(), ::testing::ElementsAre(1, 0, 2, 0, 0, 0, 3, 0, 4, 5, 0, 6));
}

TEST(TestTensor, PadMiddleExtremes) {
  Tensor<double> t{std::vector<double>{1, 2, 3, 4, 5, 6}, std::vector<std::size_t>{3, 2}};
  EXPECT_THAT(t.pad({2, 3}, {0, 2}).data(),
              ::testing::ElementsAre(0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 2, 0, 0, 0, 3, 4, 0, 0, 0, 5, 6, 0, 0, 0));
}

TEST(TestTensor, PadMiddleInvalid) {
  Tensor<double> t{std::vector<double>{1, 2, 3, 4, 5, 6}, std::vector<std::size_t>{3, 2}};
  EXPECT_THROW(t.pad({2, 3}, {10, 11}), LucidInvalidArgumentException);
}

TEST(TestTensor, PadHighRank) {
  Tensor<double> tensor(
      {1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 11.0, 12.0, 13.0, 14.0, 15.0, 16.0, 17.0, 18.0},
      {3ul, 1ul, 3ul, 2ul, 1ul});
  const Tensor<double> padded_tensor = tensor.pad({{1, 0}, {0, 2}, {2, 1}, {0, 0}, {1, 1}}, 0.0);
  std::array<double, 432> expected_value{
      0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,  0, 0, 0,  0, 0, 0,  0, 0, 0,  0, 0, 0,  0, 0, 0,  0, 0, 0, 0, 0, 0, 0,
      0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,  0, 0, 0,  0, 0, 0,  0, 0, 0,  0, 0, 0,  0, 0, 0,  0, 0, 0, 0, 0, 0, 0,
      0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,  0, 0, 0,  0, 0, 0,  0, 0, 0,  0, 0, 0,  0, 0, 0,  0, 0, 0, 0, 0, 0, 0,
      0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1,  0, 0, 2,  0, 0, 3,  0, 0, 4,  0, 0, 5,  0, 0, 6,  0, 0, 0, 0, 0, 0, 0,
      0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,  0, 0, 0,  0, 0, 0,  0, 0, 0,  0, 0, 0,  0, 0, 0,  0, 0, 0, 0, 0, 0, 0,
      0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,  0, 0, 0,  0, 0, 0,  0, 0, 0,  0, 0, 0,  0, 0, 0,  0, 0, 0, 0, 0, 0, 0,
      0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 7,  0, 0, 8,  0, 0, 9,  0, 0, 10, 0, 0, 11, 0, 0, 12, 0, 0, 0, 0, 0, 0, 0,
      0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,  0, 0, 0,  0, 0, 0,  0, 0, 0,  0, 0, 0,  0, 0, 0,  0, 0, 0, 0, 0, 0, 0,
      0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,  0, 0, 0,  0, 0, 0,  0, 0, 0,  0, 0, 0,  0, 0, 0,  0, 0, 0, 0, 0, 0, 0,
      0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 13, 0, 0, 14, 0, 0, 15, 0, 0, 16, 0, 0, 17, 0, 0, 18, 0, 0, 0, 0, 0, 0, 0,
      0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,  0, 0, 0,  0, 0, 0,  0, 0, 0,  0, 0, 0,  0, 0, 0,  0, 0, 0, 0, 0, 0, 0,
      0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,  0, 0, 0,  0, 0, 0,  0, 0, 0,  0, 0, 0,  0, 0, 0,  0, 0, 0, 0, 0, 0, 0};
  EXPECT_THAT(padded_tensor.data(), ::testing::ElementsAreArray(expected_value));
}

TEST(TestTensor, PadWithNegativePadding) {
  Tensor<double> tensor({1.0, 2.0, 3.0, 4.0}, {2ul, 2ul});
  EXPECT_THROW(tensor.pad({{-1, 1}, {1, 1}}, 0.0), LucidInvalidArgumentException);
}

TEST(TestTensor, UpsampleEvenToEven1D) {
  const Tensor<double> upsampled_tensor = trigonometric_tensor<1>(6ul).fft_upsample({42ul});
  EXPECT_VECTOR_NEAR(upsampled_tensor.data(), trigonometric_tensor<1>(42ul).data());
}

TEST(TestTensor, UpsampleEvenToOdd1D) {
  const Tensor<double> upsampled_tensor = trigonometric_tensor<1>(6ul).fft_upsample({43ul});
  EXPECT_VECTOR_NEAR(upsampled_tensor.data(), trigonometric_tensor<1>(43ul).data());
}

TEST(TestTensor, UpsampleOddToOdd1D) {
  const Tensor<double> upsampled_tensor = trigonometric_tensor<1>(7ul).fft_upsample({43ul});
  EXPECT_VECTOR_NEAR(upsampled_tensor.data(), trigonometric_tensor<1>(43ul).data());
}

TEST(TestTensor, UpsampleOddToEven1D) {
  const Tensor<double> upsampled_tensor = trigonometric_tensor<1>(7ul).fft_upsample({44ul});
  EXPECT_VECTOR_NEAR(upsampled_tensor.data(), trigonometric_tensor<1>(44ul).data());
}

TEST(TestTensor, UpsampleEvenToEven2D) {
  const Tensor<double> upsampled_tensor = trigonometric_tensor<2>(6ul).fft_upsample({20ul, 20ul});
  EXPECT_VECTOR_NEAR(upsampled_tensor.data(), trigonometric_tensor<2>(20ul).data());
}

TEST(TestTensor, UpsampleEvenToOdd2D) {
  const Tensor<double> upsampled_tensor = trigonometric_tensor<2>(6ul).fft_upsample({21ul, 21ul});
  EXPECT_VECTOR_NEAR(upsampled_tensor.data(), trigonometric_tensor<2>(21ul).data());
}

TEST(TestTensor, UpsampleOddToOdd2D) {
  const Tensor<double> upsampled_tensor = trigonometric_tensor<2>(7ul).fft_upsample({21ul, 21ul});
  EXPECT_VECTOR_NEAR(upsampled_tensor.data(), trigonometric_tensor<2>(21ul).data());
}

TEST(TestTensor, UpsampleOddToEven2D) {
  const Tensor<double> upsampled_tensor = trigonometric_tensor<2>(7ul).fft_upsample({26ul, 26ul});
  EXPECT_VECTOR_NEAR(upsampled_tensor.data(), trigonometric_tensor<2>(26ul).data());
}

TEST(TestTensor, UpsampleEvenToEven3D) {
  const Tensor<double> upsampled_tensor = trigonometric_tensor<3>(8ul).fft_upsample({16ul, 16ul, 16ul});
  EXPECT_VECTOR_NEAR(upsampled_tensor.data(), trigonometric_tensor<3>(16ul).data());
}

TEST(TestTensor, UpsampleEvenToOdd3D) {
  const Tensor<double> upsampled_tensor = trigonometric_tensor<3>(8l).fft_upsample({17ul, 17ul, 17ul});
  EXPECT_VECTOR_NEAR(upsampled_tensor.data(), trigonometric_tensor<3>(17ul).data());
}

TEST(TestTensor, UpsampleOddToOdd3D) {
  const Tensor<double> upsampled_tensor = trigonometric_tensor<3>(7ul).fft_upsample({17ul, 17ul, 17ul});
  EXPECT_VECTOR_NEAR(upsampled_tensor.data(), trigonometric_tensor<3>(17ul).data());
}

TEST(TestTensor, UpsampleOddToEven3D) {
  const Tensor<double> upsampled_tensor = trigonometric_tensor<3>(7ul).fft_upsample({18ul, 18ul, 18ul});
  EXPECT_VECTOR_NEAR(upsampled_tensor.data(), trigonometric_tensor<3>(18ul).data());
}

TEST(TestTensor, UpsampleDownsampleInvalid) {
  EXPECT_THROW(Tensor<double>{trigonometric_tensor<3>(7ul).fft_upsample({1ul, 1ul, 1ul})},
               LucidInvalidArgumentException);
}

TEST(TestTensor, UpsampleRankInvalid) {
  EXPECT_THROW(Tensor<double>{trigonometric_tensor<3>(7ul).fft_upsample({18ul})}, LucidInvalidArgumentException);
}

TEST(TestTensor, TensorIterator1D) {
  Tensor<int> tensor{std::vector<int>{1, 2, 3, 4, 5, 6}, std::vector<std::size_t>{6l}};
  EXPECT_THAT(std::vector<int>(tensor.begin(), tensor.end()), ::testing::ElementsAre(1, 2, 3, 4, 5, 6));
}

TEST(TestTensor, TensorIterator2D) {
  Tensor<int> tensor{std::vector<int>{1, 2, 3, 4, 5, 6}, std::vector<std::size_t>{3ul, 2ul}};
  EXPECT_THAT(std::vector<int>(tensor.begin(), tensor.end()), ::testing::ElementsAre(1, 2, 3, 4, 5, 6));
}

TEST(TestTensor, TensorIterator3D) {
  Tensor<int> tensor{std::vector<int>{1, 2, 3, 4, 5, 6}, std::vector<std::size_t>{3ul, 1ul, 2ul}};
  EXPECT_THAT(std::vector<int>(tensor.begin(), tensor.end()), ::testing::ElementsAre(1, 2, 3, 4, 5, 6));
}

TEST(TestTensor, ToMatrixInvalid) {
  const Tensor t{std::vector{1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 11.0, 12.0},
                 std::vector{2ul, 2ul, 3ul}};
  EXPECT_THROW(
      (static_cast<Eigen::Map<const Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>>>(t)),
      LucidNotSupportedException);
}
