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

TEST(TestTensor, Constructor0D) {
  const Tensor t{std::vector<double>{}, std::vector<std::size_t>{}};
  EXPECT_EQ(t.size(), 0u);
}

TEST(TestTensor, Constructor0DInvalidElements) {
  EXPECT_THROW(Tensor(std::vector{1.0}, std::vector<std::size_t>{}), LucidInvalidArgumentException);
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
  EXPECT_COMPLEX_DOUBLE_EQ(fft(0ul, 3ul), std::complex<double>(0, 0));
  EXPECT_COMPLEX_DOUBLE_EQ(fft(0ul, 4ul), std::complex<double>(0, 0));
  EXPECT_COMPLEX_DOUBLE_EQ(fft(1ul, 0ul), std::complex<double>(-25.0, 0.0));
  EXPECT_COMPLEX_DOUBLE_EQ(fft(1ul, 1ul), std::complex<double>(0.0, 0.0));
  EXPECT_COMPLEX_DOUBLE_EQ(fft(1ul, 2ul), std::complex<double>(-8.8817841970012523e-16, 0.0));
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
  EXPECT_COMPLEX_DOUBLE_EQ(fft(0ul, 0ul, 2ul), std::complex<double>(0, 0));
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

TEST(TestTensor, UpsampleEven1D) {
  Tensor<double> tensor({1.0, 2.0, 3.0, 4.0}, {4ul});
  const Tensor<double> upsampled_tensor = tensor.fft_upsample({6ul});
  EXPECT_EQ(upsampled_tensor.size(), 6u);
  EXPECT_THAT(upsampled_tensor.data(), ::testing::ElementsAre(1.0, 1.3839745962155614, 2.3839745962155616, 3.0,
                                                              4.1160254037844384, 3.1160254037844379));
}

TEST(TestTensor, UpsampleOdd1D) {
  Tensor<double> tensor({1.0, 2.0, 3.0, 4.0, 5.0}, {5ul});
  const Tensor<double> upsampled_tensor = tensor.fft_upsample({7ul});
  EXPECT_EQ(upsampled_tensor.size(), 7u);
  EXPECT_THAT(upsampled_tensor.data(),
              ::testing::ElementsAre(0.99999999999999978, 1.2061591336983095, 2.9225940224834375, 2.9343217797246823,
                                     3.620636352362689, 5.3243855812340284, 3.9919031304968517));
}

TEST(TestTensor, ToMatrixInvalid) {
  const Tensor t{std::vector{1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 11.0, 12.0},
                 std::vector{2ul, 2ul, 3ul}};
  EXPECT_THROW(
      (static_cast<Eigen::Map<const Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>>>(t)),
      LucidNotSupportedException);
}
