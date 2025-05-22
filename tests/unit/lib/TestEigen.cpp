/**
 * @author Room 6.030
 * @copyright 2025 lucid
 * @licence BSD 3-Clause License
 */
#include <gtest/gtest.h>

#include "lucid/lib/eigen.h"

using lucid::circulant;
using lucid::combvec;
using lucid::diff;
using lucid::fft2;
using lucid::fftn;
using lucid::fftshift;
using lucid::ifft2;
using lucid::ifftn;
using lucid::ifftshift;
using lucid::Matrix;
using lucid::median;
using lucid::mvnrnd;
using lucid::pad;
using lucid::pdist;
using lucid::peaks;
using lucid::Vector;

TEST(TestEigen, Diff) {
  Vector x{8};
  x << 1, 1, 2, 3, 5, 8, 13, 21;
  const Vector y = diff(x);
  std::cout << y << std::endl;
  Vector expected{7};
  expected << 0, 1, 1, 2, 3, 5, 8;
  EXPECT_EQ(y, expected);
}

TEST(TestEigen, DiffN) {
  Matrix x{3, 3};
  x << 1, 1, 1, 5, 5, 5, 25, 25, 25;
  const Matrix y = diff(x);
  Matrix expected{2, 3};
  expected << 4, 4, 4, 20, 20, 20;
  EXPECT_EQ(y, expected);
}

TEST(TestEigen, DiffMatrixRow) {
  Vector x{7};
  x << 0, 5, 15, 30, 50, 75, 105;
  const Vector y = diff(x, 2);
  Vector expected{5};
  expected << 5, 5, 5, 5, 5;
  EXPECT_EQ(y, expected);
}

TEST(TestEigen, DiffMatrixCol) {
  Matrix x{3, 3};
  x << 1, 3, 5, 7, 11, 13, 17, 19, 23;
  const Matrix y = diff(x, 1, false);
  Matrix expected{3, 2};
  expected << 2, 2, 4, 2, 2, 4;
  EXPECT_EQ(y, expected);
}

TEST(TestEigen, DiffCombvec) {
  Matrix m1{2, 3};
  m1 << 1, 2, 3, 4, 5, 6;
  Matrix m2{2, 2};
  m2 << 7, 8, 9, 10;
  const Matrix m3 = combvec(m1, m2);
  Matrix expected{4, 6};
  expected << 1, 2, 3, 1, 2, 3,  //
      4, 5, 6, 4, 5, 6,          //
      7, 7, 7, 8, 8, 8,          //
      9, 9, 9, 10, 10, 10;
  EXPECT_EQ(m3, expected);
}

TEST(TestEigen, DiffCombvecColumnVector) {
  Matrix m1{2, 3};
  m1 << 1, 2, 3, 4, 5, 6;
  Matrix m2{2, 2};
  m2 << 7, 8, 9, 10;
  Matrix m3{4, 1};
  m3 << 11, 12, 13, 14;
  const Matrix m4 = combvec(m1, m2, m3);
  Matrix expected{8, 6};
  expected << 1, 2, 3, 1, 2, 3,  //
      4, 5, 6, 4, 5, 6,          //
      7, 7, 7, 8, 8, 8,          //
      9, 9, 9, 10, 10, 10,       //
      11, 11, 11, 11, 11, 11,    //
      12, 12, 12, 12, 12, 12,    //
      13, 13, 13, 13, 13, 13,    //
      14, 14, 14, 14, 14, 14;
  EXPECT_EQ(m4, expected);
}

TEST(TestEigen, DiffCombvecDifferentSizes) {
  Matrix m1{2, 3};
  m1 << 1, 2, 3, 4, 5, 6;
  Matrix m2{2, 2};
  m2 << 7, 8, 9, 10;
  Matrix m3{4, 2};
  m3 << 11, 12, 13, 14, 15, 16, 17, 18;
  const Matrix m4 = combvec(m1, m2, m3);
  Matrix expected{8, 12};
  expected << 1, 2, 3, 1, 2, 3, 1, 2, 3, 1, 2, 3,      //
      4, 5, 6, 4, 5, 6, 4, 5, 6, 4, 5, 6,              //
      7, 7, 7, 8, 8, 8, 7, 7, 7, 8, 8, 8,              //
      9, 9, 9, 10, 10, 10, 9, 9, 9, 10, 10, 10,        //
      11, 11, 11, 11, 11, 11, 12, 12, 12, 12, 12, 12,  //
      13, 13, 13, 13, 13, 13, 14, 14, 14, 14, 14, 14,  //
      15, 15, 15, 15, 15, 15, 16, 16, 16, 16, 16, 16,  //
      17, 17, 17, 17, 17, 17, 18, 18, 18, 18, 18, 18;
  EXPECT_EQ(m4, expected);
}

TEST(TestEigen, Pdist) {
  Matrix m1{4, 3};
  m1 << 1, 2, 3,   //
      4, 5, 6,     //
      10, 10, 10,  //
      5, 6, 4;
  const Vector dist = pdist(m1);
  Vector expected{6};
  expected << 5.196152422706632, 13.928388277184119, 8.7749643873921226, 5.7445626465380286, 2.4494897427831779,
      8.7749643873921226;
  EXPECT_EQ(dist, expected);
}

TEST(TestEigen, Pdist3P) {
  Matrix m1{4, 3};
  m1 << 1, 2, 3, 4, 5, 6, 10, 10, 10, 5, 6, 4;
  const Vector dist = pdist<3>(m1);
  Vector expected{6};
  expected << 4.3267487109222245, 11.656953366502911, 7.3986362229914091, 5.05277434720856, 2.1544346900318838,
      7.3986362229914091;
  EXPECT_EQ(dist, expected);
}

TEST(TestEigen, FftshiftEven) {
  Vector x{6};
  x << 1, 2, 3, 4, 5, 6;
  const Vector y = fftshift(x);
  Vector expected{6};
  expected << 4, 5, 6, 1, 2, 3;
  EXPECT_EQ(y, expected);
}

TEST(TestEigen, FftshiftEvenMatrix) {
  Matrix x{4, 4};
  x << 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16;
  const Matrix y = fftshift(x);
  Matrix expected{4, 4};
  expected << 11, 12, 9, 10, 15, 16, 13, 14, 3, 4, 1, 2, 7, 8, 5, 6;
  EXPECT_EQ(y, expected);
}

TEST(TestEigen, FftshiftOddMatrix) {
  Matrix x{3, 3};
  x << 1, 2, 3, 4, 5, 6, 7, 8, 9;
  const Matrix y = fftshift(x);
  Matrix expected{3, 3};
  expected << 9, 7, 8, 3, 1, 2, 6, 4, 5;
  EXPECT_EQ(y, expected);
}

TEST(TestEigen, FftshiftOdd) {
  Vector x{7};
  x << 1, 2, 3, 4, 5, 6, 7;
  const Vector y = fftshift(x);
  Vector expected{7};
  expected << 5, 6, 7, 1, 2, 3, 4;
  EXPECT_EQ(y, expected);
}

TEST(TestEigen, IFftshiftEven) {
  Vector x{6};
  x << 4, 5, 6, 1, 2, 3;
  const Vector y = ifftshift(x);
  Vector expected{6};
  expected << 1, 2, 3, 4, 5, 6;
  EXPECT_EQ(y, expected);
}

TEST(TestEigen, IFftshiftOdd) {
  Vector x{7};
  x << 5, 6, 7, 1, 2, 3, 4;
  const Vector y = ifftshift(x);
  Vector expected{7};
  expected << 1, 2, 3, 4, 5, 6, 7;
  EXPECT_EQ(y, expected);
}

TEST(TestEigen, IFftshiftEvenMatrix) {
  Matrix x{4, 4};
  x << 11, 12, 9, 10, 15, 16, 13, 14, 3, 4, 1, 2, 7, 8, 5, 6;
  const Matrix y = ifftshift(x);
  Matrix expected{4, 4};
  expected << 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16;
  EXPECT_EQ(y, expected);
}

TEST(TestEigen, IFftshiftOddMatrix) {
  Matrix x{3, 3};
  x << 9, 7, 8, 3, 1, 2, 6, 4, 5;
  const Matrix y = ifftshift(x);
  Matrix expected{3, 3};
  expected << 1, 2, 3, 4, 5, 6, 7, 8, 9;
  EXPECT_EQ(y, expected);
}

TEST(TestEigen, Circulant) {
  Vector x{4};
  x << 1, 2, 4, 8;
  const Matrix y = circulant(x);
  Matrix expected{4, 4};
  expected << 1, 8, 4, 2, 2, 1, 8, 4, 4, 2, 1, 8, 8, 4, 2, 1;
  EXPECT_EQ(y, expected);
}

TEST(TestEigen, PadSingleValueVector) {
  Vector x{3};
  x << 1, 2, 3;
  const Matrix y = pad(x, 2, 0);
  Matrix expected{7, 5};
  expected << 0, 0, 0, 0, 0,  //
      0, 0, 0, 0, 0,          //
      0, 0, 1, 0, 0,          //
      0, 0, 2, 0, 0,          //
      0, 0, 3, 0, 0,          //
      0, 0, 0, 0, 0,          //
      0, 0, 0, 0, 0;
  EXPECT_EQ(y, expected);
}

TEST(TestEigen, PadSingleValueMatrix) {
  Matrix x{3, 2};
  x << 1, 2, 3, 4, 5, 6;
  const Matrix y = pad(x, 2, 0);
  Matrix expected{7, 6};
  expected << 0, 0, 0, 0, 0, 0,  //
      0, 0, 0, 0, 0, 0,          //
      0, 0, 1, 2, 0, 0,          //
      0, 0, 3, 4, 0, 0,          //
      0, 0, 5, 6, 0, 0,          //
      0, 0, 0, 0, 0, 0,          //
      0, 0, 0, 0, 0, 0;
  EXPECT_EQ(y, expected);
}

TEST(TestEigen, PadSingleValueMatrixDifferentValue) {
  Matrix x{3, 2};
  x << 1, 2, 3, 4, 5, 6;
  const Matrix y = pad(x, 2, 9.0);
  Matrix expected{7, 6};
  expected << 9, 9, 9, 9, 9, 9,  //
      9, 9, 9, 9, 9, 9,          //
      9, 9, 1, 2, 9, 9,          //
      9, 9, 3, 4, 9, 9,          //
      9, 9, 5, 6, 9, 9,          //
      9, 9, 9, 9, 9, 9,          //
      9, 9, 9, 9, 9, 9;
  EXPECT_EQ(y, expected);
}

TEST(TestEigen, PadSingleValueMatrixDifferentValueRowCol) {
  Matrix x{3, 2};
  x << 1, 2, 3, 4, 5, 6;
  const Matrix y = pad(x, 2, 3, 9.0);
  Matrix expected{7, 8};
  expected << 9, 9, 9, 9, 9, 9, 9, 9,  //
      9, 9, 9, 9, 9, 9, 9, 9,          //
      9, 9, 9, 1, 2, 9, 9, 9,          //
      9, 9, 9, 3, 4, 9, 9, 9,          //
      9, 9, 9, 5, 6, 9, 9, 9,          //
      9, 9, 9, 9, 9, 9, 9, 9,          //
      9, 9, 9, 9, 9, 9, 9, 9;
  EXPECT_EQ(y, expected);
}

TEST(TestEigen, PadSingleValueMatrixDifferentValueTopBottomLeftRight) {
  Matrix x{3, 2};
  x << 1, 2, 3, 4, 5, 6;
  const Matrix y = pad(x, 1, 2, 3, 4, 9.0);
  Matrix expected{6, 9};
  expected << 9, 9, 9, 9, 9, 9, 9, 9, 9,  //
      9, 9, 9, 1, 2, 9, 9, 9, 9,          //
      9, 9, 9, 3, 4, 9, 9, 9, 9,          //
      9, 9, 9, 5, 6, 9, 9, 9, 9,          //
      9, 9, 9, 9, 9, 9, 9, 9, 9,          //
      9, 9, 9, 9, 9, 9, 9, 9, 9;          //

  EXPECT_EQ(y, expected);
}

TEST(TestEigen, Fft2) {
  Matrix x{3, 3};
  x << 1, 2, 3, 4, 5, 6, 7, 8, 9;
  const Eigen::MatrixXcd y = fft2(x);
  Eigen::MatrixXcd expected{3, 3};
  expected << std::complex<double>{45, 0}, std::complex<double>{-4.5, 2.59807621135332},
      std::complex<double>{-4.5, -2.59807621135332}, std::complex<double>{-13.5, 7.79422863405995},
      std::complex<double>{0, 0}, std::complex<double>{0, 0}, std::complex<double>{-13.5, -7.79422863405995},
      std::complex<double>{0, 0}, std::complex<double>{0, 0};
  EXPECT_TRUE(y.isApprox(expected));
}

TEST(TestEigen, Fftn) {
  Matrix x{3, 3};
  x << 1, 2, 3, 4, 5, 6, 7, 8, 9;
  const Eigen::MatrixXcd y = fft2(x);
  Eigen::MatrixXcd expected{3, 3};
  expected << std::complex<double>{45, 0}, std::complex<double>{-4.5, 2.59807621135332},
      std::complex<double>{-4.5, -2.59807621135332}, std::complex<double>{-13.5, 7.79422863405995},
      std::complex<double>{0, 0}, std::complex<double>{0, 0}, std::complex<double>{-13.5, -7.79422863405995},
      std::complex<double>{0, 0}, std::complex<double>{0, 0};
  EXPECT_TRUE(y.isApprox(expected));
}

TEST(TestEigen, IFft2) {
  Matrix x{3, 3};
  x << 1, 2, 3, 4, 5, 6, 7, 8, 9;
  const Matrix y = ifft2(fft2(x));
  EXPECT_EQ(y, x);
}

TEST(TestEigen, IFftn) {
  Matrix x{3, 3};
  x << 1, 2, 3, 4, 5, 6, 7, 8, 9;
  const Matrix y = ifftn(fftn(x));
  EXPECT_TRUE(y.isApprox(x));
}

TEST(TestEigen, MedianOddNumberOfElements) {
  Vector v(5);
  v << 5, 1, 3, 4, 2;
  EXPECT_EQ(median(v), 3.0);
}

TEST(TestEigen, MedianEvenNumberOfElements) {
  Eigen::Vector<double, 10> v{6, 1, 3, 4, 2, 5, 9, -1, 10, -2};
  EXPECT_EQ(median(v), 3.5);
}

TEST(TestEigen, SortedVector) {
  Eigen::Vector<double, 5> v{1, 2, 3, 4, 5};
  EXPECT_EQ(median(v), 3.0);
}

TEST(TestEigen, MedianUnsortedVector) {
  Eigen::Vector<double, 5> v{5, 2, 1, 4, 3};
  EXPECT_EQ(median(v), 3.0);
}

TEST(TestEigen, MedianDuplicates) {
  Eigen::Vector<double, 7> v{1, 2, 2, 3, 3, 3, 4};
  EXPECT_EQ(median(v), 3.0);
}

TEST(TestEigen, MedianNegativeValues) {
  Eigen::Vector<double, 5> v{-5, -2, 0, 2, 5};
  EXPECT_EQ(median(v), 0.0);
}

TEST(TestEigen, MedianSingleElement) {
  Eigen::Vector<double, 1> v{42};
  EXPECT_EQ(median(v), 42.0);
}

TEST(TestEigen, MedianMatrixOdd) {
  Eigen::Matrix<double, 3, 3> m;
  m << 1, 2, 3, 4, 5, 6, 7, 8, 9;
  EXPECT_EQ(median(m), 5.0);
}

TEST(TestEigen, MedianMatrixEven) {
  Eigen::Matrix<double, 2, 3> m;
  m << 4, 5, 6, 1, 2, 3;
  EXPECT_EQ(median(m), 3.5);
}

TEST(TestEigen, StaticAssertions) {
  static_assert(std::is_same_v<decltype(fftshift(Matrix{} + Matrix{}))::CoeffReturnType, double>,
                "Support expressions");
  static_assert(std::is_same_v<decltype(ifftshift(ifftshift(Matrix{})))::CoeffReturnType, double>,
                "Support nested application");
}
