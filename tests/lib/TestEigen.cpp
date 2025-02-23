/**
 * @author Room 6.030
 * @copyright 2025 lucid
 * @licence BSD 3-Clause License
 */
#include <gtest/gtest.h>

#include "lucid/lib/eigen.h"

using lucid::Matrix;
using lucid::Vector;

TEST(TestEigen, Diff) {
  Vector x{8};
  x << 1, 1, 2, 3, 5, 8, 13, 21;
  const Vector y = lucid::diff(x);
  std::cout << y << std::endl;
  Vector expected{7};
  expected << 0, 1, 1, 2, 3, 5, 8;
  EXPECT_EQ(y, expected);
}

TEST(TestEigen, DiffN) {
  Matrix x{3, 3};
  x << 1, 1, 1, 5, 5, 5, 25, 25, 25;
  const Matrix y = lucid::diff(x);
  Matrix expected{2, 3};
  expected << 4, 4, 4, 20, 20, 20;
  EXPECT_EQ(y, expected);
}

TEST(TestEigen, DiffMatrixRow) {
  Vector x{7};
  x << 0, 5, 15, 30, 50, 75, 105;
  const Vector y = lucid::diff(x, 2);
  Vector expected{5};
  expected << 5, 5, 5, 5, 5;
  EXPECT_EQ(y, expected);
}

TEST(TestEigen, DiffMatrixCol) {
  Matrix x{3, 3};
  x << 1, 3, 5, 7, 11, 13, 17, 19, 23;
  const Matrix y = lucid::diff(x, 1, false);
  Matrix expected{3, 2};
  expected << 2, 2, 4, 2, 2, 4;
  EXPECT_EQ(y, expected);
}

TEST(TestEigen, DiffCombvec) {
  Matrix m1{2, 3};
  m1 << 1, 2, 3, 4, 5, 6;
  Matrix m2{2, 2};
  m2 << 7, 8, 9, 10;
  const Matrix m3 = lucid::combvec(m1, m2);
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
  const Matrix m4 = lucid::combvec(m1, m2, m3);
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
  const Matrix m4 = lucid::combvec(m1, m2, m3);
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
  const Vector dist = lucid::pdist(m1);
  Vector expected{6};
  expected << 5.196152422706632, 13.928388277184119, 8.7749643873921226, 5.7445626465380286, 2.4494897427831779,
      8.7749643873921226;
  EXPECT_EQ(dist, expected);
}

TEST(TestEigen, Pdist3P) {
  Matrix m1{4, 3};
  m1 << 1, 2, 3, 4, 5, 6, 10, 10, 10, 5, 6, 4;
  const Vector dist = lucid::pdist<3>(m1);
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

TEST(TestEigen, FftshiftOdd) {
  Vector x{7};
  x << 1, 2, 3, 4, 5, 6, 7;
  const Vector y = fftshift(x);
  Vector expected{7};
  expected << 5, 6, 7, 1, 2, 3, 4;
  EXPECT_EQ(y, expected);
}
