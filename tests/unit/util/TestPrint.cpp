/**
 * @author Room 6.030
 * @copyright 2025 lucid
 * @licence BSD 3-Clause License
 */
#include <fmt/format.h>
#include <gtest/gtest.h>

#include "lucid/lucid.h"

using lucid::GaussianKernel;
using lucid::GramMatrix;
using lucid::Index;
using lucid::InverseGramMatrix;
using lucid::KernelRidgeRegressor;
using lucid::Matrix;
using lucid::Parameter;
using lucid::ParameterValue;
using lucid::ParameterValues;
using lucid::Vector;

TEST(TestPrint, Parameter) {
  EXPECT_EQ(fmt::format("{}", Parameter::SIGMA_F), "Parameter( Sigma_f )");
  EXPECT_EQ(fmt::format("{}", Parameter::SIGMA_L), "Parameter( Sigma_l )");
  EXPECT_EQ(fmt::format("{}", Parameter::REGULARIZATION_CONSTANT), "Parameter( RegularizationConstant )");
}

TEST(TestPrint, ParameterValue) {
  Vector vec{4};
  vec << 1, 2, 3, 4;
  EXPECT_EQ(fmt::format("{}", ParameterValue(Parameter::REGULARIZATION_CONSTANT, 1e-10)),
            "ParameterValue( Parameter( RegularizationConstant ) value( 1e-10 )");
  EXPECT_EQ(fmt::format("{}", ParameterValue(Parameter::SIGMA_L, vec)),
            "ParameterValue( Parameter( Sigma_l ) value( 1 2 3 4 )");
  EXPECT_EQ(fmt::format("{}", ParameterValue(Parameter::DEGREE, 5)), "ParameterValue( Parameter( Degree ) value( 5 )");
}

TEST(TestPrint, ParameterValues) {
  Vector vec1{4}, vec2{4};
  vec1 << 1, 2, 3, 4;
  vec2 << 5, 6, 7, 8;
  EXPECT_EQ(fmt::format("{}", ParameterValues(Parameter::REGULARIZATION_CONSTANT, 1e-10)),
            "ParameterValues( Parameter( RegularizationConstant ) values( [1e-10] )");
  EXPECT_EQ(fmt::format("{}", ParameterValues(Parameter::SIGMA_L, vec1, vec2)),
            "ParameterValues( Parameter( Sigma_l ) values( [[1, 2, 3, 4], [5, 6, 7, 8]] )");
  EXPECT_EQ(fmt::format("{}", ParameterValues(Parameter::DEGREE, 1, 2, 3)),
            "ParameterValues( Parameter( Degree ) values( [1, 2, 3] )");
}

TEST(TestPrint, GaussianKernel) {
  Vector sigma_l{4};
  sigma_l << 3.2, 5.1, 3.4, 1.24;
  const GaussianKernel kernel{sigma_l, 2.01};
  EXPECT_EQ(fmt::format("{}", kernel), "GaussianKernel( sigma_l(  3.2  5.1  3.4 1.24 ) sigma_f( 2.01 ) )");
}

TEST(TestPrint, GramMatrix) {
  Vector sigma_l{2};
  sigma_l << 3.2, 5.1;
  const GaussianKernel kernel{sigma_l, 2.01};
  const GramMatrix gram_matrix{
      kernel, Matrix::NullaryExpr(2, 2, [](const Index row, const Index col) { return row + col + 1; })};
  EXPECT_EQ(fmt::format("{}", gram_matrix),
            "GramMatrix(\n"
            " 4.0401       0\n"
            "3.77431  4.0401"
            "\n)");
}

TEST(TestPrint, InverseGramMatrix) {
  Vector sigma_l{2};
  sigma_l << 3.2, 5.1;
  const GaussianKernel kernel{sigma_l, 2.01};
  const GramMatrix gram_matrix{
      kernel, Matrix::NullaryExpr(2, 2, [](const Index row, const Index col) { return row + col + 1; })};
  EXPECT_EQ(fmt::format("{}", gram_matrix.inverse()),
            "GramMatrix(\n"
            " 4.0401       0\n"
            "3.77431  4.0401"
            "\n)^-1");
}

TEST(TestPrint, KernelRidgeRegressor) {
  KernelRidgeRegressor regressor{std::make_unique<GaussianKernel>(2, 3.2, 5.1), 1e-6};
  EXPECT_EQ(fmt::format("{}", regressor),
            "KernelRidgeRegressor( kernel( GaussianKernel( sigma_l( 3.2 3.2 ) sigma_f( 5.1 ) ) ) "
            "regularization_constant( 1e-06 ) )");
}