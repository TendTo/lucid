/**
 * @author Room 6.030
 * @copyright 2025 lucid
 * @licence BSD 3-Clause License
 */
#include <fmt/format.h>
#include <gtest/gtest.h>

#include "lucid/lucid.h"

using lucid::ConstantTruncatedFourierFeatureMap;
using lucid::GaussianKernel;
using lucid::GramMatrix;
using lucid::GridSearchTuner;
using lucid::Index;
using lucid::InverseGramMatrix;
using lucid::KernelRidgeRegressor;
using lucid::KFold;
using lucid::LeaveOneOut;
using lucid::LinearTruncatedFourierFeatureMap;
using lucid::LogTruncatedFourierFeatureMap;
using lucid::Matrix;
using lucid::MedianHeuristicTuner;
using lucid::MontecarloSimulation;
using lucid::MultiSet;
using lucid::Parameter;
using lucid::ParameterValue;
using lucid::ParameterValues;
using lucid::PolytopeSet;
using lucid::RectSet;
using lucid::Request;
using lucid::SphereSet;
using lucid::ValleePoussinKernel;
using lucid::Vector;

TEST(TestPrint, Request) {
  EXPECT_EQ(fmt::format("{}", Request::_), "Request( NoRequest )");
  EXPECT_EQ(fmt::format("{}", Request::OBJECTIVE_VALUE), "Request( ObjectiveValue )");
  EXPECT_EQ(fmt::format("{}", Request::GRADIENT), "Request( Gradient )");
}

TEST(TestPrint, Parameter) {
  EXPECT_EQ(fmt::format("{}", Parameter::_), "Parameter( NoParameter )");
  EXPECT_EQ(fmt::format("{}", Parameter::SIGMA_F), "Parameter( Sigma_f )");
  EXPECT_EQ(fmt::format("{}", Parameter::SIGMA_L), "Parameter( Sigma_l )");
  EXPECT_EQ(fmt::format("{}", Parameter::REGULARIZATION_CONSTANT), "Parameter( RegularizationConstant )");
  EXPECT_EQ(fmt::format("{}", Parameter::DEGREE), "Parameter( Degree )");
  EXPECT_EQ(fmt::format("{}", Parameter::GRADIENT_OPTIMIZABLE), "Parameter( GradientOptimizable )");
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
            "ParameterValues( Parameter( Sigma_l ) values( [1 2 3 4, 5 6 7 8] )");
  EXPECT_EQ(fmt::format("{}", ParameterValues(Parameter::DEGREE, 1, 2, 3)),
            "ParameterValues( Parameter( Degree ) values( [1, 2, 3] )");
}

TEST(TestPrint, RectSet) {
  EXPECT_EQ(fmt::format("{}", RectSet(Vector::Zero(2), Vector::Constant(2, 1.2))),
            "RectSet( lb( [0 0] ) ub( [1.2 1.2] ) )");
}

TEST(TestPrint, SphereSet) {
  EXPECT_EQ(fmt::format("{}", SphereSet(Vector::Zero(3), 1.5)), "SphereSet( center( [0 0 0] ) radius( 1.5 ) )");
}

TEST(TestPrint, PolytopeSet) {
  Matrix A(3, 2);
  Vector b(3);
  A << 1, 0,     // x <= 1
      -1, 0,     // x >= -1 (i.e., -x <= 1)
      0, 1;      // y <= 1
  b << 1, 1, 1;  // b vector: [1, 1, 1]
  EXPECT_EQ(fmt::format("{}", PolytopeSet(A, b)),
            "PolytopeSet( A(  1  0\n"
            "-1  0\n"
            " 0  1 ) b( 1 1 1 ) )");
}

TEST(TestPrint, MultiSet) {
  EXPECT_EQ(fmt::format("{}", MultiSet(RectSet(Vector::Zero(3), Vector::Constant(3, 1.5)),
                                       RectSet(Vector::Constant(3, 1.0), Vector::Constant(3, 2.1)),
                                       RectSet(Vector::Constant(3, -1.0), Vector::Constant(3, 0.2)))),
            "MultiSet( RectSet( lb( [0 0 0] ) ub( [1.5 1.5 1.5] ) ) "
            "RectSet( lb( [1 1 1] ) ub( [2.1 2.1 2.1] ) ) "
            "RectSet( lb( [-1 -1 -1] ) ub( [0.2 0.2 0.2] ) ) )");
}

TEST(TestPrint, LinearTruncatedFourierFeatureMap) {
  constexpr int num_frequencies = 3;
  Vector sigma_l{2};
  sigma_l << 3.2, 5.1;
  constexpr double sigma_f = 2.01;
  const RectSet X_bounds{Vector::Zero(2), Vector::Constant(2, 1.0)};
  const LinearTruncatedFourierFeatureMap feature_map{num_frequencies, sigma_l, sigma_f, X_bounds};
  EXPECT_EQ(fmt::format("{}", feature_map),
            "LinearTruncatedFourierFeatureMap( num_frequencies( 3 ) dimension( 17 ) "
            "weights(  0.451494  0.463899  0.463899  0.176708  0.176708  0.463899  0.463899  0.476646  "
            "0.476646  0.181563  0.181563  0.176708  0.176708  0.181563  0.181563 0.0691608 0.0691608 ) )");
}
TEST(TestPrint, LogTruncatedFourierFeatureMap) {
  constexpr int num_frequencies = 3;
  Vector sigma_l{2};
  sigma_l << 3.2, 5.1;
  constexpr double sigma_f = 2.01;
  const RectSet X_bounds{Vector::Zero(2), Vector::Constant(2, 1.0)};
  const LogTruncatedFourierFeatureMap feature_map{num_frequencies, sigma_l, sigma_f, X_bounds};
  EXPECT_EQ(fmt::format("{}", feature_map),
            "LogTruncatedFourierFeatureMap( num_frequencies( 3 ) dimension( 17 ) "
            "weights(  0.293604  0.327088  0.327088  0.149072  0.149072  0.306567  0.306567   0.34153   0.34153  "
            "0.155654  0.155654  0.133693  0.133693   0.14894   0.14894 0.0678805 0.0678805 ) )");
}
TEST(TestPrint, ConstantTruncatedFourierFeatureMap) {
  constexpr int num_frequencies = 3;
  Vector sigma_l{2};
  sigma_l << 3.2, 5.1;
  constexpr double sigma_f = 2.01;
  const RectSet X_bounds{Vector::Zero(2), Vector::Constant(2, 1.0)};
  const ConstantTruncatedFourierFeatureMap feature_map{num_frequencies, sigma_l, sigma_f, X_bounds};
  EXPECT_EQ(fmt::format("{}", feature_map),
            "ConstantTruncatedFourierFeatureMap( num_frequencies( 3 ) dimension( 17 ) "
            "weights( 1 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 ) )");  // TODO(tend): this does not seem right
}

TEST(TestPrint, GaussianKernelIsotropic) {
  const GaussianKernel kernel{5.2, 2.01};
  EXPECT_EQ(fmt::format("{}", kernel), "GaussianKernel( sigma_l( 5.2 ) sigma_f( 2.01 ) isotropic( 1 ) )");
}
TEST(TestPrint, GaussianKernelAnisotropic) {
  Vector sigma_l{4};
  sigma_l << 3.2, 5.1, 3.4, 1.24;
  const GaussianKernel kernel{sigma_l, 2.01};
  EXPECT_EQ(fmt::format("{}", kernel),
            "GaussianKernel( sigma_l(  3.2  5.1  3.4 1.24 ) sigma_f( 2.01 ) isotropic( 0 ) )");
}

TEST(TestPrint, Valle) {
  const ValleePoussinKernel kernel{10, 2.5};
  EXPECT_EQ(fmt::format("{}", kernel), "ValleePoussinKernel( a( 10 ) b( 2.5 ) )");
}

TEST(TestPrint, GramMatrix) {
  Vector sigma_l{2};
  sigma_l << 3.2, 5.1;
  const GaussianKernel kernel{sigma_l, 2.01};
  const GramMatrix gram_matrix{
      kernel, Matrix::NullaryExpr(2, 2, [](const Index row, const Index col) { return row + col + 1; })};
  EXPECT_EQ(fmt::format("{}", gram_matrix),
            "GramMatrix(\n"
            " 4.0401 3.77431\n"
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
            " 4.0401 3.77431\n"
            "3.77431  4.0401"
            "\n)^-1");
}

TEST(TestPrint, KernelRidgeRegressorIsotropic) {
  KernelRidgeRegressor regressor{std::make_unique<GaussianKernel>(3.2, 5.1), 1e-6};
  EXPECT_EQ(fmt::format("{}", regressor),
            "KernelRidgeRegressor( kernel( GaussianKernel( sigma_l( 3.2 ) sigma_f( 5.1 ) isotropic( 1 ) ) ) "
            "regularization_constant( 1e-06 ) )");
}

TEST(TestPrint, KernelRidgeRegressorAnisotropic) {
  KernelRidgeRegressor regressor{std::make_unique<GaussianKernel>(Vector::Constant(3, 1.1), 5.1), 1e-6};
  EXPECT_EQ(fmt::format("{}", regressor),
            "KernelRidgeRegressor( kernel( GaussianKernel( sigma_l( 1.1 1.1 1.1 ) sigma_f( 5.1 ) isotropic( 0 ) ) ) "
            "regularization_constant( 1e-06 ) )");
}

TEST(TestPrint, MedianHeuristicTuner) {
  EXPECT_EQ(fmt::format("{}", MedianHeuristicTuner()), "MedianHeuristicTuner( )");
}

TEST(TestPrint, GridSearchTuner) {
  Vector vec1{4}, vec2{4};
  vec1 << 1, 2, 3, 4;
  vec2 << 5, 6, 7, 8;
  EXPECT_EQ(fmt::format("{}", GridSearchTuner({ParameterValues(Parameter::REGULARIZATION_CONSTANT, 1e-10),
                                               ParameterValues(Parameter::SIGMA_L, vec1, vec2),
                                               ParameterValues(Parameter::DEGREE, 1, 2, 3)},
                                              4)),
            "GridSearchTuner( parameters( ["
            "ParameterValues( Parameter( RegularizationConstant ) values( [1e-10] ), "
            "ParameterValues( Parameter( Sigma_l ) values( [1 2 3 4, 5 6 7 8] ), "
            "ParameterValues( Parameter( Degree ) values( [1, 2, 3] )"
            "] ) n_jobs( 4 )");
}

TEST(TestPrint, MontecarloSimulation) {
  EXPECT_EQ(fmt::format("{}", MontecarloSimulation()), "MontecarloSimulation( )");
}

TEST(TestPrint, KFold) { EXPECT_EQ(fmt::format("{}", KFold(5, true)), "KFold( num_folds( 5 ), shuffle( true ) )"); }

TEST(TestPrint, LeaveOneOut) { EXPECT_EQ(fmt::format("{}", LeaveOneOut()), "LeaveOneOut( )"); }
