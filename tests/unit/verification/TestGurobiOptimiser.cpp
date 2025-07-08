/**
 * @author Room 6.030
 * @copyright 2025 lucid
 * @licence BSD 3-Clause License
 */
#include <gtest/gtest.h>

#include "lucid/util/exception.h"
#include "lucid/verification/GurobiOptimiser.h"
#ifdef LUCID_GUROBI_BUILD
#include "lucid/lib/gurobi.h"
#endif

using lucid::Dimension;
using lucid::GurobiOptimiser;
using lucid::Index;
using lucid::Matrix;

constexpr int T = 10;
constexpr double gamma_ = 0.1;
constexpr double epsilon = 0.01;
constexpr double b_norm = 0.5;
constexpr double b_kappa = 0.3;
constexpr double sigma_f = 0.3;
constexpr double C_coeff = 1.0;
const std::string problem_log_file{"problem.lp"};
const std::string iis_log_file{"iis.ilp"};

TEST(IndexIterator, Constructor) {
  const GurobiOptimiser o(T, gamma_, epsilon, b_norm, b_kappa, sigma_f);
  EXPECT_EQ(o.T(), T);
  EXPECT_EQ(o.gamma(), gamma_);
  EXPECT_EQ(o.epsilon(), epsilon);
  EXPECT_EQ(o.b_norm(), b_norm);
  EXPECT_EQ(o.b_kappa(), b_kappa);
  EXPECT_EQ(o.sigma_f(), sigma_f);
  EXPECT_EQ(o.problem_log_file(), "");
  EXPECT_EQ(o.iis_log_file(), "");
}

TEST(IndexIterator, ConstructorFiles) {
  const GurobiOptimiser o(T, gamma_, epsilon, b_norm, b_kappa, sigma_f, C_coeff, problem_log_file, iis_log_file);
  EXPECT_EQ(o.T(), T);
  EXPECT_EQ(o.gamma(), gamma_);
  EXPECT_EQ(o.epsilon(), epsilon);
  EXPECT_EQ(o.b_norm(), b_norm);
  EXPECT_EQ(o.b_kappa(), b_kappa);
  EXPECT_EQ(o.sigma_f(), sigma_f);
  EXPECT_EQ(o.problem_log_file(), problem_log_file);
  EXPECT_EQ(o.iis_log_file(), iis_log_file);
}

TEST(IndexIterator, ConstructorInvalidProblemFile) {
  const std::string invalid_problem_log_file{"invalid_problem.txt"};
  EXPECT_THROW(
      GurobiOptimiser(T, gamma_, epsilon, b_norm, b_kappa, sigma_f, C_coeff, invalid_problem_log_file, iis_log_file),
      lucid::exception::LucidInvalidArgumentException);
}

TEST(IndexIterator, ConstructorInvalidIisFile) {
  const std::string invalid_iis_log_file{"invalid_iis.txt"};
  EXPECT_THROW(
      GurobiOptimiser(T, gamma_, epsilon, b_norm, b_kappa, sigma_f, C_coeff, problem_log_file, invalid_iis_log_file),
      lucid::exception::LucidInvalidArgumentException);
}

#ifdef LUCID_GUROBI_BUILD
TEST(IndexIterator, Solve) {
  const GurobiOptimiser o(T, gamma_, epsilon, b_norm, b_kappa, sigma_f, C_coeff);
  const Matrix f0_lattice{Matrix::Random(10, 10)};
  const Matrix fu_lattice{Matrix::Random(10, 10)};
  const Matrix phi_mat{Matrix::Random(10, 10)};
  const Matrix w_mat{Matrix::Random(10, 10)};
  const GurobiOptimiser::SolutionCallback cb = [](bool, double, const Eigen::Matrix<double, 1, -1>&, double, double,
                                                  double) {};
  constexpr Dimension num_frequency_samples_per_dim = 5;
  constexpr Dimension num_frequencies_per_dim = 2;
  constexpr Dimension rkhs_dim = 10;
  constexpr Dimension original_dim = 10;

  try {
    o.solve(f0_lattice, fu_lattice, phi_mat, w_mat, rkhs_dim, num_frequencies_per_dim, num_frequency_samples_per_dim,
            original_dim, cb);
    FAIL();
  } catch (const GRBException& e) {
    EXPECT_EQ(e.getErrorCode(), GRB_ERROR_NO_LICENSE);
  }
}
#endif
