/**
 * @author Room 6.030
 * @copyright 2025 lucid
 * @licence BSD 3-Clause License
 */
#include <gtest/gtest.h>

#include "lucid/util/exception.h"
#include "lucid/verification/GurobiOptimiser.h"

using lucid::GurobiOptimiser;
using lucid::Index;

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
  EXPECT_THROW(GurobiOptimiser(T, gamma_, epsilon, b_norm, b_kappa, sigma_f, C_coeff, invalid_problem_log_file,
                                     iis_log_file),
               lucid::exception::LucidInvalidArgumentException);
}

TEST(IndexIterator, ConstructorInvalidIisFile) {
  const std::string invalid_iis_log_file{"invalid_iis.txt"};
  EXPECT_THROW(GurobiOptimiser(T, gamma_, epsilon, b_norm, b_kappa, sigma_f, C_coeff, problem_log_file,
                                     invalid_iis_log_file),
               lucid::exception::LucidInvalidArgumentException);
}
