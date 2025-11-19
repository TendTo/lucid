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

TEST(TestGurobiOptimiser, Constructor) {
  const GurobiOptimiser o{};
  EXPECT_EQ(o.problem_log_file(), "");
  EXPECT_EQ(o.iis_log_file(), "");
}

TEST(TestGurobiOptimiser, ConstructorFiles) {
  const GurobiOptimiser o{problem_log_file, iis_log_file};
  EXPECT_EQ(o.problem_log_file(), problem_log_file);
  EXPECT_EQ(o.iis_log_file(), iis_log_file);
}

TEST(TestGurobiOptimiser, ConstructorInvalidProblemFile) {
  const std::string invalid_problem_log_file{"invalid_problem.txt"};
  EXPECT_THROW(GurobiOptimiser(invalid_problem_log_file, iis_log_file),
               lucid::exception::LucidInvalidArgumentException);
}

TEST(TestGurobiOptimiser, ConstructorInvalidIisFile) {
  const std::string invalid_iis_log_file{"invalid_iis.txt"};
  EXPECT_THROW(GurobiOptimiser(problem_log_file, invalid_iis_log_file),
               lucid::exception::LucidInvalidArgumentException);
}

#ifdef LUCID_GUROBI_BUILD
TEST(TestGurobiOptimiser, Solve) {
  const GurobiOptimiser o{};
  const Matrix fxn_lattice{Matrix::Random(10, 10)};
  const Matrix dn_lattice{Matrix::Random(10, 10)};
  const std::vector<Index> masks;
  const GurobiOptimiser::SolutionCallback cb = [](bool, double, const Eigen::Matrix<double, 1, -1>&, double, double,
                                                  double) {};

  try {
    o.solve_fourier_barrier_synthesis(
        {
            .num_constraints = 0,
            .fxn_lattice = fxn_lattice,
            .dn_lattice = dn_lattice,
            .x_include_mask = masks,
            .x_exclude_mask = masks,
            .x0_include_mask = masks,
            .x0_exclude_mask = masks,
            .xu_include_mask = masks,
            .xu_exclude_mask = masks,
            .T = T,
            .gamma = gamma_,
            .eta_coeff = 0.0,
            .min_x0_coeff = 0.0,
            .diff_sx0_coeff = 0.0,
            .gamma_coeff = 0.0,
            .max_xu_coeff = 0.0,
            .diff_sxu_coeff = 0.0,
            .ebk = epsilon * b_norm * b_kappa,
            .c_ebk_coeff = 0.0,
            .min_d_coeff = 0.0,
            .diff_d_sx_coeff = 0.0,
            .max_x_coeff = 0.0,
            .diff_sx_coeff = 0.0,
        },
        cb);
    FAIL();
  } catch (const GRBException& e) {
    EXPECT_EQ(e.getErrorCode(), GRB_ERROR_NO_LICENSE);
  }
}
#endif
