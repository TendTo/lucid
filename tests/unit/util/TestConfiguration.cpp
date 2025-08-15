/**
 * @author Room 6.030
 * @copyright 2025 lucid
 * @licence BSD 3-Clause License
 */
#include <gtest/gtest.h>

#include "lucid/util/Configuration.h"

using lucid::Configuration;

TEST(TestConfiguration, DefaultConstructor) {
  const Configuration config;
  EXPECT_EQ(config.verbose(), 3);
  EXPECT_EQ(config.seed(), -1);
  EXPECT_FALSE(config.plot());
  EXPECT_FALSE(config.verify());
  EXPECT_EQ(config.problem_log_file(), "");
  EXPECT_EQ(config.iis_log_file(), "");
  EXPECT_FALSE(config.system_dynamics());
  EXPECT_EQ(config.X_bounds().get(), nullptr);
  EXPECT_EQ(config.X_init().get(), nullptr);
  EXPECT_EQ(config.X_unsafe().get(), nullptr);
  EXPECT_EQ(config.x_samples().size(), 0);
  EXPECT_EQ(config.xp_samples().size(), 0);
  EXPECT_EQ(config.f_xp_samples().size(), 0);
  EXPECT_EQ(config.num_samples(), 1000);
  EXPECT_DOUBLE_EQ(config.noise_scale(), 0.01);
}