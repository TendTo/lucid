/**
 * @author Room 6.030
 * @copyright 2025 lucid
 * @licence BSD 3-Clause License
 */
#include <gtest/gtest.h>

#include "lucid/math/math.h"
#include "lucid/tuning/MedianHeuristicOptimiser.h"
#include "lucid/util/exception.h"

using lucid::GaussianKernel;
using lucid::Index;
using lucid::Kernel;
using lucid::Vector;
using lucid::exception::LucidInvalidArgumentException;
using lucid::tuning::MedianHeuristicOptimiser;

TEST(TestMedianHeuristicOptimiser, Constructor) {
  Vector sigma_l{3};
  sigma_l << 1, 2, 3;
  const MedianHeuristicOptimiser o{GaussianKernel{1.0, sigma_l}};
  // FAIL();
}
