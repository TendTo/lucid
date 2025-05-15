/**
 * @author Room 6.030
 * @copyright 2025 lucid
 * @licence BSD 3-Clause License
 */
#include <gtest/gtest.h>

#include "lucid/math/math.h"
#include "lucid/tuning/MedianHeuristicTuner.h"
#include "lucid/util/exception.h"

using lucid::GaussianKernel;
using lucid::Index;
using lucid::Kernel;
using lucid::Vector;
using lucid::exception::LucidInvalidArgumentException;
using lucid::tuning::MedianHeuristicTuner;

TEST(TestMedianHeuristicTuner, Constructor) {
  Vector sigma_l{3};
  sigma_l << 1, 2, 3;
  const MedianHeuristicTuner o{GaussianKernel{sigma_l}};
  // FAIL();
}
