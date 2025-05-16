/**
 * @author Room 6.030
 * @copyright 2025 lucid
 * @licence BSD 3-Clause License
 */
#include <gtest/gtest.h>

#include "lucid/model/MedianHeuristicTuner.h"
#include "lucid/model/model.h"
#include "lucid/util/exception.h"

using lucid::GaussianKernel;
using lucid::Index;
using lucid::Kernel;
using lucid::Vector;
using lucid::exception::LucidInvalidArgumentException;
using lucid::MedianHeuristicTuner;

TEST(TestMedianHeuristicTuner, Constructor) {
  Vector sigma_l{3};
  sigma_l << 1, 2, 3;
  const MedianHeuristicTuner o{};
  // FAIL();
}
