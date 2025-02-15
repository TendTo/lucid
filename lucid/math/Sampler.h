/**
 * @author Room 6.030
 * @copyright 2025 lucid
 * @licence BSD 3-Clause License
 * @file
 * Sampler functor.
 */
#pragma once

#include <functional>

#include "lucid/lib/eigen.h"

namespace lucid {
/**
 * Function or functor that samples a vector space @X to produce an arbitrary number of vectors.
 * @return matrix of samples, where each row is a sample
 */
using Sampler = std::function<Matrix()>;
}  // namespace lucid
