/**
 * @author Room 6.030
 * @copyright 2025 lucid
 * @licence BSD 3-Clause License
 * @file
 */
#include "lucid/tuning/Optimiser.h"

#include <memory>

#include "lucid/math/GaussianKernel.h"
#include "lucid/tuning/LbfgsOptimiser.h"

namespace lucid::tuning {

Optimiser::Optimiser(const Kernel& estimator) : estimator_{estimator} {}

Vector Optimiser::optimise(const Matrix& x, const Matrix& y) const { return optimise_impl(x, y); }

// Scalar Optimiser::evaluate(const Kernel&) const { return Scalar(); }

}  // namespace lucid::tuning
