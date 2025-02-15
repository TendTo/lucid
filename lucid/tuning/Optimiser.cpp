/**
 * @author Room 6.030
 * @copyright 2025 lucid
 * @licence BSD 3-Clause License
 * @file
 */
#include "lucid/tuning/Optimiser.h"

#include "lucid/math/GaussianKernel.h"
#include "lucid/tuning/LbfgsOptimiser.h"

namespace lucid::tuning {

Optimiser::Optimiser(const Sampler& sampler, const Dimension num_samples)
    : num_samples_{num_samples}, sampler_{sampler} {}

std::unique_ptr<Kernel> Optimiser::optimise(const Kernel& kernel) const { return optimise_impl(kernel); }

Scalar Optimiser::evaluate(const Kernel&) const { return Scalar(); }

}  // namespace lucid::tuning
