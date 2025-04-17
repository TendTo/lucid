/**
 * @author Room 6.030
 * @copyright 2025 lucid
 * @licence BSD 3-Clause License
 * @file
 */
#include "lucid/tuning/MedianHeuristicOptimiser.h"

#include <memory>

#include "lucid/lib/eigen.h"
#include "lucid/util/logging.h"

namespace lucid::tuning {

namespace {

/**
 * Compute the median of a vector.
 * If the vector has an even number of elements, the median is the lower of the two middle elements.
 * @param d vector
 * @return median of the vector
 * @see https://stackoverflow.com/a/62698308/15153171
 */
Scalar median(Vector& d) {
  auto r{d.reshaped()};
  std::ranges::sort(r);
  return (r.size() & 1) == 0 ? r.segment((r.size() - 2) / 2, 2).minCoeff() : r(r.size() / 2);
}

}  // namespace

MedianHeuristicOptimiser::MedianHeuristicOptimiser(const Sampler& sampler, const Dimension num_samples)
    : Optimiser{sampler, num_samples} {}

std::unique_ptr<Kernel> MedianHeuristicOptimiser::optimise_impl(const Kernel& kernel) const {
  const Matrix samples = sampler_();
  Vector dist = pdist(samples);
  median(dist);
  return kernel.clone();
}

}  // namespace lucid::tuning
