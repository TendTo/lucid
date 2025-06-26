/**
 * @author c3054737
 * @copyright 2025 lucid
 * @licence BSD 3-Clause License
 * @file
 */
#include "lucid/util/random.h"

namespace lucid::random {

// NOLINTNEXTLINE(whitespace/braces)
std::mt19937 gen{std::random_device{}()};  ///< Global random number generator instance.

/**
 * Seed the random number generator.
 * If the seed is negative, a random seed will be generated using `std::random_device`.
 * @param seed Seed value. If negative, a random seed will be generated.
 */
void seed(const int seed) { gen.seed(seed < 0 ? std::random_device{}() : seed); }  // NOLINT(whitespace/braces)

}  // namespace lucid::random
