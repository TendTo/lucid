/**
 * @author c3054737
 * @copyright 2025 lucid
 * @licence BSD 3-Clause License
 * @file
 * Collection of random functions.
 */
#pragma once

#include <random>

/**
 * @namespace lucid::random
 * Collection of random functions.
 */
namespace lucid::random {

extern std::mt19937 gen;  ///< Global random number generator instance.

/**
 * Seed the random number generator.
 * This is used to ensure reproducibility between runs of the program.
 * If the seed is not set, the generator will use a `std::random_device` to generate a random seed.
 * @param seed value to use for the random number generator. If negative, a random seed will be generated.
 */
void seed(int seed);

}  // namespace lucid::random
