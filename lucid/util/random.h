/**
 * @author c3054737
 * @copyright 2025 lucid
 * @licence BSD 3-Clause License
 * @file
 * Collection of random functions.
 */
#pragma once

#include <random>

namespace lucid::random {

extern std::mt19937 gen;

void seed(int seed);

}  // namespace lucid::random
