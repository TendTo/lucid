/**
 * @author Room 6.030
 * @copyright 2025 lucid
 * @licence BSD 3-Clause License
 * @file
 * Optimiser class.
 */
#pragma once

#include <memory>

#include "lucid/math/Kernel.h"

namespace lucid::tuning {

class Optimiser {
 public:
  virtual ~Optimiser() = default;

  [[nodiscard]] virtual std::unique_ptr<Kernel> Optimise(const Kernel& kernel) const = 0;
};

}  // namespace lucid::tuning
