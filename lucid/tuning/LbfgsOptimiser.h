/**
 * @author Room 6.030
 * @copyright 2025 lucid
 * @licence BSD 3-Clause License
 * @file
 * LbfgsOptimiser class.
 */
#pragma once

#include <memory>

#include "lucid/tuning/Optimiser.h"

namespace lucid::tuning {

class LbfgsOptimiser final : public Optimiser {
 public:
  [[nodiscard]] std::unique_ptr<Kernel> Optimise(const Kernel& kernel) const override;
};

}  // namespace lucid::tuning
