/**
 * @author c3054737
 * @copyright 2025 lucid
 * @licence BSD 3-Clause License
 * @file
 * Optimiser class.
 */
#pragma once

#include <functional>

#include "lucid/lib/eigen.h"

namespace lucid {

class Optimiser {
 public:
  /**
   * Callback function called when the optimisation is done.
   * @param success true if the optimisation was successful, false if no solution was found
   * @param obj_val objective value. 0 if no solution was found
   * @param eta eta value. 0 if no solution was found
   * @param c c value. 0 if no solution was found
   * @param norm actual norm of the barrier function. 0 if no solution was found
   */
  using SolutionCallback = std::function<void(bool, double, const Vector&, double, double, double)>;
};

}  // namespace lucid
