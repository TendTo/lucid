/**
 * @author c3054737
 * @copyright 2025 lucid
 * @licence BSD 3-Clause License
 * @file
 * PsoParameters class.
 */
#pragma once

#include <iosfwd>

namespace lucid {

/** Parameters for the Particle Swarm Optimisation (PSO) algorithm. */
struct PsoParameters {
  int num_particles = 40;   ///< Number of particles in the swarm
  double phi_local = 0.5;   ///< Cognitive coefficient
  double phi_global = 0.3;  ///< Social coefficient
  double weight = 0.9;      ///< Inertia weight
  int max_iter = 150;       ///< Maximum number of iterations. 0 means no limit
  double max_vel = 0.0;     ///< Maximum velocity for each particle. 0 means no limit
  double ftol = 1e-8;       ///< Function value tolerance for convergence
  double xtol = 1e-8;       ///< Position change tolerance for convergence
  int threads = 0;          ///< Number of threads to use. 0 means automatic detection
};

std::ostream& operator<<(std::ostream& os, const PsoParameters& params);

}  // namespace lucid

#ifdef LUCID_INCLUDE_FMT

#include "lucid/util/logging.h"

OSTREAM_FORMATTER(lucid::PsoParameters)

#endif  // LUCID_INCLUDE_FMT
