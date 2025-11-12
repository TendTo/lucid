/**
 * @author c3054737
 * @copyright 2025 lucid
 * @licence BSD 3-Clause License
 * @file
 * PsoParameters class.
 */
#include "lucid/verification/PsoParameters.h"

#include <ostream>

namespace lucid {

std::ostream& operator<<(std::ostream& os, const PsoParameters& params) {
  os << "PsoParameters( "
     << "num_particles( " << params.num_particles << " ) "
     << "phi_local( " << params.phi_local << " ) "
     << "phi_global( " << params.phi_global << " ) "
     << "weight( " << params.weight << " ) "
     << "max_iter( " << params.max_iter << " ) "
     << "max_vel( " << params.max_vel << " ) "
     << "ftol( " << params.ftol << " ) "
     << "xtol( " << params.xtol << " ) "
     << "threads( " << params.threads << " ) "
     << ")";
  return os;
}

}  // namespace lucid
