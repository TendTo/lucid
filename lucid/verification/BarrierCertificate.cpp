/**
 * @author c3054737
 * @copyright 2025 lucid
 * @licence BSD 3-Clause License
 * @file
 */
#include "lucid/verification/BarrierCertificate.h"

#include "lucid/util/Stats.h"
#include "lucid/util/Timer.h"
#include "lucid/util/error.h"
#include "lucid/verification/FourierBarrierCertificate.h"

namespace lucid {

BarrierCertificate::BarrierCertificate(const int T, const double gamma, const double eta, const double c)
    : T_{T}, gamma_{gamma}, eta_{eta}, c_{c}, norm_{0}, safety_{0} {
  LUCID_CHECK_ARGUMENT_CMP(T, >, 0);
  LUCID_CHECK_ARGUMENT_CMP(gamma, >, 0);
  LUCID_CHECK_ARGUMENT_CMP(eta, >=, 0);
  LUCID_CHECK_ARGUMENT_CMP(c, >=, 0);
}

double BarrierCertificate::operator()(ConstVectorRef x) const { return apply_impl(x); }

std::ostream& operator<<(std::ostream& os, const BarrierCertificate& barrier) {
  if (const auto* casted = dynamic_cast<const FourierBarrierCertificate*>(&barrier)) return os << *casted;
  return os << "BarrierCertificate( )";
}

}  // namespace lucid
