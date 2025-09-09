/**
 * @author Room 6.030
 * @copyright 2025 lucid
 * @licence BSD 3-Clause License
 * @file
 */
#include "lucid/model/FeatureMap.h"

#include "lucid/model/ConstantTruncatedFourierFeatureMap.h"
#include "lucid/model/LinearTruncatedFourierFeatureMap.h"
#include "lucid/model/LogTruncatedFourierFeatureMap.h"
#include "lucid/util/Stats.h"
#include "lucid/util/Timer.h"

namespace lucid {

Matrix FeatureMap::operator()(ConstMatrixRef x) const {
  TimerGuard tg{Stats::Scoped::top() ? &Stats::Scoped::top()->value().feature_map_timer : nullptr};
  if (Stats::Scoped::top()) Stats::Scoped::top()->value().num_feature_map_applications++;
  return apply_impl(x);
}
std::ostream& operator<<(std::ostream& os, const FeatureMap& f) {
  if (const auto* casted = dynamic_cast<const ConstantTruncatedFourierFeatureMap*>(&f)) return os << *casted;
  if (const auto* casted = dynamic_cast<const LinearTruncatedFourierFeatureMap*>(&f)) return os << *casted;
  if (const auto* casted = dynamic_cast<const LogTruncatedFourierFeatureMap*>(&f)) return os << *casted;
  return os << "FeatureMap( )";
}

}  // namespace lucid
