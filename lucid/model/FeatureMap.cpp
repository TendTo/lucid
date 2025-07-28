/**
 * @author lucid_authors
 * @copyright 2025 lucid
 * @licence BSD 3-Clause License
 * @file
 */
#include "lucid/model/FeatureMap.h"

#include "lucid/model/ConstantTruncatedFourierFeatureMap.h"
#include "lucid/model/LinearTruncatedFourierFeatureMap.h"
#include "lucid/model/LogTruncatedFourierFeatureMap.h"

namespace lucid {

std::ostream& operator<<(std::ostream& os, const FeatureMap& f) {
  if (const auto* casted = dynamic_cast<const ConstantTruncatedFourierFeatureMap*>(&f)) return os << *casted;
  if (const auto* casted = dynamic_cast<const LinearTruncatedFourierFeatureMap*>(&f)) return os << *casted;
  if (const auto* casted = dynamic_cast<const LogTruncatedFourierFeatureMap*>(&f)) return os << *casted;
  return os << "FeatureMap( )";
}

}  // namespace lucid
