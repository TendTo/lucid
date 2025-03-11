/**
 * @author Room 6.030
 * @copyright 2025 lucid
 * @licence BSD 3-Clause License
 * @file
 * MultiSet class.
 */
#pragma once

#include <iosfwd>
#include <memory>
#include <random>
#include <string>

#include "lucid/math/Set.h"

namespace lucid {

/**
 * Set composed of the union of multiple sets.
 * Checking whether a vector is in the set is equivalent to checking if it is in any of the sets.
 * @todo Simplistic implementation of sampling and dimension.
 */
class MultiSet final : public Set {
 public:
  template <class... S>
    requires(std::derived_from<S, Set> && ...)
  explicit MultiSet(S&&... sets) : sets_{}, dis_{0, sizeof...(S)} {
    sets_.reserve(sizeof...(S));
    (sets_.emplace_back(std::make_unique<S>(std::forward<S>(sets))), ...);
  }
  template <class... S>
    requires(std::derived_from<S, Set> && ...)
  explicit MultiSet(std::unique_ptr<S>&&... sets) : sets_{}, dis_{0, sizeof...(S)} {
    sets_.reserve(sizeof...(S));
    (sets_.emplace_back(std::forward<S>(sets)), ...);
  }
  explicit MultiSet(std::vector<std::unique_ptr<Set>> sets)
      : sets_{std::move(sets)}, dis_{0, static_cast<int>(sets_.size())} {}

  /** @getter{sets, multi set} */
  [[nodiscard]] const std::vector<std::unique_ptr<Set>>& sets() const { return sets_; }

  [[nodiscard]] Dimension dimension() const override { return sets_.empty() ? 0 : sets_.front()->dimension(); }

  [[nodiscard]] Matrix sample_element(int num_samples) const override;

  [[nodiscard]] bool operator()(ConstMatrixRef x) const override;

  [[nodiscard]] Matrix lattice(const Eigen::VectorX<Index>& points_per_dim, bool include_endpoints) const override;

  void plot(const std::string& color) const override;
  void plot3d(const std::string& color) const override;

 private:
  std::vector<std::unique_ptr<Set>> sets_;  ///< Sets in the union
  std::uniform_int_distribution<> dis_;     ///< Random distribution for picking the set to sample from
};

std::ostream& operator<<(std::ostream& os, const MultiSet& set);

}  // namespace lucid

#ifdef LUCID_INCLUDE_FMT

#include "lucid/util/logging.h"

OSTREAM_FORMATTER(lucid::MultiSet)

#endif
