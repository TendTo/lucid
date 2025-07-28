/**
 * @author lucid_authors
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
#include <utility>
#include <vector>

#include "lucid/model/Set.h"

namespace lucid {

/**
 * Set composed of the union of multiple sets.
 * Checking whether a vector is in the set is equivalent to checking if it is in any of the sets.
 * @todo Simplistic implementation of sampling.
 */
class MultiSet final : public Set {
 public:
  using Set::lattice;

  template <class... S>
    requires(std::derived_from<S, Set> && ...)
  explicit MultiSet(S&&... sets) : sets_{} {
    sets_.reserve(sizeof...(S));
    (sets_.emplace_back(std::make_unique<S>(std::forward<S>(sets))), ...);
#ifndef NCHECK
    validate();
#endif
  }
  template <class... S>
    requires(std::derived_from<S, Set> && ...)
  explicit MultiSet(std::unique_ptr<S>&&... sets) : sets_{} {
    sets_.reserve(sizeof...(S));
    (sets_.emplace_back(std::forward<S>(sets)), ...);
#ifndef NCHECK
    validate();
#endif
  }
  explicit MultiSet(std::vector<std::unique_ptr<Set>> sets);

  /** @getter{sets, multi set} */
  [[nodiscard]] const std::vector<std::unique_ptr<Set>>& sets() const { return sets_; }

  [[nodiscard]] Dimension dimension() const override { return sets_.empty() ? 0 : sets_.front()->dimension(); }
  [[nodiscard]] Matrix sample(Index num_samples) const override;
  [[nodiscard]] bool operator()(ConstVectorRef x) const override;

  /** @todo Improve the naive implementation that only concatenates the lattices from the internal sets (polytopes?) */
  [[nodiscard]] Matrix lattice(const VectorI& points_per_dim, bool include_endpoints) const override;

 private:
#ifndef NCHECK
  /** Utility function to validate the MultiSet. */
  void validate();
#endif
  std::vector<std::unique_ptr<Set>> sets_;  ///< Sets in the union
};

std::ostream& operator<<(std::ostream& os, const MultiSet& set);

}  // namespace lucid

#ifdef LUCID_INCLUDE_FMT

#include "lucid/util/logging.h"

OSTREAM_FORMATTER(lucid::MultiSet)

#endif
