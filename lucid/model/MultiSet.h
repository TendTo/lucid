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
#include <utility>
#include <vector>

#include "lucid/model/Set.h"

namespace lucid {

// Forward declaration
class RectSet;

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
    validate();
  }
  template <class... S>
    requires(std::derived_from<S, Set> && ...)
  explicit MultiSet(std::unique_ptr<S>&&... sets) : sets_{} {
    sets_.reserve(sizeof...(S));
    (sets_.emplace_back(std::forward<S>(sets)), ...);
    validate();
  }
  explicit MultiSet(std::vector<std::unique_ptr<Set>> sets);

  /** @getter{sets, multi set} */
  [[nodiscard]] const std::vector<std::unique_ptr<Set>>& sets() const { return sets_; }

  [[nodiscard]] Dimension dimension() const override { return sets_.empty() ? 0 : sets_.front()->dimension(); }
  [[nodiscard]] Matrix sample(Index num_samples) const override;
  [[nodiscard]] bool operator()(ConstVectorRef x) const override;
  [[nodiscard]] const Set& operator[](const std::size_t index) const { return *sets_.at(index); }

  /** @todo Improve the naive implementation that only concatenates the lattices from the internal sets (polytopes?) */
  [[nodiscard]] Matrix lattice(const VectorI& points_per_dim, bool endpoint) const override;

  void change_size(ConstVectorRef delta_size) override;

  [[nodiscard]] Vector general_lower_bound() const override;
  [[nodiscard]] Vector general_upper_bound() const override;

  [[nodiscard]] std::unique_ptr<Set> to_rect_set() const override;

 private:
  [[nodiscard]] std::unique_ptr<Set> scale_wrapped_impl(ConstVectorRef scale, const RectSet& bounds,
                                                        bool relative_to_bounds) const override;
  /** Utility function to validate the MultiSet. */
  void validate();

  std::vector<std::unique_ptr<Set>> sets_;  ///< Sets in the union
};

std::ostream& operator<<(std::ostream& os, const MultiSet& set);

}  // namespace lucid

#ifdef LUCID_INCLUDE_FMT

#include "lucid/util/logging.h"

OSTREAM_FORMATTER(lucid::MultiSet)

#endif
