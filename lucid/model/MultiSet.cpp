/**
 * @author Room 6.030
 * @copyright 2025 lucid
 * @licence BSD 3-Clause License
 * @file
 */
#include "lucid/model/MultiSet.h"

#include <memory>
#include <string>
#include <utility>
#include <vector>

#include "lucid/model/RectSet.h"
#include "lucid/util/error.h"
#include "lucid/util/random.h"

namespace lucid {

namespace {

std::uniform_int_distribution<std::size_t> dist;

}  // namespace

MultiSet::MultiSet(std::vector<std::unique_ptr<Set>> sets) : sets_{std::move(sets)} {
#ifndef NCHECK
  validate();
#endif
}
Matrix MultiSet::sample(const Index num_samples) const {
  if (sets_.empty()) return Matrix::Zero(0, 0);
  // TODO(tend): not thread-safe
  // TODO(tend): A random one is selected uniformly and then the sample is taken from that set.
  // This is clearly not uniform in terms of the probability distribution of the union of the sets.
  dist.param(std::uniform_int_distribution<std::size_t>::param_type{0, sets_.size() - 1});
  Matrix samples(num_samples, dimension());
  for (int i = 0; i < num_samples; i++) samples.row(i) = sets_.at(dist(random::gen))->sample();
  return samples;
}
bool MultiSet::operator()(ConstVectorRef x) const {
  return std::ranges::any_of(sets_, [&x](const std::unique_ptr<Set>& set) { return set->contains(x); });
}
Matrix MultiSet::lattice(const VectorI& points_per_dim, const bool endpoint) const {
  Matrix rect_multiset_lattice{0, dimension()};
  for (const auto& set : sets_) {
    const Matrix initial_lattice{set->lattice(points_per_dim, endpoint)};
    rect_multiset_lattice.conservativeResize(rect_multiset_lattice.rows() + initial_lattice.rows(),
                                             initial_lattice.cols());
    rect_multiset_lattice.bottomRows(initial_lattice.rows()) = initial_lattice;
  }
  return rect_multiset_lattice;
}
void MultiSet::change_size(ConstVectorRef delta_size) {
  for (const auto& set : sets_) set->change_size(delta_size);
}
Vector MultiSet::general_lower_bound() const {
  LUCID_ASSERT(!sets_.empty(), "MultiSet must contain at least one set to get the general lower bound");
  Vector lower_bound = sets_.front()->general_lower_bound();
  for (const auto& set : sets_) {
    lower_bound = lower_bound.cwiseMin(set->general_lower_bound());
  }
  return lower_bound;
}
Vector MultiSet::general_upper_bound() const {
  LUCID_ASSERT(!sets_.empty(), "MultiSet must contain at least one set to get the general upper bound");
  Vector upper_bound = sets_.front()->general_upper_bound();
  for (const auto& set : sets_) {
    upper_bound = upper_bound.cwiseMax(set->general_upper_bound());
  }
  return upper_bound;
}
std::unique_ptr<Set> MultiSet::to_rect_set() const {
  std::vector<std::unique_ptr<Set>> rect_sets;
  rect_sets.reserve(sets_.size());
  for (const auto& set : sets_) {
    rect_sets.emplace_back(set->to_rect_set());
  }
  return std::make_unique<MultiSet>(std::move(rect_sets));
}
std::unique_ptr<Set> MultiSet::scale_wrapped_impl(ConstVectorRef scale, const RectSet& bounds,
                                                  const bool relative_to_bounds) const {
  std::vector<std::unique_ptr<Set>> scaled_sets;
  scaled_sets.reserve(sets_.size());
  for (const auto& set : sets_) {
    scaled_sets.emplace_back(set->scale_wrapped(scale, bounds, relative_to_bounds));
  }
  return std::make_unique<MultiSet>(std::move(scaled_sets));
}
void MultiSet::validate() {
#ifndef NCHECK
  LUCID_CHECK_ARGUMENT_CMP(sets_.size(), >, 0);
  [[maybe_unused]] const Dimension dim = sets_.front()->dimension();
  LUCID_CHECK_ARGUMENT(
      std::ranges::all_of(sets_, [dim](const std::unique_ptr<Set>& set) { return set->dimension() == dim; }), "sets",
      "all sets must have the same dimension");
#endif
}
std::string MultiSet::to_string() const {
  std::string result = "MultiSet( ";
  for (const std::unique_ptr<Set>& s : sets_) result += s->to_string() + " ";
  result += ")";
  return result;
}

std::ostream& operator<<(std::ostream& os, const MultiSet& set) { return os << set.to_string(); }

}  // namespace lucid
