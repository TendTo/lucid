/**
 * @author Room 6.030
 * @copyright 2025 lucid
 * @licence BSD 3-Clause License
 * @file
 */
#include "lucid/math/MultiSet.h"

#include <memory>
#include <string>

#include "lucid/util/error.h"

namespace lucid {

namespace {

std::random_device rd;
std::mt19937 gen(rd());
std::uniform_int_distribution<std::size_t> dist;

}  // namespace

Matrix MultiSet::sample_element(const Index num_samples) const {
  if (sets_.empty()) return Matrix::Zero(0, 0);
  // TODO(tend): not thread-safe
  // TODO(tend): A random one is selected uniformly and then the sample is taken from that set.
  // This is clearly not uniform in terms of the probability distribution of the union of the sets.
  dist.param(std::uniform_int_distribution<std::size_t>::param_type{0, sets_.size() - 1});
  Matrix samples(num_samples, dimension());
  for (int i = 0; i < num_samples; i++) samples.row(i) = sets_.at(dist(gen))->sample_element();
  return samples;
}
bool MultiSet::operator()(ConstMatrixRef x) const {
  return std::ranges::any_of(sets_, [&x](const std::unique_ptr<Set>& set) { return set->contains(x); });
}
Matrix MultiSet::lattice(const Eigen::VectorX<Index>& points_per_dim, const bool include_endpoints) const {
  Matrix rect_multiset_lattice{0, dimension()};
  for (const auto& set : sets_) {
    const Matrix initial_lattice{set->lattice(points_per_dim, include_endpoints)};
    rect_multiset_lattice.conservativeResize(rect_multiset_lattice.rows() + initial_lattice.rows(),
                                             initial_lattice.cols());
    rect_multiset_lattice.bottomRows(initial_lattice.rows()) = initial_lattice;
  }
  return rect_multiset_lattice;
}
void MultiSet::plot(const std::string& color) const {
  std::ranges::for_each(sets_, [&color](const std::unique_ptr<Set>& set) { set->plot(color); });
}
void MultiSet::plot3d(const std::string& color) const {
  std::ranges::for_each(sets_, [&color](const std::unique_ptr<Set>& set) { set->plot3d(color); });
}

std::ostream& operator<<(std::ostream& os, const MultiSet& set) {
  os << "MultiSet(";
  bool first = true;
  for (const std::unique_ptr<Set>& s : set.sets()) {
    if (!first) os << ', ';
    os << *s;
    first = false;
  }
  return os << ")";
}
}  // namespace lucid
