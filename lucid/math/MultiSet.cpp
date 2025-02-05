/**
 * @author Room 6.030
 * @copyright 2025 lucid
 * @licence BSD 3-Clause License
 * @file
 */
#include "lucid/math/MultiSet.h"

#include <random>

namespace lucid {

namespace {

std::random_device rd;
std::mt19937 gen(rd());
std::uniform_int_distribution<std::size_t> dist;

}  // namespace

Matrix MultiSet::sample_element(const int num_samples) const {
  if (sets_.empty()) return Matrix::Zero(0, 0);
  // TODO(tend): not thread-safe
  dist.param(std::uniform_int_distribution<std::size_t>::param_type{0, sets_.size() - 1});
  Matrix samples(dimension(), num_samples);
  for (int i = 0; i < num_samples; i++) samples.col(i) = sets_.at(dist(gen))->sample_element();
  return samples;
}
bool MultiSet::operator()(ConstMatrixRef x) const {
  return std::ranges::any_of(sets_, [&x](const std::unique_ptr<Set>& set) { return set->contains(x); });
}
void MultiSet::plot(const std::string& color) const {
  std::ranges::for_each(sets_, [&color](const std::unique_ptr<Set>& set) { set->plot(color); });
}

std::ostream& operator<<(std::ostream& os, const MultiSet& set) {
  os << "MultiSet\n";
  bool first = false;
  for (const std::unique_ptr<Set>& s : set.sets()) {
    if (!first) {
      os << '\n';
      first = true;
    }
    os << *s;
  }
  return os;
}
}  // namespace lucid
