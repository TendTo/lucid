/**
 * @author Room 6.030
 * @copyright 2025 lucid
 * @licence BSD 3-Clause License
 * @file
 */
#include "lucid/math/RectSet.h"

#include <ostream>
#include <random>
#include <utility>

#include "lucid/util/error.h"
#ifdef LUCID_MATPLOTLIB_BUILD
#include "lucid/util/matplotlibcpp.h"
#endif

namespace {

std::random_device rd;   // Will be used to obtain a seed for the random number engine
std::mt19937 gen(rd());  // Standard mersenne_twister_engine seeded with rd()
std::uniform_real_distribution<> dis(0.0, 1.0);

}  // namespace

namespace lucid {

RectSet::RectSet(Vector lb, Vector ub, const int seed) : lb_{std::move(lb)}, ub_{std::move(ub)} {
  if (lb_.size() != ub_.size()) LUCID_INVALID_ARGUMENT("lb and ub", "they must have the same size");
  if (seed >= 0) gen.seed(seed);
}
bool RectSet::operator()(ConstMatrixRef x) const {
  if (x.rows() != lb_.rows() || x.cols() != lb_.cols()) {
    LUCID_INVALID_ARGUMENT_EXPECTED("x shape", fmt::format("{} x {}", x.rows(), x.cols()),
                                   fmt::format("{} x {}", lb_.rows(), lb_.cols()));
  }
  return (x.array() >= lb_.array()).all() && (x.array() <= ub_.array()).all();
}

void RectSet::plot(const std::string& color) const {
#ifdef LUCID_MATPLOTLIB_BUILD
  namespace plt = matplotlibcpp;
  Vector x(lb_.size());
  x << lb_(0), ub_(0);
  Vector y1(1);
  y1 << lb_(1);
  Vector y2(1);
  y2 << ub_(1);
  plt::fill_between(x, y1, y2, 1, {{"edgecolor", color}, {"facecolor", "none"}});
#else
  LUCID_NOT_SUPPORTED("plot without matplotlib");
#endif
}

Matrix RectSet::sample_element(const int num_samples) const {
  Matrix samples(lb_.rows(), num_samples);
  for (int i = 0; i < num_samples; i++) {
    Vector random{lb_.size()};
    for (Index j = 0; j < lb_.size(); j++) random(j) = dis(gen);
    samples.col(i) = lb_ + (ub_ - lb_).cwiseProduct(random);
  }
  return samples;
}

std::ostream& operator<<(std::ostream& os, const RectSet& set) {
  return os << fmt::format("RectInterval[[{}], [{}]]", set.lower_bound(), set.upper_bound());
}

}  // namespace lucid