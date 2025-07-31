/**
 * @author Room 6.030
 * @copyright 2025 lucid
 * @licence BSD 3-Clause License
 * @file
 * PolytopeSet class.
 */
#pragma once

#include <optional>
#include <utility>
#include <vector>

#include "lucid/model/Set.h"

namespace lucid {

/**
 * A convex polytope set in half-space representation.
 * A vector @x is in the set if @f$ A x \le b @f$,
 * where @f$ A @f$ is a matrix of hyperplane normals and @f$ b @f$ is a vector of hyperplane offsets.
 */
class PolytopeSet final : public Set {
 public:
  using Set::lattice;

  /**
   * Construct a polytope from hyperplane normals and offsets.
   * @param A matrix of hyperplane normals (each row is a normal vector)
   * @param b vector of hyperplane offsets
   */
  PolytopeSet(Matrix A, Vector b);

  /**
   * Construct a polytope from hyperplane normals and offsets.
   * @param A matrix of hyperplane normals (each row is a normal vector)
   * @param b vector of hyperplane offsets
   */
  PolytopeSet(std::initializer_list<std::initializer_list<Scalar>> A, std::initializer_list<Scalar> b);

  /**
   * Create a polytope from a bounding box (hyperrectangle).
   * @param bounds vector of pairs of lower and upper bounds
   * @return polytope representing the hyperrectangle
   */
  static PolytopeSet from_box(const std::vector<std::pair<Scalar, Scalar>>& bounds);

  [[nodiscard]] Dimension dimension() const override { return A_.cols(); }
  [[nodiscard]] Matrix sample(Index num_samples) const override;
  [[nodiscard]] bool operator()(ConstVectorRef x) const override;
  [[nodiscard]] Matrix lattice(const VectorI& points_per_dim, bool include_endpoints) const override;

  /**
   * Scale the polytope by a factor.
   * The constraints @f$ A x \le b @f$ are scaled to @f$ A x \le s b @f$.
   * @param factor scaling factor @f$ s @f$
   */
  void scale(Scalar factor);

  /** @getter{hyperplane normals matrix, polytope} */
  [[nodiscard]] const Matrix& A() const { return A_; }
  /** @getter{hyperplane offsets vector, polytope} */
  [[nodiscard]] const Vector& b() const { return b_; }
  [[nodiscard]] const std::pair<Vector, Vector>& bounding_box() const;

 private:
  /**
   * Compute bounding box of the polytope using linear programming.
   * @return pair of lower and upper bounds
   */
  std::pair<Vector, Vector> compute_bounding_box() const;

  Matrix A_;                                               ///< Hyperplane normals matrix (m x n)
  Vector b_;                                               ///< Hyperplane offsets vector (m)
  mutable std::optional<std::pair<Vector, Vector>> bbox_;  ///< Cached bounding box
};

std::ostream& operator<<(std::ostream& os, const PolytopeSet& set);

}  // namespace lucid

#ifdef LUCID_INCLUDE_FMT

#include "lucid/util/logging.h"

OSTREAM_FORMATTER(lucid::PolytopeSet)

#endif
