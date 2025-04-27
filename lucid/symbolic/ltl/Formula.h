/**
 * @author Room 6.030
 * @copyright 2025 lucid
 * @licence BSD 3-Clause License
 * @file
 * Formula class.
 */
#pragma once

#include <set>
#include <string>
#include <vector>

#include "lucid/symbolic/ltl/AtomicProposition.h"
#include "lucid/symbolic/ltl/FormulaPair.h"
#include "lucid/symbolic/ltl/Operator.h"
#include "lucid/util/intrusive_ptr.h"

namespace lucid::ltl {

// Forward declaration
class FormulaImpl;

/**
 * Symbolic representation of an [LTL](https://en.wikipedia.org/wiki/Linear_temporal_logic) formula.
 * The Formula class provides mechanisms to parse, evaluate, and manipulate LTL formulae.
 * It can handle the following operations:
 *
 * | Operator | Description | Math notation    | C++ notation |
 * |----------|-------------|------------------|--------------|
 * | `!`      | Negation    | @f$ \neg @f$     | `!`          |
 * | `&&`     | Conjunction | @f$ \land @f$    | `&&`         |
 * | `\|\| `  | Disjunction | @f$ \lor @f$     | `\|\|`       |
 * | `~`      | Next        | @f$ \bigcirc @f$ | `~`          |
 *
 */
class Formula {
  friend Formula F(const Formula &f);
  friend Formula F(const AtomicProposition &ap);
  friend Formula operator!(const AtomicProposition &ap);
  friend Formula operator&&(const AtomicProposition &lhs, const AtomicProposition &rhs);
  friend Formula operator||(const AtomicProposition &lhs, const AtomicProposition &rhs);
  friend Formula operator%(const AtomicProposition &lhs, const AtomicProposition &rhs);
  friend Formula operator&&(const AtomicProposition &lhs, const Formula &rhs);
  friend Formula operator||(const AtomicProposition &lhs, const Formula &rhs);
  friend Formula operator%(const AtomicProposition &lhs, const Formula &rhs);
  friend Formula operator~(const AtomicProposition &ap);
  friend Formula operator++(const AtomicProposition &ap);

 public:
  /**
   * Construct a new Formula object initialised with the given atomic proposition.
   * @param ap The atomic proposition used to create the Formula.
   */
  Formula(const AtomicProposition &ap);
  Formula(const Formula &e) = default;
  Formula(Formula &&e) noexcept = default;
  Formula &operator=(const Formula &e) = default;
  Formula &operator=(Formula &&e) noexcept = default;
  ~Formula();

  /** @getter{operand, formula} */
  [[nodiscard]] const Formula &operand() const;
  /** @getter{pair of operands (left-hand-side and right-hand-size), formula} */
  [[nodiscard]] FormulaPair operands() const;
  /** @getter{atomic proposition, formula} */
  [[nodiscard]] const AtomicProposition &atomic_proposition() const;
  /** @getter{all the atomic propositions collected recursively, formula} */
  [[nodiscard]] std::set<AtomicProposition> atomic_propositions() const;
  /** @getter{operator, formula} */
  [[nodiscard]] Operator op() const noexcept;
  /** @hash{formula} */
  [[nodiscard]] std::size_t hash() const noexcept;
  /** @equal_to{formula} */
  [[nodiscard]] bool equal_to(const Formula &o) const noexcept;
  /** @less{formula} */
  [[nodiscard]] bool less(const Formula &o) const noexcept;
  /** @getter{pointer, formula} */
  [[nodiscard]] intrusive_ptr<const FormulaImpl> ptr() const noexcept;
  /** @getter{implementation, formula} */
  [[nodiscard]] const FormulaImpl &impl() const noexcept;

  /** Create a new formula that is the negation of the current formula. */
  [[nodiscard]] Formula operator!() const;
  /** Combines the current formula with another formula using the logical AND operator. */
  [[nodiscard]] Formula operator&&(const Formula &o) const;
  /** Combines the current formula with another formula using the logical OR operator. */
  [[nodiscard]] Formula operator||(const Formula &o) const;
  /** Creates a new formula obtained by applying the `next` operator. */
  [[nodiscard]] Formula operator++() const;
  /** Creates a new formula obtained by applying the `until` operator. */
  [[nodiscard]] Formula operator%(const Formula &o) const;

 private:
  /**
   * Construct a new Formula object from an existing FormulaImpl object.
   * The formula will just be a wrapper around the FormulaImpl object, without copying any data.
   * @param ptr Pointer to the FormulaImpl object
   */
  explicit Formula(const intrusive_ptr<FormulaImpl> &ptr);

  intrusive_ptr<FormulaImpl> ptr_;  ///< Internal smart pointer to the FormulaImpl
};

/** Create a new formula obtained by applying the `finally` operator. */
Formula F(const Formula &f);
Formula F(const AtomicProposition &ap);

Formula operator!(const AtomicProposition &ap);
Formula operator&&(const AtomicProposition &lhs, const AtomicProposition &rhs);
Formula operator||(const AtomicProposition &lhs, const AtomicProposition &rhs);
Formula operator%(const AtomicProposition &lhs, const AtomicProposition &rhs);
Formula operator++(const AtomicProposition &ap);

Formula operator&&(const AtomicProposition &lhs, const Formula &rhs);
Formula operator||(const AtomicProposition &lhs, const Formula &rhs);
Formula operator%(const AtomicProposition &lhs, const Formula &rhs);

std::ostream &operator<<(std::ostream &os, const Formula &f);

}  // namespace lucid::ltl

template <>
struct std::hash<lucid::ltl::Formula> {
  std::size_t operator()(const lucid::ltl::Formula &v) const noexcept { return v.hash(); }
};

template <>
struct std::less<lucid::ltl::Formula> {
  bool operator()(const lucid::ltl::Formula &lhs, const lucid::ltl::Formula &rhs) const noexcept {
    return lhs.less(rhs);
  }
};

template <>
struct std::equal_to<lucid::ltl::Formula> {
  bool operator()(const lucid::ltl::Formula &lhs, const lucid::ltl::Formula &rhs) const noexcept {
    return lhs.equal_to(rhs);
  }
};

#ifdef LUCID_INCLUDE_FMT

#include "lucid/util/logging.h"

OSTREAM_FORMATTER(lucid::ltl::Formula);

#endif
