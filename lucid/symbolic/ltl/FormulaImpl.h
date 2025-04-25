/**
 * @author Room 6.030
 * @copyright 2025 lucid
 * @licence BSD 3-Clause License
 * @file
 * Formula class.
 */
#pragma once

#include <optional>
#include <set>
#include <string>
#include <utility>
#include <vector>

#include "lucid/symbolic/ltl/AtomicProposition.h"
#include "lucid/symbolic/ltl/Formula.h"
#include "lucid/symbolic/ltl/FormulaPair.h"
#include "lucid/symbolic/ltl/Operator.h"
#include "lucid/util/SelfReferenceCountingObject.h"
#include "lucid/util/intrusive_ptr.h"

namespace lucid::ltl {

/**
 * Symbolic representation of an [LTL](https://en.wikipedia.org/wiki/Linear_temporal_logic) formula.
 * The FormulaImpl class is responsible for managing and evaluating mathematical formulas or computational expressions.
 * It provides functionalities to parse, store, and compute the results of a formula based on the provided inputs.
 */
class FormulaImpl : public SelfReferenceCountingObject {
 protected:
  explicit FormulaImpl(const Operator op) : op_{op} {}

  /**
   * Computes the hash value for the formula.
   * This is a pure virtual method that must be implemented by derived classes to calculate and define
   * the hash value specific to their respective formula structure.
   * The value is stored in the @ref hash_ member variable.
   */
  virtual void compute_hash() const = 0;

 public:
  /** @getter{operator, formula} */
  [[nodiscard]] Operator op() const { return op_; }
  /** @getter{operand, formula} */
  [[nodiscard]] virtual const Formula& operand() const;
  /** @getter{pair of operands (left-hand-side and right-hand-size), formula} */
  [[nodiscard]] virtual FormulaPair operands() const;
  /** @getter{atomic proposition, formula} */
  [[nodiscard]] virtual const AtomicProposition& atomic_proposition() const;
  /** @getter{all the atomic propositions collected recursively, formula} */
  [[nodiscard]] virtual std::set<AtomicProposition> atomic_propositions() const;

  /** @hash{formula} */
  [[nodiscard]] std::size_t hash() const noexcept {
    if (!hash_) compute_hash();
    return *hash_;
  }
  /** @equal_to{formula} */
  [[nodiscard]] virtual bool equal_to(const Formula& o) const noexcept = 0;
  /** @less{formula} */
  [[nodiscard]] virtual bool less(const Formula& o) const noexcept = 0;

 protected:
  mutable std::optional<std::size_t> hash_;  ///< Hash value of the formula. Computed on demand and cached
  Operator op_;                              ///< Operator that created the formula
};

class UnaryFormulaImpl : public FormulaImpl {
 protected:
  UnaryFormulaImpl(Formula f, const Operator op) : FormulaImpl{op}, f_{std::move(f)} {}

 public:
  /** @getter{formula, formula} */
  [[nodiscard]] const Formula& f() const { return f_; }
  [[nodiscard]] bool less(const Formula& o) const noexcept override;
  [[nodiscard]] bool equal_to(const Formula& o) const noexcept override;
  [[nodiscard]] const Formula& operand() const override { return f_; }
  [[nodiscard]] std::set<AtomicProposition> atomic_propositions() const override { return f_.atomic_propositions(); }

 protected:
  void compute_hash() const override;

  Formula f_;
};

class BinaryFormulaImpl : public FormulaImpl {
 protected:
  BinaryFormulaImpl(Formula lhs, Formula rhs, const Operator op)
      : FormulaImpl{op}, lhs_{std::move(lhs)}, rhs_{std::move(rhs)} {}

 public:
  /** @getter{left-hand side, formula} */
  [[nodiscard]] const Formula& lhs() const { return lhs_; }
  /** @getter{right-hand side, formula} */
  [[nodiscard]] const Formula& rhs() const { return rhs_; }
  [[nodiscard]] bool less(const Formula& o) const noexcept override;
  [[nodiscard]] bool equal_to(const Formula& o) const noexcept override;
  [[nodiscard]] FormulaPair operands() const override { return {lhs_, rhs_}; }
  [[nodiscard]] std::set<AtomicProposition> atomic_propositions() const override {
    std::set<AtomicProposition> aps{lhs_.atomic_propositions()};
    aps.merge(rhs_.atomic_propositions());
    return aps;
  }

 protected:
  void compute_hash() const override;

  Formula lhs_;
  Formula rhs_;
};

class AtomicPropositionFormulaImpl final : public FormulaImpl {
 public:
  [[nodiscard]] static intrusive_ptr<FormulaImpl> instantiate(const AtomicProposition& ap) {
    return intrusive_ptr<FormulaImpl>{new AtomicPropositionFormulaImpl{ap}};
  }

  [[nodiscard]] const AtomicProposition& ap() const { return ap_; }
  [[nodiscard]] bool less(const Formula& o) const noexcept override;
  [[nodiscard]] bool equal_to(const Formula& o) const noexcept override;
  [[nodiscard]] const AtomicProposition& atomic_proposition() const override { return ap_; }
  [[nodiscard]] std::set<AtomicProposition> atomic_propositions() const override { return {ap_}; }

 protected:
  explicit AtomicPropositionFormulaImpl(const AtomicProposition& ap) : FormulaImpl{Operator::VARIABLE}, ap_{ap} {}
  void compute_hash() const override { hash_ = ap_.id(); }

  AtomicProposition ap_;
};

class NotFormulaImpl final : public UnaryFormulaImpl {
 public:
  [[nodiscard]] static intrusive_ptr<FormulaImpl> instantiate(const Formula& f) {
    return intrusive_ptr<FormulaImpl>{new NotFormulaImpl{f}};
  }

 protected:
  explicit NotFormulaImpl(const Formula& f) : UnaryFormulaImpl{f, Operator::NOT} {}
};

class AndFormulaImpl final : public BinaryFormulaImpl {
 public:
  [[nodiscard]] static intrusive_ptr<FormulaImpl> instantiate(const Formula& lhs, const Formula& rhs) {
    return intrusive_ptr<FormulaImpl>{new AndFormulaImpl{lhs, rhs}};
  }

 protected:
  AndFormulaImpl(const Formula& lhs, const Formula& rhs) : BinaryFormulaImpl{lhs, rhs, Operator::AND} {}
};

class OrFormulaImpl final : public BinaryFormulaImpl {
 public:
  [[nodiscard]] static intrusive_ptr<FormulaImpl> instantiate(const Formula& lhs, const Formula& rhs) {
    return intrusive_ptr<FormulaImpl>{new OrFormulaImpl{lhs, rhs}};
  }

 protected:
  OrFormulaImpl(const Formula& lhs, const Formula& rhs) : BinaryFormulaImpl{lhs, rhs, Operator::OR} {}
};

class UntilFormulaImpl final : public BinaryFormulaImpl {
 public:
  [[nodiscard]] static intrusive_ptr<FormulaImpl> instantiate(const Formula& lhs, const Formula& rhs) {
    return intrusive_ptr<FormulaImpl>{new UntilFormulaImpl{lhs, rhs}};
  }

 protected:
  UntilFormulaImpl(const Formula& lhs, const Formula& rhs) : BinaryFormulaImpl{lhs, rhs, Operator::UNTIL} {}
};

class NextFormulaImpl final : public UnaryFormulaImpl {
 public:
  [[nodiscard]] static intrusive_ptr<FormulaImpl> instantiate(const Formula& f) {
    return intrusive_ptr<FormulaImpl>{new NextFormulaImpl{f}};
  }

 protected:
  explicit NextFormulaImpl(const Formula& f) : UnaryFormulaImpl{f, Operator::NEXT} {}
};

class AlwaysFormulaImpl final : public UnaryFormulaImpl {
 public:
  [[nodiscard]] static intrusive_ptr<FormulaImpl> instantiate(const Formula& f) {
    return intrusive_ptr<FormulaImpl>{new AlwaysFormulaImpl{f}};
  }

 protected:
  explicit AlwaysFormulaImpl(const Formula& f) : UnaryFormulaImpl{f, Operator::ALWAYS} {}
};

class EventuallyFormulaImpl final : public UnaryFormulaImpl {
 public:
  [[nodiscard]] static intrusive_ptr<FormulaImpl> instantiate(const Formula& f) {
    return intrusive_ptr<FormulaImpl>{new EventuallyFormulaImpl{f}};
  }

 protected:
  explicit EventuallyFormulaImpl(const Formula& f) : UnaryFormulaImpl{f, Operator::EVENTUALLY} {}
};

}  // namespace lucid::ltl

// #ifdef LUCID_INCLUDE_FMT
//
// #include "lucid/util/logging.h"
//
// OSTREAM_FORMATTER(lucid::ltl::FormulaImpl);
//
// #endif
