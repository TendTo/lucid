/**
 * @author Room 6.030
 * @copyright 2025 lucid
 * @licence BSD 3-Clause License
 * @file
 * Formula class.
 */
#include "lucid/symbolic/ltl/Formula.h"

#include "lucid/symbolic/ltl/FormulaImpl.h"
#include "lucid/util/error.h"

namespace lucid::ltl {

Formula::Formula(const AtomicProposition& ap) : ptr_{AtomicPropositionFormulaImpl::instantiate(ap)} {}
Formula::~Formula() {}  // Needed to avoid compiler errors with the smart pointer
const Formula& Formula::operand() const { return ptr_->operand(); }
FormulaPair Formula::operands() const { return ptr_->operands(); }
const AtomicProposition& Formula::atomic_proposition() const { return ptr_->atomic_proposition(); }
std::set<AtomicProposition> Formula::atomic_propositions() const { return ptr_->atomic_propositions(); }
Formula::Formula(const intrusive_ptr<FormulaImpl>& ptr) : ptr_{ptr} {}
Operator Formula::op() const noexcept { return ptr_->op(); }
std::size_t Formula::hash() const noexcept { return ptr_->hash(); }
bool Formula::equal_to(const Formula& o) const noexcept { return ptr_->equal_to(o); }
bool Formula::less(const Formula& o) const noexcept { return ptr_->less(o); }
intrusive_ptr<const FormulaImpl> Formula::ptr() const noexcept { return intrusive_ptr<const FormulaImpl>{ptr_.get()}; }
const FormulaImpl& Formula::impl() const noexcept { return *ptr_; }

Formula Formula::operator!() const { return Formula{NotFormulaImpl::instantiate(*this)}; }
Formula Formula::operator&&(const Formula& o) const { return Formula{AndFormulaImpl::instantiate(*this, o)}; }
Formula Formula::operator||(const Formula& o) const { return Formula{OrFormulaImpl::instantiate(*this, o)}; }
Formula Formula::operator++() const { return Formula{NextFormulaImpl::instantiate(*this)}; }
Formula Formula::operator%(const Formula& o) const { return Formula{UntilFormulaImpl::instantiate(*this, o)}; }

Formula F(const Formula& f) { return Formula{FinallyFormulaImpl::instantiate(f)}; }
Formula F(const AtomicProposition& ap) { return Formula{FinallyFormulaImpl::instantiate(ap)}; }

Formula operator!(const AtomicProposition& ap) { return Formula{NotFormulaImpl::instantiate(ap)}; }
Formula operator&&(const AtomicProposition& lhs, const AtomicProposition& rhs) {
  return Formula{AndFormulaImpl::instantiate(lhs, rhs)};
}
Formula operator||(const AtomicProposition& lhs, const AtomicProposition& rhs) {
  return Formula{OrFormulaImpl::instantiate(lhs, rhs)};
}
Formula operator%(const AtomicProposition& lhs, const AtomicProposition& rhs) {
  return Formula{UntilFormulaImpl::instantiate(lhs, rhs)};
}
Formula operator++(const AtomicProposition& ap) { return Formula{NextFormulaImpl::instantiate(ap)}; }
Formula operator&&(const AtomicProposition& lhs, const Formula& rhs) {
  return Formula{AndFormulaImpl::instantiate(lhs, rhs)};
}
Formula operator||(const AtomicProposition& lhs, const Formula& rhs) {
  return Formula{OrFormulaImpl::instantiate(lhs, rhs)};
}
Formula operator%(const AtomicProposition& lhs, const Formula& rhs) {
  return Formula{UntilFormulaImpl::instantiate(lhs, rhs)};
}

std::ostream& operator<<(std::ostream& os, const Formula& f) { return os << f.impl(); }

}  // namespace lucid::ltl