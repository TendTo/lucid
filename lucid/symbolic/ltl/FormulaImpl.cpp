/**
 * @author Room 6.030
 * @copyright 2025 lucid
 * @licence BSD 3-Clause License
 * @file
 */

#include "lucid/symbolic/ltl/FormulaImpl.h"

#include "lucid/util/error.h"
#include "lucid/util/hash.h"

namespace lucid::ltl {

std::set<AtomicProposition> FormulaImpl::atomic_propositions() const { return {}; }
const Formula& FormulaImpl::operand() const {
  LUCID_NOT_SUPPORTED("Retrieving an operand from a formula which is not a unary formula.");
}
FormulaPair FormulaImpl::operands() const {
  LUCID_NOT_SUPPORTED("Retrieving a pair of operands from a formula which is not a binary formula.");
}
const AtomicProposition& FormulaImpl::atomic_proposition() const {
  LUCID_NOT_SUPPORTED("Retrieving an atomic proposition from a formula which is not an atomic proposition.");
}

bool AtomicPropositionFormulaImpl::less(const Formula& o) const noexcept {
  if (this == &o.impl()) return false;
  if (op_ < o.op()) return true;
  if (op_ > o.op()) return false;
  return ap_.less(static_cast<const AtomicPropositionFormulaImpl&>(o.impl()).ap_);
}
bool AtomicPropositionFormulaImpl::equal_to(const Formula& o) const noexcept {
  if (this == &o.impl()) return true;
  if (op_ != o.op()) return false;
  return ap_.equal_to(static_cast<const AtomicPropositionFormulaImpl&>(o.impl()).ap_);
}

bool UnaryFormulaImpl::less(const Formula& o) const noexcept {
  if (this == &o.impl()) return false;
  if (op_ < o.op()) return true;
  if (op_ > o.op()) return false;
  return f_.less(static_cast<const UnaryFormulaImpl&>(o.impl()).f_);
}
bool UnaryFormulaImpl::equal_to(const Formula& o) const noexcept {
  if (this == &o.impl()) return true;
  if (op_ != o.op()) return false;
  return f_.equal_to(static_cast<const UnaryFormulaImpl&>(o.impl()).f_);
}

bool BinaryFormulaImpl::less(const Formula& o) const noexcept {
  if (this == &o.impl()) return false;
  if (op_ < o.op()) return true;
  if (op_ > o.op()) return false;
  if (lhs_.less(static_cast<const BinaryFormulaImpl&>(o.impl()).lhs_)) return true;
  if (rhs_.less(static_cast<const BinaryFormulaImpl&>(o.impl()).rhs_)) return true;
  return false;
}
bool BinaryFormulaImpl::equal_to(const Formula& o) const noexcept {
  if (this == &o.impl()) return true;
  if (op_ != o.op()) return false;
  return lhs_.equal_to(static_cast<const BinaryFormulaImpl&>(o.impl()).lhs_) &&
         rhs_.equal_to(static_cast<const BinaryFormulaImpl&>(o.impl()).rhs_);
}

void UnaryFormulaImpl::compute_hash() const { hash::hash_combine(0, op_, f_); }
void BinaryFormulaImpl::compute_hash() const { hash::hash_combine(0, lhs_, op_, rhs_); }

std::ostream& UnaryFormulaImpl::print(std::ostream& os) const { return os << "(" << op_ << f_ << ")"; }
std::ostream& BinaryFormulaImpl::print(std::ostream& os) const {
  return os << "(" << lhs_ << " " << op_ << " " << rhs_ << ")";
}

std::ostream& operator<<(std::ostream& os, const FormulaImpl& f) { return f.print(os); }

}  // namespace lucid::ltl
