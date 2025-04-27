/**
 * @author Room 6.030
 * @copyright 2025 lucid
 * @licence BSD 3-Clause License
 */
#include <gtest/gtest.h>

#include <sstream>

#include "lucid/symbolic/ltl/AtomicProposition.h"
#include "lucid/symbolic/ltl/Formula.h"
#include "lucid/symbolic/ltl/Operator.h"
#include "lucid/util/exception.h"
#include "lucid/util/logging.h"

#define EXPECT_AP_EQ(lhs, rhs) EXPECT_TRUE((lhs).equal_to(rhs))
#define EXPECT_AP_ITERATOR_EQ(lhs, rhs)                                                                              \
  do {                                                                                                               \
    auto exp_lhs = (lhs);                                                                                            \
    auto exp_rhs = (rhs);                                                                                            \
    ASSERT_EQ(exp_lhs.size(), exp_rhs.size());                                                                       \
    EXPECT_TRUE(std::ranges::lexicographical_compare(exp_lhs.begin(), exp_lhs.end(), exp_rhs.begin(), exp_rhs.end(), \
                                                     [](const auto& l, const auto& r) { return l.equal_to(r); }));   \
  } while (0)

using lucid::ltl::AtomicProposition;
using lucid::ltl::Formula;
using lucid::ltl::Operator;

class TestFormula : public ::testing::Test {
 protected:
  const AtomicProposition ap1_{"p1"};
  const AtomicProposition ap2_{"p2"};
};

TEST_F(TestFormula, ConstructorWithAtomicProposition) {
  const Formula formula{ap1_};
  EXPECT_EQ(formula.op(), Operator::VARIABLE);
  EXPECT_EQ(formula.hash(), ap1_.hash());
  EXPECT_AP_EQ(formula.atomic_proposition(), ap1_);
  EXPECT_THROW(static_cast<void>(formula.operands()), lucid::exception::LucidNotSupportedException);
  EXPECT_THROW(static_cast<void>(formula.operand()), lucid::exception::LucidNotSupportedException);
  EXPECT_AP_ITERATOR_EQ(formula.atomic_propositions(), std::set<AtomicProposition>{ap1_});
  EXPECT_FALSE((std::stringstream{} << formula).str().empty());
}

TEST_F(TestFormula, NegationOperator) {
  const Formula formula{ap1_};
  const Formula negated = !formula;

  EXPECT_EQ(negated.op(), Operator::NOT);
  EXPECT_NE(formula.hash(), negated.hash());
  EXPECT_TRUE(negated.operand().equal_to(formula));
  EXPECT_THROW(static_cast<void>(negated.operands()), lucid::exception::LucidNotSupportedException);
  EXPECT_THROW(static_cast<void>(negated.atomic_proposition()), lucid::exception::LucidNotSupportedException);
  EXPECT_AP_ITERATOR_EQ(negated.atomic_propositions(), std::set<AtomicProposition>{ap1_});
  EXPECT_FALSE((std::stringstream{} << negated).str().empty());
}

TEST_F(TestFormula, ConjunctionOperator) {
  const Formula formula1{ap1_};
  const Formula formula2{ap2_};
  const Formula conjunction = formula1 && formula2;

  EXPECT_EQ(conjunction.op(), Operator::AND);
  EXPECT_NE(formula1.hash(), conjunction.hash());
  EXPECT_NE(formula2.hash(), conjunction.hash());
  EXPECT_TRUE(conjunction.operands().lhs.equal_to(formula1));
  EXPECT_TRUE(conjunction.operands().rhs.equal_to(formula2));
  EXPECT_THROW(static_cast<void>(conjunction.operand()), lucid::exception::LucidNotSupportedException);
  EXPECT_THROW(static_cast<void>(conjunction.atomic_proposition()), lucid::exception::LucidNotSupportedException);
  EXPECT_AP_ITERATOR_EQ(conjunction.atomic_propositions(), std::set<AtomicProposition>({ap1_, ap2_}));
  EXPECT_FALSE((std::stringstream{} << conjunction).str().empty());
}

TEST_F(TestFormula, DisjunctionOperator) {
  const Formula formula1{ap1_};
  const Formula formula2{ap2_};
  const Formula disjunction = formula1 || formula2;

  EXPECT_EQ(disjunction.op(), Operator::OR);
  EXPECT_NE(formula1.hash(), disjunction.hash());
  EXPECT_NE(formula2.hash(), disjunction.hash());
  EXPECT_TRUE(disjunction.operands().lhs.equal_to(formula1));
  EXPECT_TRUE(disjunction.operands().rhs.equal_to(formula2));
  EXPECT_THROW(static_cast<void>(disjunction.operand()), lucid::exception::LucidNotSupportedException);
  EXPECT_THROW(static_cast<void>(disjunction.atomic_proposition()), lucid::exception::LucidNotSupportedException);
  EXPECT_AP_ITERATOR_EQ(disjunction.atomic_propositions(), std::set<AtomicProposition>({ap1_, ap2_}));
  EXPECT_FALSE((std::stringstream{} << disjunction).str().empty());
}

TEST_F(TestFormula, UntilOperator) {
  const Formula formula1{ap1_};
  const Formula formula2{ap2_};
  const Formula until = formula1 % formula2;

  EXPECT_EQ(until.op(), Operator::UNTIL);
  EXPECT_NE(formula1.hash(), until.hash());
  EXPECT_NE(formula2.hash(), until.hash());
  EXPECT_TRUE(until.operands().lhs.equal_to(formula1));
  EXPECT_TRUE(until.operands().rhs.equal_to(formula2));
  EXPECT_THROW(static_cast<void>(until.operand()), lucid::exception::LucidNotSupportedException);
  EXPECT_THROW(static_cast<void>(until.atomic_proposition()), lucid::exception::LucidNotSupportedException);
  EXPECT_AP_ITERATOR_EQ(until.atomic_propositions(), std::set<AtomicProposition>({ap1_, ap2_}));
  EXPECT_FALSE((std::stringstream{} << until).str().empty());
}

TEST_F(TestFormula, NextOperator) {
  const Formula formula{ap1_};
  const Formula next = ++formula;

  EXPECT_EQ(next.op(), Operator::NEXT);
  EXPECT_NE(formula.hash(), next.hash());
  EXPECT_FALSE((std::stringstream{} << next).str().empty());
}

TEST_F(TestFormula, NestedLogicalOperations) {
  const Formula formula1{ap1_};
  const Formula formula2{ap2_};
  const Formula nested = !(formula1 && formula2) || ++formula1;

  EXPECT_EQ(nested.op(), Operator::OR);
  EXPECT_FALSE((std::stringstream{} << nested).str().empty());
}

TEST_F(TestFormula, EqualityOperator) {
  const Formula formula1{ap1_};
  const Formula formula2{ap1_};

  EXPECT_TRUE(formula1.equal_to(formula2));
}

TEST_F(TestFormula, LessOperator) {
  const Formula formula1{ap1_};
  const Formula formula2{ap2_};

  EXPECT_TRUE(formula1.less(formula2) || formula2.less(formula1));
}

TEST_F(TestFormula, HashFunction) {
  const Formula formula1{ap1_};
  const Formula formula2{!formula1};

  EXPECT_NE(formula1.hash(), formula2.hash());
}

TEST_F(TestFormula, Exp) {
  const Formula formula1{ap1_};
  const Formula formula2{!formula1};
  const AtomicProposition p1{"p1"}, p2{"p2"}, p3{"p3"};
  const Formula phi{!p2 % p1};
  fmt::print("phi: {}\n", phi);
  const Formula phi2{F(p1 && (!p2 % p3))};
  fmt::print("phi2: {}\n", phi2);
  fmt::print("phi3: {}\n", (p1 && ++p1 && ++(++p1) && ++(++(++p1))));
  FAIL();
}

// TEST_F(TestFormula, StreamOutput) {
//   const Formula formula{ap1_};
//   std::ostringstream oss;
//   oss << formula;
//
//   EXPECT_FALSE(oss.str().empty());
// }