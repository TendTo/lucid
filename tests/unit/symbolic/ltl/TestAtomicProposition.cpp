/**
 * @author Room 6.030
 * @copyright 2025 lucid
 * @licence BSD 3-Clause License
 */
#include <gtest/gtest.h>

#include "lucid/symbolic/ltl/AtomicProposition.h"

using lucid::ltl::AtomicProposition;

TEST(TestAtomicProposition, DefaultConstructorCreatesDummy) {
  const AtomicProposition dummy;
  EXPECT_TRUE(dummy.is_dummy());
  EXPECT_EQ(dummy.id(), AtomicProposition::dummy_id);
  EXPECT_EQ(dummy.name(), "dummy");
}

TEST(TestAtomicProposition, ConstructorWithName) {
  const AtomicProposition ap1{"p"};
  const AtomicProposition ap2{"q"};

  EXPECT_FALSE(ap1.is_dummy());
  EXPECT_FALSE(ap2.is_dummy());
  EXPECT_NE(ap1.id(), ap2.id());
  EXPECT_EQ(ap1.name(), "p");
  EXPECT_EQ(ap2.name(), "q");
}

TEST(TestAtomicProposition, ConstructorWithId) {
  const AtomicProposition ap1{"p"};
  const AtomicProposition ap2{ap1.id()};

  EXPECT_EQ(ap1.id(), ap2.id());
  EXPECT_EQ(ap2.name(), "p");
}

TEST(TestAtomicProposition, EqualityOperator) {
  const AtomicProposition ap1{"p"};
  const AtomicProposition ap2{ap1.id()};
  const AtomicProposition ap3{"q"};

  EXPECT_TRUE(ap1.equal_to(ap2));
  EXPECT_FALSE(ap1.equal_to(ap3));
}

TEST(TestAtomicProposition, LessOperator) {
  const AtomicProposition ap1{"p"};
  const AtomicProposition ap2{"q"};

  EXPECT_TRUE(ap1.less(ap2));
  EXPECT_FALSE(ap2.less(ap1));
}

TEST(TestAtomicProposition, HashFunction) {
  const AtomicProposition ap1{"p"};
  const AtomicProposition ap2{ap1.id()};
  const AtomicProposition ap3{"q"};

  EXPECT_EQ(ap1.hash(), ap2.hash());
  EXPECT_NE(ap1.hash(), ap3.hash());
}

TEST(TestAtomicProposition, StreamOutput) {
  const AtomicProposition ap{"p"};
  EXPECT_EQ((std::ostringstream{} << ap).str(), "p");
}

TEST(TestAtomicProposition, MultipleDummyPropositions) {
  const AtomicProposition dummy1;
  const AtomicProposition dummy2;

  EXPECT_TRUE(dummy1.is_dummy());
  EXPECT_TRUE(dummy2.is_dummy());
  EXPECT_EQ(dummy1.id(), dummy2.id());
  EXPECT_EQ(dummy1.name(), "dummy");
}

TEST(TestAtomicProposition, LargeNumberOfPropositions) {
  constexpr size_t num_props = 1000;
  std::vector<AtomicProposition> props;

  for (size_t i = 0; i < num_props; ++i) {
    props.emplace_back("p" + std::to_string(i));
  }

  const std::size_t first_id = props[0].id();
  for (size_t i = 0; i < num_props; ++i) {
    EXPECT_EQ(props[i].name(), "p" + std::to_string(i));
    EXPECT_EQ(props[i].id(), first_id + i);
  }
}

TEST(TestAtomicProposition, InvalidIdThrows) { EXPECT_THROW(AtomicProposition(9999), std::runtime_error); }