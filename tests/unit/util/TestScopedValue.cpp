/**
 * @author c3054737
 * @copyright 2025 lucid
 * @licence BSD 3-Clause License
 * @file
 * TestScopedValue class.
 */
#include <gmock/gmock.h>
#include <gtest/gtest.h>

#include <functional>
#include <memory>
#include <stdexcept>
#include <string>
#include <thread>

#include "lucid/util/ScopedValue.h"

using lucid::BaseScopedValue;
using lucid::PolymorphicScopedValue;
using lucid::ScopedValue;
using lucid::ScopedValueShield;

using ScopedValueI = ScopedValue<int>;
using ScopedValueS = ScopedValue<std::string>;
using ScopedValueD = ScopedValue<double>;

// Tag types for testing different scoped value stacks
struct TagA {};
struct TagB {};
struct TagC {};

using ScopedValueITagA = ScopedValue<int, TagA>;
using ScopedValueITagB = ScopedValue<int, TagB>;
using ScopedValueITagAB = ScopedValue<int, TagA, TagB>;

// Test fixture for consistent setup/teardown
class TestScopedValue : public ::testing::Test {
 protected:
  void SetUp() override {
    // Ensure no leftover scoped values from previous tests
    EXPECT_FALSE(ScopedValueI::top());
    EXPECT_FALSE(ScopedValueS::top());
    EXPECT_FALSE(ScopedValueD::top());
    EXPECT_FALSE(ScopedValueITagA::top());
    EXPECT_FALSE(ScopedValueITagB::top());
  }

  void TearDown() override {
    // Verify cleanup after each test
    EXPECT_FALSE(ScopedValueI::top());
    EXPECT_FALSE(ScopedValueS::top());
    EXPECT_FALSE(ScopedValueD::top());
    EXPECT_FALSE(ScopedValueITagA::top());
    EXPECT_FALSE(ScopedValueITagB::top());
  }
};

// Mock class for testing polymorphic behavior
class MockBase {
 public:
  virtual ~MockBase() = default;
  virtual int getValue() const = 0;
  virtual std::string getName() const = 0;
};

class MockDerived : public MockBase {
 public:
  explicit MockDerived(int value, std::string name) : value_(value), name_(std::move(name)) {}
  int getValue() const override { return value_; }
  std::string getName() const override { return name_; }

 private:
  int value_;
  std::string name_;
};

using PolymorphicScopedMock = PolymorphicScopedValue<MockDerived, MockBase>;

TEST_F(TestScopedValue, DefaultConstructor) {
  const ScopedValueI scope{};
  EXPECT_TRUE(ScopedValueI::top());
  EXPECT_TRUE(ScopedValueI::bottom());
  EXPECT_EQ(scope.value(), int{});
  EXPECT_EQ(ScopedValueI::top(), &scope);
  EXPECT_EQ(ScopedValueI::bottom(), &scope);
  EXPECT_EQ(ScopedValueI::scope_stack().size(), 1);
}

TEST_F(TestScopedValue, ValueConstructor) {
  constexpr int expected_value = 42;
  const ScopedValueI scope{expected_value};
  EXPECT_TRUE(ScopedValueI::top());
  EXPECT_TRUE(ScopedValueI::bottom());
  EXPECT_EQ(scope.value(), expected_value);
  EXPECT_EQ(ScopedValueI::top(), &scope);
  EXPECT_EQ(ScopedValueI::bottom(), &scope);
  EXPECT_EQ(ScopedValueI::scope_stack().size(), 1);
}

TEST_F(TestScopedValue, CopyConstructor) {
  constexpr int expected_value = 42;
  const ScopedValueI scope{expected_value};
  const ScopedValueI scope_copy{scope};
  EXPECT_TRUE(ScopedValueI::top());
  EXPECT_TRUE(ScopedValueI::bottom());
  EXPECT_EQ(ScopedValueI::top(), &scope_copy);
  EXPECT_EQ(ScopedValueI::bottom(), &scope);
  EXPECT_EQ(scope_copy.value(), expected_value);
  EXPECT_EQ(scope.value(), expected_value);
  EXPECT_EQ(ScopedValueI::scope_stack().size(), 2);
}

TEST_F(TestScopedValue, CopyAssignment) {
  constexpr int expected_value = 42;
  const ScopedValueI scope{expected_value};
  ScopedValueI scope_copy{};
  scope_copy = scope;
  EXPECT_TRUE(ScopedValueI::top());
  EXPECT_TRUE(ScopedValueI::bottom());
  EXPECT_EQ(ScopedValueI::top(), &scope_copy);
  EXPECT_EQ(ScopedValueI::bottom(), &scope);
  EXPECT_EQ(scope_copy.value(), expected_value);
  EXPECT_EQ(scope.value(), expected_value);
  EXPECT_EQ(ScopedValueI::scope_stack().size(), 2);
}

TEST_F(TestScopedValue, MultipleNestesdScopes) {
  EXPECT_FALSE(ScopedValueI::top());
  {
    constexpr int value_scope = 1;
    ScopedValueI scope_1(value_scope);
    EXPECT_TRUE(ScopedValueI::top());
    EXPECT_EQ(**ScopedValueI::top(), value_scope);
    {
      ScopedValueI scope_2(value_scope + 1);
      EXPECT_TRUE(ScopedValueI::top());
      EXPECT_EQ(**ScopedValueI::top(), value_scope + 1);
    }
    EXPECT_TRUE(ScopedValueI::top());
    EXPECT_EQ(**ScopedValueI::top(), value_scope);
    {
      ScopedValueI scope_3(value_scope + 2);
      EXPECT_TRUE(ScopedValueI::top());
      EXPECT_EQ(**ScopedValueI::top(), value_scope + 2);
    }
    EXPECT_TRUE(ScopedValueI::top());
    EXPECT_EQ(**ScopedValueI::top(), value_scope);
  }
  EXPECT_FALSE(ScopedValueI::top());
}

TEST_F(TestScopedValue, ComplexTypes) {
  const std::vector<int> vec = {1, 2, 3};
  {
    ScopedValue<std::vector<int>> scoped_vec(vec);
    std::vector<int> value = **ScopedValue<std::vector<int>>::top();
    EXPECT_THAT(vec, ::testing::ElementsAre(1, 2, 3));
    value.push_back(4);
    EXPECT_THAT(value, ::testing::ElementsAre(1, 2, 3, 4));
  }
  EXPECT_THAT(vec, ::testing::ElementsAre(1, 2, 3));  // Changes have not persisted
}

TEST_F(TestScopedValue, Swap) {
  [[maybe_unused]] const auto& a = ScopedValueI::scope_stack();
  constexpr int value1 = 10;
  constexpr int value2 = 20;
  ScopedValueI scoped1(value1);
  ScopedValueI scoped2(value2);

  EXPECT_EQ(**ScopedValueI::bottom(), value1);
  EXPECT_EQ(**ScopedValueI::top(), value2);

  scoped1.swap(scoped2);

  EXPECT_EQ(**ScopedValueI::bottom(), value2);
  EXPECT_EQ(**ScopedValueI::top(), value1);
}

TEST_F(TestScopedValue, DereferenceOperators) {
  constexpr int expected_value = 42;
  ScopedValueI scope(expected_value);

  // Test operator*
  EXPECT_EQ(*scope, expected_value);
  EXPECT_EQ(**ScopedValueI::top(), expected_value);

  // Test operator->
  auto* ptr = scope.operator->();
  EXPECT_EQ(*ptr, expected_value);

  // Test const versions
  const auto& const_scope = scope;
  EXPECT_EQ(*const_scope, expected_value);
  auto* const_ptr = const_scope.operator->();
  EXPECT_EQ(*const_ptr, expected_value);
}

TEST_F(TestScopedValue, ValueModification) {
  ScopedValueI scope(10);
  EXPECT_EQ(scope.value(), 10);

  scope.value() = 20;
  EXPECT_EQ(scope.value(), 20);
  EXPECT_EQ(**ScopedValueI::top(), 20);

  *scope = 30;
  EXPECT_EQ(*scope, 30);
}

TEST_F(TestScopedValue, RandomOrderDestruction) {
  [[maybe_unused]] const auto& a = ScopedValueI::scope_stack();
  std::vector<ScopedValueI> scopes;
  constexpr int num_scopes = 5;

  // Create scopes
  scopes.reserve(num_scopes);
  for (int i = 0; i < num_scopes; ++i) {
    scopes.emplace_back(i);
  }

  EXPECT_EQ(ScopedValueI::scope_stack().size(), num_scopes);

  // Remove middle scope
  scopes.erase(scopes.begin() + 2);  // Remove scope with value 2
  EXPECT_EQ(ScopedValueI::scope_stack().size(), num_scopes - 1);

  // Verify remaining scopes are still valid
  const std::vector<int> expected_values = {0, 1, 3, 4};
  for (size_t i = 0; i < expected_values.size(); ++i) {
    EXPECT_EQ(ScopedValueI::scope_stack()[i]->value(), expected_values[i]);
  }
}

TEST_F(TestScopedValue, DifferentTypesIndependentStacks) {
  ScopedValueI int_scope(42);
  ScopedValueS string_scope("hello");
  ScopedValueD double_scope(3.14);

  EXPECT_EQ(ScopedValueI::scope_stack().size(), 1);
  EXPECT_EQ(ScopedValueS::scope_stack().size(), 1);
  EXPECT_EQ(ScopedValueD::scope_stack().size(), 1);

  EXPECT_EQ(**ScopedValueI::top(), 42);
  EXPECT_EQ(**ScopedValueS::top(), "hello");
  EXPECT_DOUBLE_EQ(**ScopedValueD::top(), 3.14);
}

TEST_F(TestScopedValue, TaggedTypesIndependentStacks) {
  ScopedValueITagA tagged_a(100);
  ScopedValueITagB tagged_b(200);
  ScopedValueITagAB tagged_ab(300);

  EXPECT_EQ(ScopedValueITagA::scope_stack().size(), 1);
  EXPECT_EQ(ScopedValueITagB::scope_stack().size(), 1);
  EXPECT_EQ(ScopedValueITagAB::scope_stack().size(), 1);

  EXPECT_EQ(**ScopedValueITagA::top(), 100);
  EXPECT_EQ(**ScopedValueITagB::top(), 200);
  EXPECT_EQ(**ScopedValueITagAB::top(), 300);

  // Regular ScopedValueI should still be empty
  EXPECT_FALSE(ScopedValueI::top());
}

TEST_F(TestScopedValue, StringOperations) {
  ScopedValueS scope("initial");
  EXPECT_EQ(scope.value(), "initial");

  scope->append(" value");
  EXPECT_EQ(*scope, "initial value");

  *scope = "modified";
  EXPECT_EQ(scope.value(), "modified");
}

TEST_F(TestScopedValue, VectorOperations) {
  std::vector<int> initial_vec = {1, 2, 3};
  ScopedValue<std::vector<int>> scope(initial_vec);

  EXPECT_THAT(*scope, ::testing::ElementsAre(1, 2, 3));

  scope->push_back(4);
  EXPECT_THAT(*scope, ::testing::ElementsAre(1, 2, 3, 4));

  scope->clear();
  EXPECT_TRUE(scope->empty());

  // Original vector unchanged
  EXPECT_THAT(initial_vec, ::testing::ElementsAre(1, 2, 3));
}

TEST_F(TestScopedValue, UniquePointerOperations) {
  auto unique_ptr = std::make_unique<int>(42);
  int* raw_ptr = unique_ptr.get();

  ScopedValue<std::unique_ptr<int>> scope(std::move(unique_ptr));

  EXPECT_EQ(scope->get(), raw_ptr);
  EXPECT_EQ(**scope, 42);

  **scope = 100;
  EXPECT_EQ(**scope, 100);
}

TEST_F(TestScopedValue, SwapPreservesStackOrder) {
  ScopedValueI scope1(10);
  ScopedValueI scope2(20);
  ScopedValueI scope3(30);

  EXPECT_EQ(**ScopedValueI::bottom(), 10);
  EXPECT_EQ(**ScopedValueI::scope_stack()[1], 20);
  EXPECT_EQ(**ScopedValueI::top(), 30);

  scope1.swap(scope3);

  // Stack order preserved, values swapped
  EXPECT_EQ(**ScopedValueI::bottom(), 30);
  EXPECT_EQ(**ScopedValueI::scope_stack()[1], 20);
  EXPECT_EQ(**ScopedValueI::top(), 10);
}

TEST_F(TestScopedValue, SwapComplexTypes) {
  std::vector<int> vec1 = {1, 2, 3};
  std::vector<int> vec2 = {4, 5, 6, 7};

  ScopedValue<std::vector<int>> scope1(vec1);
  ScopedValue<std::vector<int>> scope2(vec2);

  scope1.swap(scope2);

  EXPECT_THAT(*scope1, ::testing::ElementsAre(4, 5, 6, 7));
  EXPECT_THAT(*scope2, ::testing::ElementsAre(1, 2, 3));
}

TEST_F(TestScopedValue, PolymorphicBehavior) {
  PolymorphicScopedMock polymorphic_scope(42, "test");

  EXPECT_EQ(polymorphic_scope.value().getValue(), 42);
  EXPECT_EQ(polymorphic_scope.value().getName(), "test");

  // Test polymorphic access
  const MockBase& base_ref = polymorphic_scope.value();
  EXPECT_EQ(base_ref.getValue(), 42);
  EXPECT_EQ(base_ref.getName(), "test");
}

TEST_F(TestScopedValue, EmptyStackAccess) {
  EXPECT_FALSE(ScopedValueI::top());
  EXPECT_FALSE(ScopedValueI::bottom());
  EXPECT_TRUE(ScopedValueI::scope_stack().empty());
}

TEST_F(TestScopedValue, ZeroValueHandling) {
  ScopedValueI zero_scope(0);
  EXPECT_EQ(*zero_scope, 0);
  EXPECT_TRUE(ScopedValueI::top());  // Should still exist even with zero value
}

TEST_F(TestScopedValue, NegativeValueHandling) {
  ScopedValueI negative_scope(-42);
  EXPECT_EQ(*negative_scope, -42);
  EXPECT_EQ(**ScopedValueI::top(), -42);
}

TEST_F(TestScopedValue, ShieldBasicFunctionality) {
  ScopedValueI outer_scope(10);
  EXPECT_EQ(ScopedValueI::scope_stack().size(), 1);
  EXPECT_EQ(**ScopedValueI::top(), 10);

  {
    ScopedValueShield<int> shield;

    // Stack should be empty while shield is active
    EXPECT_FALSE(ScopedValueI::top());
    EXPECT_FALSE(ScopedValueI::bottom());
    EXPECT_TRUE(ScopedValueI::scope_stack().empty());

    // Can create new scoped values while shielded
    ScopedValueI inner_scope(20);
    EXPECT_EQ(**ScopedValueI::top(), 20);
    EXPECT_EQ(ScopedValueI::scope_stack().size(), 1);
  }

  // Original stack should be restored
  EXPECT_EQ(ScopedValueI::scope_stack().size(), 1);
  EXPECT_EQ(**ScopedValueI::top(), 10);
  EXPECT_EQ(ScopedValueI::top(), &outer_scope);
}

TEST_F(TestScopedValue, ShieldWithMultipleScopes) {
  ScopedValueI scope1(1);
  ScopedValueI scope2(2);
  ScopedValueI scope3(3);

  EXPECT_EQ(ScopedValueI::scope_stack().size(), 3);
  EXPECT_EQ(**ScopedValueI::top(), 3);
  EXPECT_EQ(**ScopedValueI::bottom(), 1);

  {
    ScopedValueShield<int> shield;
    EXPECT_TRUE(ScopedValueI::scope_stack().empty());

    ScopedValueI shielded_scope(100);
    EXPECT_EQ(**ScopedValueI::top(), 100);
    EXPECT_EQ(ScopedValueI::scope_stack().size(), 1);
  }

  // All original scopes restored
  EXPECT_EQ(ScopedValueI::scope_stack().size(), 3);
  EXPECT_EQ(**ScopedValueI::top(), 3);
  EXPECT_EQ(**ScopedValueI::bottom(), 1);
}

TEST_F(TestScopedValue, NestedShields) {
  ScopedValueI original_scope(42);

  {
    ScopedValueShield<int> outer_shield;
    EXPECT_TRUE(ScopedValueI::scope_stack().empty());

    ScopedValueI first_shielded(10);
    EXPECT_EQ(**ScopedValueI::top(), 10);

    {
      ScopedValueShield<int> inner_shield;
      EXPECT_TRUE(ScopedValueI::scope_stack().empty());

      ScopedValueI second_shielded(20);
      EXPECT_EQ(**ScopedValueI::top(), 20);
    }

    // First shield restored
    EXPECT_EQ(**ScopedValueI::top(), 10);
    EXPECT_EQ(ScopedValueI::scope_stack().size(), 1);
  }

  // Original scope restored
  EXPECT_EQ(**ScopedValueI::top(), 42);
  EXPECT_EQ(ScopedValueI::scope_stack().size(), 1);
}

TEST_F(TestScopedValue, ShieldWithTaggedTypes) {
  ScopedValueITagA tagged_scope(100);
  ScopedValueI regular_scope(200);

  {
    ScopedValueShield<int, TagA> tagged_shield;

    // Only TagA stack should be shielded
    EXPECT_FALSE(ScopedValueITagA::top());
    EXPECT_TRUE(ScopedValueI::top());
    EXPECT_EQ(**ScopedValueI::top(), 200);
  }

  // TagA stack restored, regular stack unchanged
  EXPECT_EQ(**ScopedValueITagA::top(), 100);
  EXPECT_EQ(**ScopedValueI::top(), 200);
}

TEST_F(TestScopedValue, ShieldEmptyStack) {
  EXPECT_TRUE(ScopedValueI::scope_stack().empty());

  {
    ScopedValueShield<int> shield;
    EXPECT_TRUE(ScopedValueI::scope_stack().empty());

    ScopedValueI scope(42);
    EXPECT_EQ(**ScopedValueI::top(), 42);
  }

  // Should return to empty state
  EXPECT_TRUE(ScopedValueI::scope_stack().empty());
  EXPECT_FALSE(ScopedValueI::top());
}

TEST_F(TestScopedValue, ThreadLocalBehavior) {
  ScopedValueI main_thread_scope(100);
  EXPECT_EQ(**ScopedValueI::top(), 100);

  std::thread other_thread([&]() {
    // Other thread should have empty stack
    EXPECT_FALSE(ScopedValueI::top());
    EXPECT_TRUE(ScopedValueI::scope_stack().empty());

    ScopedValueI thread_scope(200);
    EXPECT_EQ(**ScopedValueI::top(), 200);
    EXPECT_EQ(ScopedValueI::scope_stack().size(), 1);
  });

  other_thread.join();

  // Main thread stack unchanged
  EXPECT_EQ(**ScopedValueI::top(), 100);
  EXPECT_EQ(ScopedValueI::scope_stack().size(), 1);
}

TEST_F(TestScopedValue, ThreadLocalCopyCurrent) {
  ScopedValueI main_thread_scope(100);
  EXPECT_EQ(**ScopedValueI::top(), 100);

  std::thread other_thread(
      [&](const std::vector<BaseScopedValue<int>*>& parent_scope_stack) {
        ScopedValueI::set_scopes(parent_scope_stack);
        // Other thread should have empty stack
        EXPECT_TRUE(ScopedValueI::top());
        EXPECT_EQ(**ScopedValueI::top(), 100);
        EXPECT_EQ(ScopedValueI::scope_stack().size(), 1);

        ScopedValueI thread_scope(200);
        EXPECT_EQ(**ScopedValueI::top(), 200);
        EXPECT_EQ(ScopedValueI::scope_stack().size(), 2);
      },
      ScopedValueI::scope_stack());

  // Main thread stack unchanged both before and after join
  EXPECT_EQ(**ScopedValueI::top(), 100);
  EXPECT_EQ(ScopedValueI::scope_stack().size(), 1);

  other_thread.join();

  EXPECT_EQ(**ScopedValueI::top(), 100);
  EXPECT_EQ(ScopedValueI::scope_stack().size(), 1);
}

TEST_F(TestScopedValue, ComplexScenarioIntegration) {
  // Simulate a complex nested scenario with multiple operations
  ScopedValueI config_scope(1);
  ScopedValueS name_scope("test");

  EXPECT_EQ(**ScopedValueI::top(), 1);
  EXPECT_EQ(**ScopedValueS::top(), "test");

  {
    ScopedValueShield<int> shield;

    {
      ScopedValueI temp_config(2);
      EXPECT_EQ(**ScopedValueI::top(), 2);

      {
        ScopedValueI nested_config(3);
        EXPECT_EQ(**ScopedValueI::top(), 3);
        EXPECT_EQ(ScopedValueI::scope_stack().size(), 2);

        // String scope should be unaffected
        EXPECT_EQ(**ScopedValueS::top(), "test");
      }

      EXPECT_EQ(**ScopedValueI::top(), 2);
    }

    EXPECT_TRUE(ScopedValueI::scope_stack().empty());
  }

  // Everything restored
  EXPECT_EQ(**ScopedValueI::top(), 1);
  EXPECT_EQ(**ScopedValueS::top(), "test");
}

TEST_F(TestScopedValue, ConstructorArgumentForwarding) {
  // Test perfect forwarding in constructors
  ScopedValue<std::pair<int, std::string>> pair_scope(42, "hello");
  EXPECT_EQ(pair_scope->first, 42);
  EXPECT_EQ(pair_scope->second, "hello");

  // Test with multiple arguments
  ScopedValue<std::tuple<int, double, std::string>> tuple_scope(1, 2.5, "test");
  EXPECT_EQ(std::get<0>(*tuple_scope), 1);
  EXPECT_DOUBLE_EQ(std::get<1>(*tuple_scope), 2.5);
  EXPECT_EQ(std::get<2>(*tuple_scope), "test");
}

TEST_F(TestScopedValue, ConstCorrectness) {
  const ScopedValueI const_scope(42);

  // const methods should work
  EXPECT_EQ(const_scope.value(), 42);
  EXPECT_EQ(*const_scope, 42);

  // Should be able to get const pointer
  const int* const_ptr = const_scope.operator->();
  EXPECT_EQ(*const_ptr, 42);
}

TEST_F(TestScopedValue, ProperCleanupOnException) {
  ScopedValueI outer_scope(1);

  try {
    ScopedValueI inner_scope(2);
    EXPECT_EQ(ScopedValueI::scope_stack().size(), 2);

    // Simulate exception
    throw std::runtime_error("test exception");
  } catch (const std::exception&) {
    // inner_scope should be properly cleaned up
    EXPECT_EQ(ScopedValueI::scope_stack().size(), 1);
    EXPECT_EQ(**ScopedValueI::top(), 1);
  }
}
