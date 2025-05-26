/**
 * @author c3054737
 * @copyright 2025 lucid
 * @licence BSD 3-Clause License
 */
#include <gtest/gtest.h>

#include "lucid/model/Parametrizable.h"
#include "lucid/util/exception.h"

using lucid::Parameter;
using lucid::Parametrizable;
using lucid::Vector;

class MockParametrizable final : public Parametrizable {
 public:
  using Parametrizable::set;
  MockParametrizable() : int_param_(42), double_param_(3.14), vector_param_(Vector::Ones(3)) {}

  [[nodiscard]] bool has(const Parameter parameter) const override {
    switch (parameter) {
      case Parameter::DEGREE:
      case Parameter::SIGMA_F:
      case Parameter::SIGMA_L:
        return true;
      default:
        return false;
    }
  }

  [[nodiscard]] int get_i(const Parameter parameter) const override {
    if (parameter == Parameter::DEGREE) {
      return int_param_;
    }
    return Parametrizable::get_i(parameter);  // Will throw for other parameters
  }

  [[nodiscard]] double get_d(const Parameter parameter) const override {
    if (parameter == Parameter::SIGMA_F) {
      return double_param_;
    }
    return Parametrizable::get_d(parameter);  // Will throw for other parameters
  }

  [[nodiscard]] const Vector& get_v(const Parameter parameter) const override {
    if (parameter == Parameter::SIGMA_L) {
      return vector_param_;
    }
    return Parametrizable::get_v(parameter);  // Will throw for other parameters
  }

  void set(const Parameter parameter, const int value) override {
    if (parameter == Parameter::DEGREE) {
      int_param_ = value;
    } else {
      Parametrizable::set(parameter, value);  // Will throw for other parameters
    }
  }

  void set(const Parameter parameter, const double value) override {
    if (parameter == Parameter::SIGMA_F) {
      double_param_ = value;
    } else {
      Parametrizable::set(parameter, value);  // Will throw for other parameters
    }
  }

  void set(const Parameter parameter, const Vector& value) override {
    if (parameter == Parameter::SIGMA_L) {
      vector_param_ = value;
    } else {
      Parametrizable::set(parameter, value);  // Will throw for other parameters
    }
  }

 private:
  int int_param_;
  double double_param_;
  Vector vector_param_;
};

class TestParametrizable : public ::testing::Test {
 protected:
  MockParametrizable parametrizable_;
};

TEST_F(TestParametrizable, HasParameter) {
  EXPECT_TRUE(parametrizable_.has(Parameter::DEGREE));
  EXPECT_TRUE(parametrizable_.has(Parameter::SIGMA_F));
  EXPECT_TRUE(parametrizable_.has(Parameter::SIGMA_L));
  EXPECT_FALSE(parametrizable_.has(Parameter::REGULARIZATION_CONSTANT));
}

TEST_F(TestParametrizable, GetIntParameter) { EXPECT_EQ(parametrizable_.get<Parameter::DEGREE>(), 42); }

TEST_F(TestParametrizable, GetDoubleParameter) { EXPECT_DOUBLE_EQ(parametrizable_.get<Parameter::SIGMA_F>(), 3.14); }

TEST_F(TestParametrizable, GetVectorParameter) {
  const Vector expected = Vector::Ones(3);
  EXPECT_EQ(parametrizable_.get<Parameter::SIGMA_L>(), expected);
}

TEST_F(TestParametrizable, SetIntParameter) {
  parametrizable_.set<Parameter::DEGREE>(100);
  EXPECT_EQ(parametrizable_.get<Parameter::DEGREE>(), 100);
}

TEST_F(TestParametrizable, SetDoubleParameter) {
  parametrizable_.set<Parameter::SIGMA_F>(2.71);
  EXPECT_DOUBLE_EQ(parametrizable_.get<Parameter::SIGMA_F>(), 2.71);
}

TEST_F(TestParametrizable, SetVectorParameter) {
  const Vector newValue = Vector::Zero(3);
  parametrizable_.set<Parameter::SIGMA_L>(newValue);
  EXPECT_EQ(parametrizable_.get<Parameter::SIGMA_L>(), newValue);
}

TEST_F(TestParametrizable, TemplatedGetInt) { EXPECT_EQ(parametrizable_.get<Parameter::DEGREE>(), 42); }

TEST_F(TestParametrizable, TemplatedGetDouble) { EXPECT_DOUBLE_EQ(parametrizable_.get<Parameter::SIGMA_F>(), 3.14); }

TEST_F(TestParametrizable, TemplatedGetVector) {
  const Vector expected = Vector::Ones(3);
  EXPECT_EQ(parametrizable_.get<Parameter::SIGMA_L>(), expected);
  EXPECT_EQ(&parametrizable_.get<const Vector&>(Parameter::SIGMA_L), &parametrizable_.get<Parameter::SIGMA_L>());
}

TEST_F(TestParametrizable, GetNonExistentParameter) {
  EXPECT_THROW(static_cast<void>(parametrizable_.get<Parameter::REGULARIZATION_CONSTANT>()),
               lucid::exception::LucidInvalidArgumentException);
}

TEST_F(TestParametrizable, GetWithIncorrectType) {
  EXPECT_THROW(static_cast<void>(parametrizable_.get<double>(Parameter::DEGREE)),
               lucid::exception::LucidInvalidArgumentException);
  EXPECT_THROW(static_cast<void>(parametrizable_.get<int>(Parameter::SIGMA_F)),
               lucid::exception::LucidInvalidArgumentException);
  EXPECT_THROW(static_cast<void>(parametrizable_.get<const Vector&>(Parameter::SIGMA_F)),
               lucid::exception::LucidInvalidArgumentException);
}

TEST_F(TestParametrizable, SetNonExistentParameter) {
  EXPECT_THROW(parametrizable_.set(Parameter::REGULARIZATION_CONSTANT, 0.5),
               lucid::exception::LucidInvalidArgumentException);
  EXPECT_THROW(parametrizable_.set<Parameter::REGULARIZATION_CONSTANT>(0.5),
               lucid::exception::LucidInvalidArgumentException);
}

TEST_F(TestParametrizable, SetWithIncorrectType) {
  EXPECT_THROW(parametrizable_.set(Parameter::DEGREE, 1.5), lucid::exception::LucidInvalidArgumentException);
  EXPECT_THROW(parametrizable_.set(Parameter::SIGMA_F, 5), lucid::exception::LucidInvalidArgumentException);

  Vector vec = Vector::Zero(3);
  EXPECT_THROW(parametrizable_.set(Parameter::DEGREE, vec), lucid::exception::LucidInvalidArgumentException);
}

TEST_F(TestParametrizable, SetWithVariant) {
  const std::variant<int, double, Vector> intVariant = 123;
  parametrizable_.set(Parameter::DEGREE, intVariant);
  EXPECT_EQ(parametrizable_.get<int>(Parameter::DEGREE), 123);

  std::variant<int, double, Vector> doubleVariant = 5.67;
  parametrizable_.set(Parameter::SIGMA_F, doubleVariant);
  EXPECT_DOUBLE_EQ(parametrizable_.get<double>(Parameter::SIGMA_F), 5.67);

  Vector vec = Vector::Random(3);
  std::variant<int, double, Vector> vectorVariant = vec;
  parametrizable_.set(Parameter::SIGMA_L, vectorVariant);
  EXPECT_EQ(parametrizable_.get<const Vector&>(Parameter::SIGMA_L), vec);
}

TEST_F(TestParametrizable, SetWithInvalidVariant) {
  std::variant<int, double, Vector> doubleVariant = 5.67;
  EXPECT_THROW(parametrizable_.set(Parameter::DEGREE, doubleVariant), std::bad_variant_access);
}

TEST_F(TestParametrizable, SetWithIndexBasedValues) {
  const std::variant<std::vector<int>, std::vector<double>, std::vector<Vector>> intVariant{
      std::vector<int>{10, 20, 30}};

  parametrizable_.set(Parameter::DEGREE, 1, intVariant);
  EXPECT_EQ(parametrizable_.get<int>(Parameter::DEGREE), 20);

  const std::variant<std::vector<int>, std::vector<double>, std::vector<Vector>> doubleVariant{
      std::vector<double>{1.1, 2.2, 3.3}};

  parametrizable_.set(Parameter::SIGMA_F, 2, doubleVariant);
  EXPECT_DOUBLE_EQ(parametrizable_.get<double>(Parameter::SIGMA_F), 3.3);

  const std::variant<std::vector<int>, std::vector<double>, std::vector<Vector>> vectorVariant{
      std::vector<Vector>{Vector::Zero(3), Vector::Ones(3), 2 * Vector::Ones(3)}};

  parametrizable_.set(Parameter::SIGMA_L, 1, vectorVariant);
  EXPECT_EQ(parametrizable_.get<const Vector&>(Parameter::SIGMA_L), Vector::Ones(3));
}

TEST_F(TestParametrizable, SetWithInvalidIndex) {
  const std::variant<std::vector<int>, std::vector<double>, std::vector<Vector>> intVariant{
      std::vector<int>{10, 20, 30}};

  EXPECT_THROW(parametrizable_.set(Parameter::DEGREE, 5, intVariant), std::out_of_range);
}

TEST_F(TestParametrizable, SetWithIncorrectVariantType) {
  const std::variant<std::vector<int>, std::vector<double>, std::vector<Vector>> doubleVariant{
      std::vector<double>{1.1, 2.2, 3.3}};

  // Trying to set an int parameter with double values
  EXPECT_THROW(parametrizable_.set(Parameter::DEGREE, 1, doubleVariant), std::bad_variant_access);
}

TEST_F(TestParametrizable, SetEmptyVector) {
  const Vector emptyVec(0);
  parametrizable_.set(Parameter::SIGMA_L, emptyVec);
  EXPECT_EQ(parametrizable_.get<Parameter::SIGMA_L>().size(), 0);
}

TEST_F(TestParametrizable, SetLargeVector) {
  constexpr int largeSize = 1000;
  const Vector largeVec = Vector::Random(largeSize);
  parametrizable_.set(Parameter::SIGMA_L, largeVec);
  EXPECT_EQ(parametrizable_.get<Parameter::SIGMA_L>().size(), largeSize);
  EXPECT_EQ(parametrizable_.get<Parameter::SIGMA_L>(), largeVec);
}

TEST_F(TestParametrizable, SetExtremeValues) {
  // Test with extreme integer values
  parametrizable_.set<Parameter::DEGREE>(std::numeric_limits<int>::max());
  EXPECT_EQ(parametrizable_.get<int>(Parameter::DEGREE), std::numeric_limits<int>::max());

  parametrizable_.set<Parameter::DEGREE>(std::numeric_limits<int>::min());
  EXPECT_EQ(parametrizable_.get<int>(Parameter::DEGREE), std::numeric_limits<int>::min());

  // Test with extreme double values
  parametrizable_.set<Parameter::SIGMA_F>(std::numeric_limits<double>::max());
  EXPECT_DOUBLE_EQ(parametrizable_.get<double>(Parameter::SIGMA_F), std::numeric_limits<double>::max());

  parametrizable_.set<Parameter::SIGMA_F>(std::numeric_limits<double>::min());
  EXPECT_DOUBLE_EQ(parametrizable_.get<double>(Parameter::SIGMA_F), std::numeric_limits<double>::min());

  parametrizable_.set<Parameter::SIGMA_F>(std::numeric_limits<double>::infinity());
  EXPECT_TRUE(std::isinf(parametrizable_.get<double>(Parameter::SIGMA_F)));

  parametrizable_.set<Parameter::SIGMA_F>(std::numeric_limits<double>::quiet_NaN());
  EXPECT_TRUE(std::isnan(parametrizable_.get<double>(Parameter::SIGMA_F)));
}