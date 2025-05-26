/**
 * @author c3054737
 * @copyright 2025 lucid
 * @licence BSD 3-Clause License
 */
#include <gmock/gmock.h>
#include <gtest/gtest.h>

#include "lucid/model/Parametrizable.h"

using lucid::dispatch;
using lucid::Parameter;
using lucid::Vector;
using lucid::internal::ParameterType;

TEST(TestParameter, Type) {
  static_assert(std::is_same_v<ParameterType<Parameter::DEGREE>::type, int>, "DEGREE is an int");
  static_assert(std::is_same_v<ParameterType<Parameter::SIGMA_F>::type, double>, "SIGMA_F is a double");
  static_assert(std::is_same_v<ParameterType<Parameter::SIGMA_L>::type, Vector>, "SIGMA_L is a Vector");
  static_assert(std::is_same_v<ParameterType<Parameter::REGULARIZATION_CONSTANT>::type, double>,
                "REGULARIZATION_CONSTANT is a double");
}

TEST(TestParameter, TypeRef) {
  static_assert(std::is_same_v<ParameterType<Parameter::DEGREE>::ref_type, int>, "DEGREE is an int");
  static_assert(std::is_same_v<ParameterType<Parameter::SIGMA_F>::ref_type, double>, "SIGMA_F is a double");
  static_assert(std::is_same_v<ParameterType<Parameter::SIGMA_L>::ref_type, const Vector&>, "SIGMA_L is a Vector");
  static_assert(std::is_same_v<ParameterType<Parameter::REGULARIZATION_CONSTANT>::ref_type, double>,
                "REGULARIZATION_CONSTANT is a double");
}

TEST(TestParameter, DispatchInt) {
  testing::MockFunction<void()> fun_int, fun_double, fun_vec;
  EXPECT_CALL(fun_int, Call()).Times(1);
  EXPECT_CALL(fun_double, Call()).Times(0);
  EXPECT_CALL(fun_vec, Call()).Times(0);
  dispatch<void>(Parameter::DEGREE, [&]() { fun_int.Call(); }, [&]() { fun_double.Call(); }, [&]() { fun_vec.Call(); });
}
TEST(TestParameter, DispatchIntTemplate) {
  testing::MockFunction<void()> fun_int, fun_double, fun_vec;
  EXPECT_CALL(fun_int, Call()).Times(1);
  EXPECT_CALL(fun_double, Call()).Times(0);
  EXPECT_CALL(fun_vec, Call()).Times(0);
  dispatch<void, Parameter::DEGREE>([&]() { fun_int.Call(); }, [&]() { fun_double.Call(); }, [&]() { fun_vec.Call(); });
}

TEST(TestParameter, DispatchDouble) {
  testing::MockFunction<void()> fun_int, fun_double, fun_vec;
  EXPECT_CALL(fun_int, Call()).Times(0);
  EXPECT_CALL(fun_double, Call()).Times(1);
  EXPECT_CALL(fun_vec, Call()).Times(0);
  dispatch<void>(
      Parameter::SIGMA_F, [&]() { fun_int.Call(); }, [&]() { fun_double.Call(); }, [&]() { fun_vec.Call(); });
}
TEST(TestParameter, DispatchDoubleTemplate) {
  testing::MockFunction<void()> fun_int, fun_double, fun_vec;
  EXPECT_CALL(fun_int, Call()).Times(0);
  EXPECT_CALL(fun_double, Call()).Times(1);
  EXPECT_CALL(fun_vec, Call()).Times(0);
  dispatch<void, Parameter::SIGMA_F>([&]() { fun_int.Call(); }, [&]() { fun_double.Call(); },
                                     [&]() { fun_vec.Call(); });
}

TEST(TestParameter, DispatchVector) {
  testing::MockFunction<void()> fun_int, fun_double, fun_vec;
  EXPECT_CALL(fun_int, Call()).Times(0);
  EXPECT_CALL(fun_double, Call()).Times(0);
  EXPECT_CALL(fun_vec, Call()).Times(1);
  dispatch<void>(
      Parameter::SIGMA_L, [&]() { fun_int.Call(); }, [&]() { fun_double.Call(); }, [&]() { fun_vec.Call(); });
}
TEST(TestParameter, DispatchVectorTemplate) {
  testing::MockFunction<void()> fun_int, fun_double, fun_vec;
  EXPECT_CALL(fun_int, Call()).Times(0);
  EXPECT_CALL(fun_double, Call()).Times(0);
  EXPECT_CALL(fun_vec, Call()).Times(1);
  dispatch<void, Parameter::SIGMA_L>([&]() { fun_int.Call(); }, [&]() { fun_double.Call(); },
                                     [&]() { fun_vec.Call(); });
}