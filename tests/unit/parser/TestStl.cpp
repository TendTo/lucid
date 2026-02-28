/**
 * @author Ernesto Casablanca
 * @author Oliver Schön
 * @copyright 2025 lucid
 * @licence BSD 3-Clause License
 */
#include <gtest/gtest.h>

#include "lucid/parser/stl/Parser.h"

using lucid::stl::Parser;

TEST(TestStl, Preliminary) {
  const Parser parser;
  parser.parse_input("1.0, 2.5, 3.14, 4");
  parser.parse_input("1.0, 2.5_3.14_4");
}

TEST(TestStl, Parsing) {
  const Parser parser;
  parser.parse_input("(x > 10) && F[0, 2] y > 2 || G[1, 6] z > 8");
  parser.parse_input("G[2,4] F[1,3](x>=3)");
  parser.parse_input("(x <= 10) && F[0, 2] y > 2 && G[1, 6] (z < 8) && G[1,6] (z > 3)");
}

TEST(TestStl, True) {
  const Parser parser;
  parser.parse_input("true");
}

TEST(TestStl, False) {
  const Parser parser;
  parser.parse_input("false");
}

TEST(TestStl, Atomic) {
  const Parser parser;
  parser.parse_input("x > 10");
  parser.parse_input("y < 5");
  parser.parse_input("z == 3");
  parser.parse_input("p >= 0");
  parser.parse_input("q <= 100");
  parser.parse_input("r != 7");
}

TEST(TestStl, NotAtomic) {
  const Parser parser;
  parser.parse_input("!(x > 10)");
}

TEST(TestStl, AndAtomic) {
  const Parser parser;
  parser.parse_input("(x > 10) && (y < 5)");
}

TEST(TestStl, OrAtomic) {
  const Parser parser;
  parser.parse_input("(x > 10) || (y < 5)");
}

TEST(TestStl, ImpliesAtomic) {
  const Parser parser;
  parser.parse_input("(x > 10) -> (y < 5)");
}

TEST(TestStl, Until) {
  const Parser parser;
  parser.parse_input("(x > 10) U[1, 5] (y < 5)");
  parser.parse_input("(x > 10) U (y < 5)");
}

TEST(TestStl, Eventually) {
  const Parser parser;
  parser.parse_input("F[1, 5] (x > 10)");
  parser.parse_input("F (x > 10)");
}

TEST(TestStl, Always) {
  const Parser parser;
  parser.parse_input("G[1, 5] (x > 10)");
  parser.parse_input("G (x > 10)");
}

TEST(TestStl, Nested) {
  const Parser parser;
  parser.parse_input("G[1, 5] (F[0, 2] (x > 10) && G[1, 6] (y < 5))");
}

TEST(TestStl, Invalid) {
  const Parser parser;
  parser.parse_input("(x > 10) && F[0, 2] y > 2 || G[1, 6] z > 8");  // Missing parentheses around the entire formula
  parser.parse_input("G[2,4] F[1,3](x>=3)");                         // Missing parentheses around the entire formula
  parser.parse_input("(x <= 10) && F[0, 2] y > 2 && G[1, 6] (z < 8) && G[1,6] (z > 3)");  // Missing parentheses around
                                                                                          // the entire formula
  parser.parse_input("G[2,4] F[1,3](x>=3");                                               // Missing closing parenthesis
  parser.parse_input("G[2,4] F[1,3]x>=3)");                                               // Missing opening parenthesis
  parser.parse_input("G[2,4] F[1,3](x>=3))");
}