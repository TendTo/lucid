#include "lucid/parser/stl/Parser.h"

#include <boost/spirit/home/x3.hpp>
#include <iostream>
#include <istream>
#include <string>

#include "lucid/util/error.h"

namespace x3 = boost::spirit::x3;

using x3::char_;
using x3::double_;
using x3::eps;

namespace lucid::stl {

namespace atomic_ {

enum class Operator { None, And, Or, Implies, LessThan, GreaterThan, LessEqual, GreaterEqual, Equal, NotEqual };

namespace ast {
struct Atomic {
  Atomic() : term(""), op(Operator::None), value(0.0) {}
  std::string term;
  Operator op;
  double value;
};
}  // namespace ast

auto init = [](auto& ctx) { _val(ctx) = Atomic{.term = "", .op = Operator::None, .value = 0.0}; };
auto addTerm = [](auto& ctx) { _val(ctx).term += _attr(ctx); };
auto addOp = [](auto& ctx) { _val(ctx).op = _attr(ctx); };
auto addValue = [](auto& ctx) { _val(ctx).value = _attr(ctx); };

struct operators_ : x3::symbols<Operator> {
  operators_() {
    add(">", Operator::GreaterThan)     //
        ("<", Operator::LessThan)       //
        (">=", Operator::GreaterEqual)  //
        ("<=", Operator::LessEqual)     //
        ("==", Operator::Equal)         //
        ("!=", Operator::NotEqual)      //
        ("&&", Operator::And)           //
        ("||", Operator::Or)            //
        ("->", Operator::Implies)       //
        ;
  }
};

}  // namespace atomic_

void Parser::parse_input_impl(const std::string& input) const {
  std::vector<std::string> tokens;
  auto first = input.begin();
  auto last = input.end();

  const x3::rule<class atomic, atomic_::Atomic> atomic = "atomic";
  const auto atomic_def = eps[atomic_::init] >>                   //
                          char_[atomic_::addTerm] >>              //
                          atomic_::operators_[atomic_::addOp] >>  //
                          double_[atomic_::addValue];

  BOOST_FUSION_ADAPT_STRUCT(atomic_::ast::Atomic, term, op, value)

  BOOST_SPIRIT_DEFINE(atomic);

  atomic_::Atomic a;
  bool r = x3::parse(first, last,
                     //
                     atomic % (char_(',') | char_('_')),
                     //
                     a);

  if (first != last || !r)  // fail if we did not get a full match or an error occurred
    LUCID_PARSE_FAIL("STL", input, fmt::format("Parsing failed at: '{}'", std::string(first, last)));

  fmt::println("Parsed atomic: term='{}', op={}, value={}", a.term, static_cast<int>(a.op), a.value);
}

}  // namespace lucid::stl
