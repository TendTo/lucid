#include <string>

#include "lucid/parser/Parser.h"

namespace lucid::stl {

/**
 * Parser for the
 * [Signal Temporal Logic (STL)](https://en.wikipedia.org/wiki/Signal_temporal_logic) specification language.
 * This parser uses the Boost.Spirit library to parse STL formulas from strings or files.
 * The parsed formulas will be represented in an internal data structure.
 * STL is an extension of [Metric Temporal Logic (MTL)](https://en.wikipedia.org/wiki/Metric_temporal_logic)
 * that allows for reasoning about real-valued signals over time.
 *
 * The STL syntax is defined as follows:
 * @f[
 * \varphi ::= \text{true} \mid \kappa
 * \mid \neg \varphi \mid \varphi_1 \land \varphi_2
 * \mid \varphi_1 U_I \varphi_2
 * @f]
 * where @f$ \kappa @f$ is an atomic predicate belonging to the set @f$ K @f$
 * and @f$ U_I @f$ is the "until" operator with a time interval @f$ I \subseteq \mathbb{R}^+ @f$.
 *
 * @note Other logical operators such as "or" (@f$ \lor @f$) and "implies" (@f$ \rightarrow @f$)
 * can be derived from the basic boolean operators "not" (@f$ \neg @f$) and "and" (@f$ \land @f$).
 *
 * The specific STL operators are defined as follows:
 * - __Until__ (@f$ U_I @f$): The formula @f$ \varphi_1 U_I \varphi_2 @f$ holds
 * if there exists a time point @f$ t' @f$ in the interval @f$ I @f$ such that
 * @f$ \varphi_2 @f$ holds at time @f$ t' @f$
 * and for all time points @f$ t'' < t' @f$, @f$ \varphi_1 @f$ holds.
 * - **Eventually** (@f$ F_I @f$): The formula @f$ F_I \varphi @f$ is defined as
 * @f$ \text{true} U_I \varphi @f$ and holds
 * if there exists a time point @f$ t' @f$ in the interval @f$ I @f$ such that
 * @f$ \varphi @f$ holds at time @f$ t' @f$.
 * - **Always** (@f$ G_I @f$): The formula @f$ G_I \varphi @f$ is defined as
 * @f$ \neg F_I \neg \varphi @f$ and holds
 * if for all time points @f$ t' @f$ in the interval @f$ I @f$,
 * @f$ \varphi @f$ holds at time @f$ t' @f$.
 *
 * @note If the time interval @f$ I @f$ is not specified, it is assumed to be @f$ [0, \infty) @f$.
 *
 * @code
 * // Example usage
 *  Parser parser;
 *  parser.parse_input("G[0,10] (x > 0)");
 * @endcode
 * @todo Provide the internal data structure.
 */
class Parser final : public lucid::Parser {
 private:
  using lucid::Parser::Parser;

  void parse_input_impl(const std::string& input) const override;
};

}  // namespace lucid::stl
