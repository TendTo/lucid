from abc import ABC, abstractmethod
from dataclasses import dataclass
from functools import partial
from typing import Callable, Generic, TypeVar, overload

import numpy as np
import pyparsing as pp
from ._pylucid import log_trace, log_warn
import sympy as sp

T = TypeVar("T")


try:
    import dreal
except ImportError as e:
    log_warn("Could not import dreal. Make sure it is installed with 'pip install dreal'")

try:
    from z3 import z3
except ImportError as e:
    log_warn("Could not import z3. Make sure it is installed with 'pip install z3-solver'")


@dataclass(frozen=True)
class Operation:
    associativity: pp.opAssoc
    num_operands: int
    ast_action: "Callable"


@dataclass(frozen=True)
class Function:
    ast_action: "Callable"


class SymbolicParser(ABC, Generic[T]):
    """Base class for parsing a symbolic expression.
    It uses pyparsing to parse the expression and convert it into an abstract syntax tree (AST).
    The tree is then traversed and converted into a symbolic expression using the provided operations and functions.

    Args:
        un_ops: dictionary of unary operations with their names and actions
        bin_ops: dictionary of binary operations with their names and actions
        funcs: dictionary of functions with their names and actions
    """

    def __init__(self, un_ops: "dict[str, Operation]", bin_ops: "dict[str, Operation]", funcs: "dict[str, Function]"):
        pp.ParserElement.enablePackrat()
        self._xs = {}
        self._us = {}
        self._un_ops = un_ops
        self._bin_ops = bin_ops
        self._funcs = funcs
        self._expr = self.create_grammar()

    def parse(self, expr_str: str) -> T:
        """Parse a string into a symbolic expression.

        Args:
            expr_str: string representing a symbolic expression

        Returns:
            Parsed symbolic expression
        """
        parsed = self._expr.parseString(expr_str, parseAll=True).asList()
        log_trace(f"Parsed expression: {parsed}")
        return self.convert_parse_to_ast(parsed)

    @overload
    def parse_to_lambda(self, expr_str: str) -> Callable: ...
    @overload
    def parse_to_lambda(self, expr_str: "tuple[str, ...]") -> "tuple[Callable, ...]": ...
    def parse_to_lambda(self, expr_str: "str | tuple[str, ...]") -> "Callable | tuple[Callable, ...]":
        """Parse a string into a lambda function.
        Invoking the returned function will evaluate the expression with the given variables as keyword arguments.

        Args:
            expr_str: string representing a symbolic expression

        Returns:
            A lambda function that takes the variables as keyword arguments and returns the evaluated expression.
        """
        if isinstance(expr_str, str):
            expr = self.parse(expr_str)
            return partial(self.evaluate, expr)
        exprs = [self.parse(expr) for expr in expr_str]
        return tuple(partial(self.evaluate, expr) for expr in exprs)

    @property
    def xs(self):
        """State variables"""
        return self._xs

    @property
    def us(self):
        """Control variables"""
        return self._us

    def create_grammar(self):
        """Create the grammar for the parser"""
        decimal = pp.Combine(pp.Word(pp.nums) + "." + pp.Word(pp.nums)).setParseAction(lambda t: float(t[0]))
        integer = pp.Word(pp.nums).setParseAction(lambda t: int(t[0]))
        number = decimal | integer

        variable_x = pp.Word("x", pp.nums).setParseAction(self.cached_var_parse_action)
        variable_u = pp.Word("u", pp.nums).setParseAction(self.cached_var_parse_action)
        variable = variable_x | variable_u

        func_names = pp.MatchFirst(pp.Keyword(fun) for fun in self._funcs.keys())

        expr = pp.Forward()

        nested_paren_expr = pp.nestedExpr("(", ")", content=expr)
        func_call = pp.Group(func_names + nested_paren_expr)

        expr <<= pp.infixNotation(
            number | variable | nested_paren_expr | func_call,
            [(name, op.num_operands, op.associativity) for name, op in self._un_ops.items()]
            + [(name, op.num_operands, op.associativity) for name, op in self._bin_ops.items()],
        )
        return expr

    def convert_parse_to_ast(self, parsed: "list[pp.ParseResults]") -> T:
        """Convert a parsed expression to an AST.

        Args:
            parsed (list): A parsed expression.

        Returns:
            An abstract syntax tree (AST).
        """
        if not isinstance(parsed, list):
            return parsed

        if len(parsed) == 1:
            return self.convert_parse_to_ast(parsed[0])
        if len(parsed) >= 3 and len(parsed) % 2 == 1:
            op_func = self._bin_ops[parsed[1]].ast_action
            args = [self.convert_parse_to_ast(x) for x in parsed[::2]]
            acc = args[0]
            for arg in args[1:]:
                acc = op_func(acc, arg)
            return acc
        if parsed[0] in self._un_ops:
            operand = self.convert_parse_to_ast(parsed[1])
            return self._un_ops[parsed[0]].ast_action(operand)
        if parsed[0] in self._funcs:
            func = self._funcs[parsed[0]].ast_action
            args = [self.convert_parse_to_ast(arg) for arg in parsed[1]]
            return func(*args)
        raise ValueError(f"Unknown operation or function: {parsed[0]}")

    def cached_var_parse_action(self, t: pp.ParseResults):
        """Create a variable from a parsed token or return the cached variable if it already exists.

        Args:
            t: parsed token

        Raises:
            ValueError: Variable names must start with x or u

        Returns:
            The symbolic variable associated with the variable name in the token.
        """
        var_name: str = t[0]
        if var_name.startswith("x"):
            return self._xs.setdefault(var_name, self.var_parse_action(var_name))
        if var_name.startswith("u"):
            return self._us.setdefault(var_name, self.var_parse_action(var_name))
        raise ValueError(f"Invalid variable name {var_name}")

    @abstractmethod
    def var_parse_action(self, name: str):
        """Abstract method for generating symbolic variables based on the given name.

        Args:
            name: variable name

        Returns:
            Symbolic variable
        """

    @abstractmethod
    def substitute(self, expr: "T", **subs: "dict[str, float]") -> "T":
        """Substitute variables in the expression with their values.
        This method should be implemented to handle the specific symbolic expression type.

        Args:
            expr: The symbolic expression to substitute variables in

        Returns:
            The symbolic expression with variables substituted with their values.
        """
        pass

    @abstractmethod
    def evaluate(self, expr: "T", **subs: "dict[str, float]") -> "float | bool":
        """Evaluate the expression with the given substitutions.
        The result should be a float or a boolean value.
        Not all symbolic expressions can be evaluated to a number or boolean,
        so an exception may be raised if the expression cannot be evaluated.

        Args:
            expr: symbolic expression to evaluate

        Raises:
            ValueError: if the expression cannot be evaluated to a number or boolean

        Returns:
            The evaluated result of the expression as a python float or boolean value.
        """
        pass


class Z3Parser(SymbolicParser["z3.ExprRef"]):
    """Parser for Z3 expressions"""

    def __init__(self):
        super().__init__(
            un_ops={
                "-": Operation(pp.opAssoc.RIGHT, 1, lambda x: -x),
                "~": Operation(pp.opAssoc.RIGHT, 1, z3.Not),
            },
            bin_ops={
                "**": Operation(pp.opAssoc.RIGHT, 2, lambda x1, x2: x1**x2),
                "/": Operation(pp.opAssoc.LEFT, 2, lambda x1, x2: x1 / x2),
                "*": Operation(pp.opAssoc.LEFT, 2, lambda x1, x2: x1 * x2),
                "-": Operation(pp.opAssoc.LEFT, 2, lambda x1, x2: x1 - x2),
                "+": Operation(pp.opAssoc.LEFT, 2, lambda x1, x2: x1 + x2),
                "=": Operation(pp.opAssoc.LEFT, 2, lambda x1, x2: x1 == x2),
                ">": Operation(pp.opAssoc.LEFT, 2, lambda x1, x2: x1 > x2),
                ">=": Operation(pp.opAssoc.LEFT, 2, lambda x1, x2: x1 >= x2),
                "<": Operation(pp.opAssoc.LEFT, 2, lambda x1, x2: x1 < x2),
                "<=": Operation(pp.opAssoc.LEFT, 2, lambda x1, x2: x1 <= x2),
            },
            funcs={
                "and": Function(z3.And),
                "or": Function(z3.Or),
                "if": Function(z3.If),
                "sum": Function(sum),
                "not": Function(z3.Not),
                "sqrt": Function(z3.Sqrt),
            },
        )

    def var_parse_action(self, name: str):
        log_trace(f"Creating z3 variable {name}")
        return z3.Real(name)

    def substitute(self, expr: "z3.ExprRef", **subs: "dict[str, float]") -> "z3.ExprRef":
        subs_pairs = [(self.xs[var], z3.RealVal(value)) for var, value in subs.items() if var in self.xs]
        subs_pairs += [(self.us[var], z3.RealVal(value)) for var, value in subs.items() if var in self.us]
        return z3.substitute(expr, *subs_pairs)

    def evaluate(self, expr: "z3.ExprRef", **subs: "dict[str, float]"):
        val = z3.simplify(self.substitute(expr, **subs))
        if isinstance(val, z3.AlgebraicNumRef):
            val = val.approx()
        if isinstance(val, z3.BoolRef):
            return z3.is_true(val)
        if isinstance(val, z3.RatNumRef):
            return float(val.numerator_as_long()) / float(val.denominator_as_long())
        if isinstance(val, z3.IntNumRef):
            return float(val.as_long())
        raise ValueError(f"The evaluated expression is not a number or boolean: {val} -> {type(val)}")


class DrealParser(SymbolicParser["dreal.Expression | dreal.Formula"]):
    """Parser for dReal expressions"""

    def __init__(self):
        super().__init__(
            un_ops={
                "-": Operation(pp.opAssoc.RIGHT, 1, lambda x: -x),
                "~": Operation(pp.opAssoc.RIGHT, 1, dreal.Not),
            },
            bin_ops={
                "**": Operation(pp.opAssoc.RIGHT, 2, lambda x1, x2: x1**x2),
                "/": Operation(pp.opAssoc.LEFT, 2, lambda x1, x2: x1 / x2),
                "*": Operation(pp.opAssoc.LEFT, 2, lambda x1, x2: x1 * x2),
                "-": Operation(pp.opAssoc.LEFT, 2, lambda x1, x2: x1 - x2),
                "+": Operation(pp.opAssoc.LEFT, 2, lambda x1, x2: x1 + x2),
                "=": Operation(pp.opAssoc.LEFT, 2, lambda x1, x2: x1 == x2),
                ">": Operation(pp.opAssoc.LEFT, 2, lambda x1, x2: x1 > x2),
                ">=": Operation(pp.opAssoc.LEFT, 2, lambda x1, x2: x1 >= x2),
                "<": Operation(pp.opAssoc.LEFT, 2, lambda x1, x2: x1 < x2),
                "<=": Operation(pp.opAssoc.LEFT, 2, lambda x1, x2: x1 <= x2),
            },
            funcs={
                "and": Function(dreal.And),
                "or": Function(dreal.Or),
                "if": Function(lambda c, i, e: dreal.And(dreal.Implies(c, i), dreal.Implies(dreal.Not(c), e))),
                "sum": Function(sum),
                "not": Function(dreal.Not),
                "sin": Function(dreal.sin),
                "cos": Function(dreal.cos),
                "exp": Function(dreal.exp),
                "log": Function(dreal.log),
                "sqrt": Function(dreal.sqrt),
                "tanh": Function(dreal.tanh),
            },
        )

    def var_parse_action(self, name: str):
        log_trace(f"Creating dreal variable {name}")
        return dreal.Variable(name)

    def substitute(self, expr: "dreal.Expression | dreal.Formula", **subs: "dict[str, float]"):
        subs_dict = {self.xs[var]: value for var, value in subs.items() if var in self.xs}
        subs_dict.update({self.us[var]: value for var, value in subs.items() if var in self.us})
        return expr.Substitute(subs_dict)

    def evaluate(self, expr: "dreal.Expression | dreal.Formula", **subs: "dict[str, float]") -> "float | bool":
        return self.substitute(expr, **subs).Evaluate()


class SympyParser(SymbolicParser[sp.Expr]):
    """Parser for sympy expressions"""

    def __init__(self):
        super().__init__(
            un_ops={
                "-": Operation(pp.opAssoc.RIGHT, 1, lambda x: -x),
                "~": Operation(pp.opAssoc.RIGHT, 1, sp.Not),
            },
            bin_ops={
                "**": Operation(pp.opAssoc.RIGHT, 2, lambda x1, x2: x1**x2),
                "/": Operation(pp.opAssoc.LEFT, 2, lambda x1, x2: x1 / x2),
                "*": Operation(pp.opAssoc.LEFT, 2, lambda x1, x2: x1 * x2),
                "-": Operation(pp.opAssoc.LEFT, 2, lambda x1, x2: x1 - x2),
                "+": Operation(pp.opAssoc.LEFT, 2, lambda x1, x2: x1 + x2),
                "=": Operation(pp.opAssoc.LEFT, 2, sp.Equality),
                ">": Operation(pp.opAssoc.LEFT, 2, lambda x1, x2: x1 > x2),
                ">=": Operation(pp.opAssoc.LEFT, 2, lambda x1, x2: x1 >= x2),
                "<": Operation(pp.opAssoc.LEFT, 2, lambda x1, x2: x1 < x2),
                "<=": Operation(pp.opAssoc.LEFT, 2, lambda x1, x2: x1 <= x2),
            },
            funcs={
                "and": Function(sp.And),
                "or": Function(sp.Or),
                "if": Function(sp.ITE),
                "sum": Function(sum),
                "not": Function(sp.Not),
                "sin": Function(sp.sin),
                "cos": Function(sp.cos),
                "exp": Function(sp.exp),
                "log": Function(sp.log),
                "sqrt": Function(sp.sqrt),
                "tanh": Function(sp.tanh),
            },
        )

    def var_parse_action(self, name: str):
        log_trace(f"Creating sympy variable {name}")
        return sp.var(name)

    def substitute(self, expr: "sp.Expr", **subs: "dict[str, float]") -> "sp.Expr":
        subs_dict = {self.xs[var]: value for var, value in subs.items() if var in self.xs}
        subs_dict.update({self.us[var]: value for var, value in subs.items() if var in self.us})
        return expr.subs(subs_dict)

    def evaluate(self, expr: "sp.Expr", **subs: "dict[str, float]") -> "float | bool":
        return sp.lambdify(list(self.xs.values()) + list(self.us.values()), expr, modules=[np])(**subs)

    def lambdify(self, expr: "sp.Expr") -> "Callable":
        """Convert a sympy expression to a lambda function.

        Args:
            expr: A sympy expression to convert to a lambda function.

        Returns:
            A lambda function that evaluates the expression with the given substitutions.
        """
        return sp.lambdify(list(self.xs.values()) + list(self.us.values()), expr, modules=[np])


class DomainParser:
    """Parser for domains"""

    def __init__(self):
        self.expr = self.create_grammar()

    def create_grammar(self):
        """Create the grammar for the domain parser"""
        decimal = pp.Combine(pp.Optional(pp.oneOf("+ -")) + pp.Word(pp.nums) + "." + pp.Word(pp.nums)).setParseAction(
            lambda t: float(t[0])
        )
        integer = pp.Combine(pp.Optional(pp.oneOf("+ -")) + pp.Word(pp.nums)).setParseAction(lambda t: int(t[0]))
        number = decimal | integer
        number_list = pp.nestedExpr("[", "]", content=pp.delimitedList(number))
        sphere_input = pp.Group(number_list + pp.Suppress(",") + number)
        sphere = pp.Keyword("Sphere") + pp.nestedExpr("(", ")", content=sphere_input)
        rectangle_input = pp.Group(number_list + pp.Suppress(",") + number_list)
        rectangle = pp.Keyword("Rectangle") + pp.nestedExpr("(", ")", content=rectangle_input)
        torus_input = pp.Group(number_list + pp.Suppress(",") + number + pp.Suppress(",") + number)
        torus = pp.Keyword("Torus") + pp.nestedExpr("(", ")", content=torus_input)
        domain = sphere | rectangle | torus

        sphere.setParseAction(self.make_sphere)
        rectangle.setParseAction(self.make_rectangle)
        torus.setParseAction(self.make_torus)

        domain_expr = pp.Forward()
        # Domain parser. TODO: Maybe add domain operations like intersection, union, etc.
        domain_expr <<= domain
        return domain_expr

    @staticmethod
    def make_sphere(t):
        """Make a sphere object from a parsed sphere token"""
        center = t.asList()[1][0][0]
        radius = t.asList()[1][0][1]
        return domains.Sphere(center, radius)

    @staticmethod
    def make_rectangle(t):
        """Make a rectangle object from a parsed rectangle token"""
        lb = t.asList()[1][0][0]
        ub = t.asList()[1][0][1]
        return domains.Rectangle(lb, ub)

    @staticmethod
    def make_torus(t):
        """Make a torus object from a parsed torus token"""
        center = t.asList()[1][0][0]
        out_radius = t.asList()[1][0][1]
        inner_radius = t.asList()[1][0][2]
        return domains.Torus(center, out_radius, inner_radius)
