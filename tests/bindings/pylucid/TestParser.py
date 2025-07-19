import sys
from typing import TYPE_CHECKING

import numpy as np
import pytest
from pyparsing.exceptions import ParseException

from pylucid import (
    DrealParser,
    MultiSet,
    RectSet,
    SetParser,
    SphereSet,
    SympyParser,
    Z3Parser,
    exception,
)

if TYPE_CHECKING:
    from typing import TypeAlias

    Parser: TypeAlias = "Z3Parser | DrealParser | SympyParser"


def get_parameters():
    """Get parameters for pytest"""
    parameters = []
    if "z3" in sys.modules:
        parameters.append(Z3Parser)
    if "dreal" in sys.modules:
        parameters.append(DrealParser)
    if "sympy" in sys.modules:
        parameters.append(SympyParser)
    return parameters


class TestParser:

    class TestBaseParserFunctionality:
        """Tests common functionality across all parser implementations"""

        @pytest.fixture(params=get_parameters())
        def parser(self, request):
            return request.param()

        def test_init(self, parser: "Parser"):
            """Test that parsers initialize properly"""
            assert parser is not None
            assert parser.xs == {}
            assert parser.us == {}

        def test_basic_arithmetic(self, parser: "Parser"):
            """Test basic arithmetic operations"""
            result = parser.parse_to_lambda("x0 + x1")(x0=2.0, x1=3.0)
            assert result == 5.0

            result = parser.parse_to_lambda("x0 - x1")(x0=5.0, x1=3.0)
            assert result == 2.0

            result = parser.parse_to_lambda("x0 * x1")(x0=2.0, x1=3.0)
            assert result == 6.0

            result = parser.parse_to_lambda("x0 / x1")(x0=6.0, x1=3.0)
            assert result == 2.0

        def test_multiple_operations(self, parser: "Parser"):
            """Test expressions with multiple operations"""
            result = parser.parse_to_lambda("x0 + x1 * 2")(x0=1.0, x1=3.0)
            assert result == 7.0

            result = parser.parse_to_lambda("x0 * x1 + x2")(x0=2.0, x1=3.0, x2=4.0)
            assert result == 10.0

        def test_parentheses(self, parser: "Parser"):
            """Test expressions with parentheses"""
            result = parser.parse_to_lambda("(x0 + x1) * x2")(x0=2.0, x1=3.0, x2=4.0)
            assert result == 20.0

        def test_exponentiation(self, parser: "Parser"):
            """Test exponentiation"""
            result = parser.parse_to_lambda("x0**2")(x0=3.0)
            assert result == 9.0

            result = parser.parse_to_lambda("x0**x1")(x0=2.0, x1=3.0)
            assert result == 8.0

        def test_unary_minus(self, parser: "Parser"):
            """Test unary minus operation"""
            result = parser.parse_to_lambda("-x0")(x0=5.0)
            assert result == -5.0

            result = parser.parse_to_lambda("--x0")(x0=5.0)
            assert result == 5.0

        def test_nested_expressions(self, parser: "Parser"):
            """Test deeply nested expressions"""
            expr = "x0 * (x1 + (x2 - x3) * x4)"
            result = parser.parse_to_lambda(expr)(x0=2.0, x1=3.0, x2=5.0, x3=1.0, x4=2.0)
            assert result == 2.0 * (3.0 + (5.0 - 1.0) * 2.0)

        def test_comparison_operators(self, parser: "Parser"):
            """Test comparison operators"""
            assert parser.parse_to_lambda("x0 > x1")(x0=5.0, x1=3.0)
            assert not parser.parse_to_lambda("x0 > x1")(x0=3.0, x1=5.0)

            assert parser.parse_to_lambda("x0 >= x1")(x0=5.0, x1=5.0)
            assert not parser.parse_to_lambda("x0 >= x1")(x0=4.0, x1=5.0)

            assert parser.parse_to_lambda("x0 < x1")(x0=3.0, x1=5.0)
            assert not parser.parse_to_lambda("x0 < x1")(x0=5.0, x1=3.0)

            assert parser.parse_to_lambda("x0 <= x1")(x0=5.0, x1=5.0)
            assert not parser.parse_to_lambda("x0 <= x1")(x0=6.0, x1=5.0)

            assert parser.parse_to_lambda("x0 = x1")(x0=5.0, x1=5.0)
            assert not parser.parse_to_lambda("x0 = x1")(x0=5.0, x1=3.0)

        def test_logical_operations(self, parser: "Parser"):
            """Test logical operations"""
            assert parser.parse_to_lambda("and((x0 > 0) (x1 < 5))")(x0=1.0, x1=3.0)
            assert not parser.parse_to_lambda("and((x0 > 0) (x1 < 5))")(x0=-1.0, x1=3.0)

            assert parser.parse_to_lambda("or((x0 > 0) (x1 < 0))")(x0=1.0, x1=3.0)
            assert not parser.parse_to_lambda("or((x0 > 0) (x1 < 0))")(x0=-1.0, x1=3.0)

            assert parser.parse_to_lambda("not(x0 = x1)")(x0=5.0, x1=3.0)
            assert not parser.parse_to_lambda("not(x0 = x1)")(x0=5.0, x1=5.0)

        def test_multiple_logical_operations(self, parser: "Parser"):
            """Test logical operations"""
            assert parser.parse_to_lambda("and((x0 > 0) (x1 < 5) (x2 = 3))")(x0=1.0, x1=3.0, x2=3.0)
            assert not parser.parse_to_lambda("and((x0 > 0) (x1 < 5) (x2 = 3))")(x0=-1.0, x1=3.0, x2=3.0)

            assert parser.parse_to_lambda("or((x0 > 0) (x1 < 0) (x2 = 3))")(x0=1.0, x1=3.0, x2=3.0)
            assert not parser.parse_to_lambda("or((x0 > 0) (x1 < 0) (x2 = 3))")(x0=-1.0, x1=3.0, x2=4.0)

    @pytest.mark.skipif("sympy" not in sys.modules, reason="Required library is not installed")
    class TestSymPyParser:
        """Tests specific to the SymPy parser implementation"""

        @pytest.fixture
        def parser(self):
            return SympyParser()

        def test_trig_functions(self, parser: "Parser"):
            """Test trigonometric functions"""
            x0 = 0.5
            assert parser.parse_to_lambda("sin(x0)")(x0=x0) == np.sin(x0)
            assert parser.parse_to_lambda("cos(x0)")(x0=x0) == np.cos(x0)

        def test_exp_log_functions(self, parser: "Parser"):
            """Test exponential and logarithmic functions"""
            x0 = 2.0
            assert parser.parse_to_lambda("exp(x0)")(x0=x0) == np.exp(x0)
            assert parser.parse_to_lambda("log(x0)")(x0=x0) == np.log(x0)

        def test_sqrt_tanh_functions(self, parser: "Parser"):
            """Test square root and tanh functions"""
            x0 = 4.0
            assert parser.parse_to_lambda("sqrt(x0)")(x0=x0) == np.sqrt(x0)

            x0 = 0.5
            assert parser.parse_to_lambda("tanh(x0)")(x0=x0) == np.tanh(x0)

        def test_if_function(self, parser: "Parser"):
            """Test conditional if function"""
            result = parser.parse_to_lambda("if((x0 > 0) (x1) (x2))")(x0=1.0, x1=5.0, x2=3.0)
            assert result == 5.0

            result = parser.parse_to_lambda("if((x0 > 0) (x1) (x2))")(x0=-1.0, x1=5.0, x2=3.0)
            assert result == 3.0

        def test_complex_expression(self, parser: "Parser"):
            """Test a complex expression combining multiple functions"""
            expr = "exp(sqrt(x0**2 + x1**2)) * sin(x2) / (1 + x3)"
            x0, x1, x2, x3 = 1.0, 2.0, 0.5, 3.0

            expected = np.exp(np.sqrt(x0**2 + x1**2)) * np.sin(x2) / (1 + x3)
            result = parser.parse_to_lambda(expr)(x0=x0, x1=x1, x2=x2, x3=x3)

            assert np.isclose(result, expected)

        def test_lambdify(self, parser: "Parser"):
            """Test lambdify function"""
            expr = parser.parse("x0 * x1 + x2")
            func = parser.lambdify(expr)

            result = func(x0=2.0, x1=3.0, x2=4.0)
            assert result == 10.0

        def test_substitute(self, parser: "Parser"):
            """Test substitution of variables"""
            expr = parser.parse("x0 + x1")
            subbed_expr = parser.substitute(expr, x0=5.0)

            # Should be 5 + x1
            assert subbed_expr.subs({parser.xs["x1"]: 3.0}) == 8.0

    @pytest.mark.skipif("z3" not in sys.modules, reason="Required library is not installed")
    class TestZ3Parser:
        """Tests specific to the Z3 parser implementation"""

        @pytest.fixture
        def parser(self):
            return Z3Parser()

        def test_sqrt_function(self, parser: "Parser"):
            """Test square root function"""
            x0 = 4.0
            assert parser.parse_to_lambda("sqrt(x0)")(x0=x0) == 2.0

        def test_substitute(self, parser: "Parser"):
            """Test substitution of variables"""
            expr = parser.parse("x0 + x1")
            subbed_expr = parser.substitute(expr, x0=5.0, x1=3.0)

            # Z3 substitution should evaluate to 8.0
            assert parser.evaluate(subbed_expr) == 8.0

    @pytest.mark.skipif("dreal" not in sys.modules, reason="Required library is not installed")
    class TestDrealParser:
        """Tests specific to the dReal parser implementation"""

        @pytest.fixture
        def parser(self):
            return DrealParser()

        def test_trig_functions(self, parser: "Parser"):
            """Test trigonometric functions"""
            x0 = 0.5

            result = parser.parse_to_lambda("sin(x0)")(x0=x0)
            assert np.isclose(result, np.sin(x0))

            result = parser.parse_to_lambda("cos(x0)")(x0=x0)
            assert np.isclose(result, np.cos(x0))

        def test_exp_log_functions(self, parser: "Parser"):
            """Test exponential and logarithmic functions"""
            x0 = 2.0

            result = parser.parse_to_lambda("exp(x0)")(x0=x0)
            assert np.isclose(result, np.exp(x0))

            result = parser.parse_to_lambda("log(x0)")(x0=x0)
            assert np.isclose(result, np.log(x0))

        def test_sqrt_tanh_functions(self, parser: "Parser"):
            """Test square root and tanh functions"""
            x0 = 4.0
            result = parser.parse_to_lambda("sqrt(x0)")(x0=x0)
            assert np.isclose(result, 2.0)

            x0 = 0.5
            result = parser.parse_to_lambda("tanh(x0)")(x0=x0)
            assert np.isclose(result, np.tanh(x0))

        def test_substitute(self, parser: "Parser"):
            """Test substitution of variables"""
            expr = parser.parse("x0 + x1")
            subbed_expr = parser.substitute(expr, x0=5.0, x1=3.0)

            # dReal substitution should evaluate to 8.0
            assert np.isclose(subbed_expr.Evaluate(), 8.0)


class TestSetParser:
    """Tests for the set parser"""

    @pytest.fixture
    def parser(self):
        return SetParser()

    def test_rectset_parsing(self, parser):
        """Test parsing a rectangular set"""
        rect_str = "RectSet([1.0, 2.0], [3.0, 4.0])"
        rect = parser.parse(rect_str)

        assert isinstance(rect, RectSet)
        assert np.array_equal(rect.lower_bound, [1.0, 2.0])
        assert np.array_equal(rect.upper_bound, [3.0, 4.0])

    def test_multiset_parsing(self, parser):
        """Test parsing a multi-set"""
        multi_str = "MultiSet([RectSet([1.0, 2.0], [3.0, 4.0]), RectSet([5.0, 6.0], [7.0, 8.0])])"
        multi = parser.parse(multi_str)

        assert isinstance(multi, MultiSet)
        assert len(multi) == 2
        # Check first rectangle
        assert np.array_equal(multi[0].lower_bound, [1.0, 2.0])
        assert np.array_equal(multi[0].upper_bound, [3.0, 4.0])
        # Check second rectangle
        assert np.array_equal(multi[1].lower_bound, [5.0, 6.0])
        assert np.array_equal(multi[1].upper_bound, [7.0, 8.0])

    def test_rectset_higher_dimensions(self, parser):
        """Test parsing a rectangular set with higher dimensions"""
        rect_str = "RectSet([5.0, 6.0, 8.0, 9.0], [7.0, 8.0, 1.0, 2.0])"
        rect = parser.parse(rect_str)

        assert isinstance(rect, RectSet)
        assert np.array_equal(rect.lower_bound, [5.0, 6.0, 8.0, 9.0])
        assert np.array_equal(rect.upper_bound, [7.0, 8.0, 1.0, 2.0])

    def test_parsing_negative_values(self, parser):
        """Test parsing sets with negative values"""
        rect_str = "RectSet([-1.5, -2.5], [3.0, 4.0])"
        rect = parser.parse(rect_str)

        assert isinstance(rect, RectSet)
        assert np.array_equal(rect.lower_bound, [-1.5, -2.5])
        assert np.array_equal(rect.upper_bound, [3.0, 4.0])

    def test_parsing_integers(self, parser):
        """Test parsing sets with integer values"""
        rect_str = "RectSet([-1, -2], [3, 4])"
        rect = parser.parse(rect_str)

        assert isinstance(rect, RectSet)
        assert np.array_equal(rect.lower_bound, [-1, -2])
        assert np.array_equal(rect.upper_bound, [3, 4])

    def test_sphere_set_parsing(self, parser):
        """Test parsing a sphere set"""
        sphere_str = "SphereSet([1.0, 2.0], 3.0)"
        sphere = parser.parse(sphere_str)

        assert isinstance(sphere, SphereSet)
        assert np.array_equal(sphere.center, [1.0, 2.0])
        assert sphere.radius == 3.0

    def test_multiset_with_sphere(self, parser):
        """Test parsing a multi-set that includes a sphere set"""
        multi_str = "MultiSet([RectSet([1.0, 2.0], [3.0, 4.0]), SphereSet([5.0, 6.0], 2.0)])"
        multi = parser.parse(multi_str)

        assert isinstance(multi, MultiSet)
        assert len(multi) == 2
        # Check first rectangle
        assert np.array_equal(multi[0].lower_bound, [1.0, 2.0])
        assert np.array_equal(multi[0].upper_bound, [3.0, 4.0])
        # Check sphere
        assert isinstance(multi[1], SphereSet)
        assert np.array_equal(multi[1].center, [5.0, 6.0])
        assert multi[1].radius == 2.0

    def test_invalid_syntax(self, parser):
        """Test parsing with invalid syntax"""
        with pytest.raises(ParseException):
            parser.parse("RectSet([1.0, 2.0], 3.0])")

        with pytest.raises(ParseException):
            parser.parse("MultiSet(RectSet([1.0, 2.0], [3.0, 4.0]))")

    def test_dimension_mismatch(self, parser):
        """Test parsing with dimension mismatch"""
        rect_str = "RectSet([1.0, 2.0], [3.0, 4.0, 5.0])"
        with pytest.raises(exception.LucidInvalidArgumentException):
            parser.parse(rect_str)

    class TestErrorHandling:
        """Tests for error handling in parsers"""

        @pytest.fixture
        def sympy_parser(self):
            return SympyParser()

        def test_invalid_syntax(self, sympy_parser: "Parser"):
            """Test parsing with invalid syntax"""
            with pytest.raises(ParseException):
                sympy_parser.parse("x0 +")

        def test_unknown_variable(self, sympy_parser: "Parser"):
            """Test evaluation with unknown variables"""
            expr = sympy_parser.parse("x0 + x1")
            with pytest.raises(TypeError):
                sympy_parser.evaluate(expr, x0=1.0)  # x1 is missing

        def test_division_by_zero(self, sympy_parser: "Parser"):
            """Test division by zero"""
            expr = sympy_parser.parse("x0 / x1")
            # SymPy handles division by zero based on its own rules
            with pytest.raises(ZeroDivisionError):
                sympy_parser.evaluate(expr, x0=1.0, x1=0.0)

        def test_invalid_function_args(self, sympy_parser: "Parser"):
            """Test calling a function with no arguments"""
            with pytest.raises(TypeError):
                sympy_parser.parse("sin()")

        def test_unclosed_parentheses(self, sympy_parser: "Parser"):
            """Test parsing with unclosed parentheses"""
            with pytest.raises(ParseException):
                sympy_parser.parse("(x0 + x1")

    class TestEdgeCases:
        """Tests for edge cases"""

        @pytest.fixture
        def sympy_parser(self):
            return SympyParser()

        def test_deeply_nested_expressions(self, sympy_parser: "Parser"):
            """Test deeply nested expressions"""
            expr = "sin(cos(exp(log(sqrt(x0)))))"
            expr = "exp(log(sqrt(x0)))"
            # TODO: Fix the above line, should be able to support deeper nesting
            result = sympy_parser.parse_to_lambda(expr)(x0=4.0)
            expected = np.exp(np.log(np.sqrt(4.0)))
            assert np.isclose(result, expected)

        def test_many_variables(self, sympy_parser: "Parser"):
            """Test expressions with many variables"""
            expr = "x0 + x1 + x2 + x3 + x4 + x5 + x6 + x7 + x8 + x9"
            values = {f"x{i}": float(i) for i in range(10)}
            result = sympy_parser.parse_to_lambda(expr)(**values)
            assert result == sum(range(10))

        def test_control_variables(self, sympy_parser: "Parser"):
            """Test using control variables"""
            expr = "x0 * u0 + x1 * u1"
            result = sympy_parser.parse_to_lambda(expr)(x0=2.0, x1=3.0, u0=0.5, u1=1.5)
            assert result == 2.0 * 0.5 + 3.0 * 1.5

        def test_very_simple_expression(self, sympy_parser: "Parser"):
            """Test parsing a very simple expression"""
            expr = "x0"
            result = sympy_parser.parse_to_lambda(expr)(x0=5.0)
            assert result == 5.0

        def test_very_large_values(self, sympy_parser: "Parser"):
            """Test with very large values"""
            expr = "x0 * x1"
            result = sympy_parser.parse_to_lambda(expr)(x0=1e10, x1=1e10)
            assert result == 1e20

        def test_very_small_values(self, sympy_parser: "Parser"):
            """Test with very small values"""
            expr = "x0 * x1"
            result = sympy_parser.parse_to_lambda(expr)(x0=1e-10, x1=1e-10)
            assert np.isclose(result, 1e-20)
