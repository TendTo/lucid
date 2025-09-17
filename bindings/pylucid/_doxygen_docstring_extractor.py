import re
import sys

from cxxheaderparser.parser import CxxParser
from cxxheaderparser.visitor import CxxVisitor


class DocxygenVisitor(CxxVisitor):
    def __init__(self):
        super().__init__()
        self._current_class = ""
        self._strings: "dict[tuple[str, str], str]" = {}

    @property
    def strings(self) -> "dict[tuple[str, str], str]":
        return self._strings

    def on_class_start(self, state):
        class_name = state.class_decl.typename.segments[0].name
        self._current_class = class_name
        if state.class_decl.doxygen is not None:
            self._strings[(class_name, "")] = self.doxygen_to_sphinx(state.class_decl.doxygen)

    def on_class_end(self, _):
        self._current_class = ""

    def on_class_field(self, state, f):
        if f.doxygen is not None:
            self._strings[(self._current_class, f.name)] = self.doxygen_to_sphinx(f.doxygen)

    def on_class_method(self, state, method):
        method_name = method.name.segments[0].name
        method_name = (
            method_name.replace("operator()", "operator_apply")
            .replace("operator++", "operator_increment")
            .replace("operator--", "operator_decrement")
            .replace("operator*", "operator_multiply")
            .replace("operator+", "operator_add")
            .replace("operator-", "operator_subtract")
            .replace("operator/", "operator_divide")
            .replace("operator>>", "operator_shift_right")
            .replace("operator<<", "operator_shift_left")
            .replace("operator==", "operator_equals")
            .replace("operator!=", "operator_not_equals")
            .replace("operator<", "operator_less_than")
            .replace("operator<=", "operator_less_than_equals")
            .replace("operator>", "operator_greater_than")
            .replace("operator>=", "operator_greater_than_equals")
            .replace("operator[]", "operator_index")
            .replace("operator=", "operator_assign")
            .replace("operator->", "operator_arrow")
            .replace("~", "d_")
            .replace("operatornew[]", "operator_new_array")
        )
        if method.doxygen is not None:
            self._strings[(self._current_class, method_name)] = self.doxygen_to_sphinx(method.doxygen)

    def doxygen_to_sphinx(self, doxystring: str) -> str:
        sphinx = doxystring.replace("///<", "").replace("///", "").replace("/**", "").replace("*/", "")
        sphinx = (
            sphinx.replace("@return", "\n:return:")
            .replace("@code", "\n```c++")
            .replace("@endcode", "```\n")
            .replace("@pre", "\n.. precondition:: ")
            .replace("@note", "\n.. note::")
            .replace("@warning", "\n.. warning::")
            .replace("@todo", "\n.. todo::")
            .replace("@deprecated", "\n.. deprecated::")
            .replace("@see", "\n.. seealso::")
            .replace("@ref", ":ref:`")
            .replace("@brief", "")
            .replace("@f[", "\n.. math::\n   :nowrap:")
            .replace("@f]", "\n")
            .replace("@gamma", ":math: `\\gamma`")
            .replace("@epsilon", ":math: `\\epsilon`")
            .replace("@lambda", ":math: `\\lambda`")
            .replace("@sigma", ":math: `\\sigma`")
            .replace("@sigmal", ":math: `\\sigma_l`")
            .replace("@sigmaf", ":math: `\\sigma_f`")
            .replace("@mu", ":math: `\\mu`")
            .replace("@eta", ":math: `\\eta`")
            .replace("@theta", ":math: `\\theta`")
            .replace("@x", ":math: `x`")
            .replace("@X", ":math: `\\mathcal{X}`")
            .replace("@y", ":math: `y`")
            .replace("@Y", ":math: `\\mathcal{Y}`")
            .replace("@d", ":math: `d`")
            .replace("@n", ":math: `n`")
            .replace("@N", ":math: `N`")
            .replace("@rho", ":math: `\\rho`")
            .replace("@xi", ":math: `x_i`")
            .replace("@xn", ":math: `x_n`")
            .replace("@x1", ":math: `x_1`")
            .replace("@x2", ":math: `x_2`")
            .replace("@fx", ":math: `f(x)`")
            .replace("@nxm", ":math: `n \\times m`")
            .replace("@nxn", ":math: `n \\times n`")
            .replace("@n1xd", ":math: `n_1 \\times d`")
            .replace("@n2xd", ":math: `n_2 \\times d`")
            .replace("@nxd", ":math: `n \\times d`")
            .replace("@nxdx", ":math: `n \\times d_x`")
            .replace("@nxdy", ":math: `n \\times d_y`")
            .replace("@XsubRd", ":math: `\\mathcal{X} \\subseteq \\mathbb{R}^d`")
            .replace("@XsubRdx", ":math: `\\mathcal{X} \\subseteq \\mathbb{R}^{d_x}`")
            .replace("@YsubRdy", ":math: `\\mathcal{Y} \\subseteq \\mathbb{R}^{d_y}`")
        )
        sphinx = re.sub(r"@f\$ *([\w\W]+?) *@f\$", r"\n:math: `\1`:", sphinx)
        sphinx = re.sub(r"@param ([^ ]+)", r"\n:param \1:", sphinx)
        sphinx = re.sub(r"@throw ([^ ]+)", r"\n:raises \1:", sphinx)
        sphinx = re.sub(r"\n\* ", r"\n", sphinx)
        sphinx = re.sub(
            r"@getter{([^}]+), ([^}]+)}", r"Get read-only access to the \1 of the \2.\n\n:return: \1 of the \2", sphinx
        )
        sphinx = re.sub(r"@plot[\w\W]*?(?=@endplot)@endplot\n?", r"", sphinx)
        sphinx = re.sub(r"@tparam.*\n?", r"", sphinx)
        return sphinx.strip()


def valid_header(filename: str) -> bool:
    return (
        filename.endswith((".h", ".hpp", ".hh", ".hxx"))
        and "concept" not in filename
        and "eigen_matrix_base_plugin" not in filename
        and "Configuration" not in filename
        and "constants" not in filename
        and "logging" not in filename
    )


def main():
    if len(sys.argv) < 2:
        print("Usage: python _doxygen_docstring_extractor.py <input-header-file> [output-file]")
        sys.exit(1)
    # Filter out all non-header files
    headers = [f for f in sys.argv[1:-1] if valid_header(f)]
    print(f"Processing header files: {headers}", file=sys.stderr)

    visitor = DocxygenVisitor()
    for header in headers:
        with open(header, "r", encoding="utf-8") as f:
            content = re.sub(
                r"(^#[\w\W]*?(?=[^\\]\n).|\n#[\w\W]*?(?=[^\\]\n).| *requires\(.*|OSTREAM_FORMATTER\(.+|thread_local)",
                "",
                f.read(),
            )
        parser = CxxParser(header, content, visitor=visitor)
        parser.parse()

    data = "\n".join(
        f'constexpr const char* {class_name}_{member_name} = R"({doc})";'
        for (class_name, member_name), doc in visitor.strings.items()
        if doc.strip() != ""
    )
    output_file = sys.argv[-1] if len(sys.argv) >= 3 else None
    if output_file is not None:
        with open(output_file, "w", encoding="utf-8") as f:
            f.write(data)
    else:
        print(data)


if __name__ == "__main__":
    main()
