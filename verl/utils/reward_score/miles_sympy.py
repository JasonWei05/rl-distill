# Copyright 2026 The Miles authors
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Miles/DeepScaleR SymPy answer-equivalence grader.

Adapted from ``miles/rollout/rm_hub/math_utils.py`` at Miles commit
98eec3176268820f5a05a6b78478c6cd28bd5aad. Miles in turn attributes the
implementation to Agentica DeepScaleR commit e6080ccd974e. Answer extraction
is intentionally absent: callers must pass the already-selected final box.
"""

import re

import sympy
from pylatexenc import latex2text
from sympy.parsing import sympy_parser


BAD_SUBSTRINGS = ["^{", "^("]
BAD_REGEXES = [r"\^[0-9]+\^", r"\^[0-9][0-9]+"]
TUPLE_CHARS = "()[]"


def _sympy_parse(expr: str):
    py_expr = expr.replace("^", "**")
    return sympy_parser.parse_expr(
        py_expr,
        transformations=(
            sympy_parser.standard_transformations
            + (sympy_parser.implicit_multiplication_application,)
        ),
    )


def _parse_latex(expr: str) -> str:
    expr = expr.replace("\\tfrac", "\\frac")
    expr = expr.replace("\\dfrac", "\\frac")
    expr = expr.replace("\\frac", " \\frac")
    expr = latex2text.LatexNodes2Text().latex_to_text(expr)
    expr = expr.replace("√", "sqrt")
    expr = expr.replace("π", "pi")
    expr = expr.replace("∞", "inf")
    expr = expr.replace("∪", "U")
    expr = expr.replace("·", "*")
    expr = expr.replace("×", "*")
    return expr.strip()


def _is_float(num: str) -> bool:
    try:
        float(num)
        return True
    except Exception:
        return False


def _is_int(x: float) -> bool:
    try:
        return abs(x - int(round(x))) <= 1e-7
    except Exception:
        return False


def _is_frac(expr: str) -> bool:
    return bool(re.search(r"^-?[0-9]+.?/0*[1-9][0-9]*.?$", expr))


def _strip_properly_formatted_commas(expr: str) -> str:
    pattern = re.compile(r"(\d)(,)(\d\d\d)($|\D)")
    while True:
        next_expr = pattern.sub(r"\1\3\4", expr)
        if next_expr == expr:
            return next_expr
        expr = next_expr


def _str_is_int(x: str) -> bool:
    try:
        x = _strip_properly_formatted_commas(x)
        value = float(x)
        return abs(value - int(round(value))) <= 1e-7
    except Exception:
        return False


def _str_to_int(x: str) -> int:
    return int(float(x.replace(",", "")))


def _inject_implicit_mixed_number(step: str) -> str:
    return re.sub(r"([0-9]) +([0-9])", r"\1+\2", step)


def _normalize(expr: str | None) -> str | None:
    if expr is None:
        return None

    text_match = re.search(r"^\\text\{(?P<text>.+?)\}$", expr)
    if text_match is not None:
        expr = text_match.group("text")

    expr = expr.replace("\\%", "%")
    expr = expr.replace("\\$", "$")
    expr = expr.replace("$", "")
    expr = expr.replace("%", "")
    expr = expr.replace(" or ", " , ")
    expr = expr.replace(" and ", " , ")
    expr = expr.replace("million", "*10^6")
    expr = expr.replace("billion", "*10^9")
    expr = expr.replace("trillion", "*10^12")

    for unit in [
        "degree",
        "cm",
        "centimeter",
        "meter",
        "mile",
        "second",
        "minute",
        "hour",
        "day",
        "week",
        "month",
        "year",
        "foot",
        "feet",
        "inch",
        "yard",
    ]:
        expr = re.sub(rf"{unit}(es)?(s)? *(\^[0-9]+)?", "", expr)
    expr = re.sub(r"\^ *\\circ", "", expr)

    if len(expr) > 0 and expr[0] == "{" and expr[-1] == "}":
        expr = expr[1:-1]

    expr = re.sub(",\\\\! *", "", expr)
    if _is_float(expr) and _is_int(float(expr)):
        expr = str(int(round(float(expr))))
    if "\\" in expr:
        try:
            expr = _parse_latex(expr)
        except Exception:
            pass

    expr = re.sub("- *", "-", expr)
    expr = _inject_implicit_mixed_number(expr)
    expr = expr.replace(" ", "")
    expr = expr.replace("{", "")
    expr = expr.replace("}", "")
    expr = expr.lower()

    if _str_is_int(expr):
        expr = str(_str_to_int(expr))
    return expr


def _count_unknown_letters(expr: str) -> int:
    expr = expr.replace("sqrt", "")
    expr = expr.replace("frac", "")
    return len({character for character in expr if character.isalpha()})


def _should_allow_eval(expr: str) -> bool:
    if _count_unknown_letters(expr) > 2:
        return False
    if any(bad_string in expr for bad_string in BAD_SUBSTRINGS):
        return False
    return not any(re.search(bad_regex, expr) is not None for bad_regex in BAD_REGEXES)


def _are_equal_under_sympy(ground_truth: str, prediction: str) -> bool:
    try:
        expr = f"({ground_truth})-({prediction})"
        if _should_allow_eval(expr):
            return sympy.simplify(_sympy_parse(expr)) == 0
    except Exception:
        pass
    return False


def _split_tuple(expr: str) -> list[str]:
    expr = _strip_properly_formatted_commas(expr)
    if len(expr) == 0:
        return []
    if (
        len(expr) > 2
        and expr[0] in TUPLE_CHARS
        and expr[-1] in TUPLE_CHARS
        and all(character not in expr[1:-1] for character in TUPLE_CHARS)
    ):
        return [element.strip() for element in expr[1:-1].split(",")]
    return [expr]


def grade_answer_sympy(given_answer: str, ground_truth: str) -> bool:
    """Return whether Miles' normalization/SymPy rules accept the answer."""
    ground_truth_normalized = _normalize(ground_truth)
    given_normalized = _normalize(given_answer)

    if ground_truth_normalized is None or given_normalized is None:
        return False
    if ground_truth_normalized == given_normalized:
        return True
    if len(given_normalized) == 0:
        return False

    ground_truth_elements = _split_tuple(ground_truth_normalized)
    given_elements = _split_tuple(given_normalized)
    if len(ground_truth_elements) > 1 and (
        ground_truth_normalized[0] != given_normalized[0]
        or ground_truth_normalized[-1] != given_normalized[-1]
    ):
        return False
    if len(ground_truth_elements) != len(given_elements):
        return False

    for ground_truth_element, given_element in zip(
        ground_truth_elements, given_elements, strict=False
    ):
        if _is_frac(ground_truth_element) and _is_frac(given_element):
            is_correct = ground_truth_element == given_element
        elif _str_is_int(ground_truth_element) != _str_is_int(given_element):
            is_correct = False
        else:
            is_correct = _are_equal_under_sympy(ground_truth_element, given_element)
        if not is_correct:
            return False
    return True
