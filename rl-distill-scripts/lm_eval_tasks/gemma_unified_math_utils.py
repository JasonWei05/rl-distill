"""Shared utils for the unified Gemma-3 PT math eval (MATH + GSM8K + downstream).

One few-shot block, one answer contract (\\boxed{}), one extractor (math_verify), used
across every math benchmark so a single chat-templated prompt is directly comparable.

The few-shot exemplars interleave the harness's built-in MATH (Minerva 4-shot) and GSM8K
(8-shot CoT) examples, each rewritten to end in a `\\boxed{}` final answer so extraction is
uniform. Evaluated with `--apply_chat_template --fewshot_as_multiturn` (IT chat template).
"""

import datasets
from math_verify import parse, verify


# --- unified few-shot block: 4 MATH (Minerva) + 8 GSM8K, interleaved -------------------
# Each sample is a (question, solution) turn pair; solutions end in "\boxed{ANSWER}".
_MATH = [
    {
        "question": "Find the domain of the expression $\\frac{\\sqrt{x-2}}{\\sqrt{5-x}}$.",
        "solution": "The expressions inside each square root must be non-negative. So $x-2\\ge 0$, giving $x\\ge 2$, and $5-x\\ge 0$, giving $x\\le 5$. The denominator cannot be zero, so $5-x>0$, i.e. $x<5$. Therefore the domain is $[2,5)$. The final answer is $\\boxed{[2,5)}$.",
    },
    {
        "question": "If $\\det \\mathbf{A} = 2$ and $\\det \\mathbf{B} = 12,$ then find $\\det (\\mathbf{A} \\mathbf{B}).$",
        "solution": "We have $\\det(\\mathbf{A}\\mathbf{B}) = (\\det \\mathbf{A})(\\det \\mathbf{B}) = (2)(12) = 24$. The final answer is $\\boxed{24}$.",
    },
    {
        "question": "Terrell usually lifts two 20-pound weights 12 times. If he uses two 15-pound weights instead, how many times must Terrell lift them in order to lift the same total weight?",
        "solution": "Lifting two 20-pound weights 12 times gives $2\\cdot 12\\cdot 20 = 480$ pounds. With two 15-pound weights lifted $n$ times that is $2\\cdot 15\\cdot n = 30n$ pounds. Setting $30n = 480$ gives $n = 16$. The final answer is $\\boxed{16}$.",
    },
    {
        "question": "If the system of equations\n\n\\begin{align*}\n6x-4y&=a,\\\\\n6y-9x &=b.\n\\end{align*}has a solution $(x, y)$ where $x$ and $y$ are both nonzero, find $\\frac{a}{b},$ assuming $b$ is nonzero.",
        "solution": "Multiplying the first equation by $-\\frac{3}{2}$ gives $6y-9x = -\\frac{3}{2}a$. Since $6y-9x = b$, we have $-\\frac{3}{2}a = b$, so $\\frac{a}{b} = -\\frac{2}{3}$. The final answer is $\\boxed{-\\frac{2}{3}}$.",
    },
]

_GSM8K = [
    {
        "question": "There are 15 trees in the grove. Grove workers will plant trees in the grove today. After they are done, there will be 21 trees. How many trees did the grove workers plant today?",
        "solution": "There were 15 trees originally and 21 after planting, so the workers planted $21 - 15 = 6$ trees. The final answer is $\\boxed{6}$.",
    },
    {
        "question": "If there are 3 cars in the parking lot and 2 more cars arrive, how many cars are in the parking lot?",
        "solution": "There are 3 cars originally and 2 more arrive, so there are $3 + 2 = 5$ cars. The final answer is $\\boxed{5}$.",
    },
    {
        "question": "Leah had 32 chocolates and her sister had 42. If they ate 35, how many pieces do they have left in total?",
        "solution": "Together they had $32 + 42 = 74$ chocolates. After eating 35 they have $74 - 35 = 39$ left. The final answer is $\\boxed{39}$.",
    },
    {
        "question": "Jason had 20 lollipops. He gave Denny some lollipops. Now Jason has 12 lollipops. How many lollipops did Jason give to Denny?",
        "solution": "Jason went from 20 to 12 lollipops, so he gave away $20 - 12 = 8$. The final answer is $\\boxed{8}$.",
    },
    {
        "question": "Shawn has five toys. For Christmas, he got two toys each from his mom and dad. How many toys does he have now?",
        "solution": "He started with 5 toys and got $2 + 2 = 4$ more, so he now has $5 + 4 = 9$. The final answer is $\\boxed{9}$.",
    },
    {
        "question": "There were nine computers in the server room. Five more computers were installed each day, from monday to thursday. How many computers are now in the server room?",
        "solution": "Over 4 days, $5\\cdot 4 = 20$ computers were added to the original 9, giving $9 + 20 = 29$. The final answer is $\\boxed{29}$.",
    },
    {
        "question": "Michael had 58 golf balls. On tuesday, he lost 23 golf balls. On wednesday, he lost 2 more. How many golf balls did he have at the end of wednesday?",
        "solution": "After losing 23 he had $58 - 23 = 35$, then after losing 2 more he had $35 - 2 = 33$. The final answer is $\\boxed{33}$.",
    },
    {
        "question": "Olivia has $23. She bought five bagels for $3 each. How much money does she have left?",
        "solution": "Five bagels at $3 cost $5\\cdot 3 = 15$ dollars, so she has $23 - 15 = 8$ dollars left. The final answer is $\\boxed{8}$.",
    },
]

# Interleave: keep MATH's 4 and GSM8K's 8 (the "given" shot counts), spread out.
_INTERLEAVED = [
    _MATH[0], _GSM8K[0], _GSM8K[1],
    _MATH[1], _GSM8K[2], _GSM8K[3],
    _MATH[2], _GSM8K[4], _GSM8K[5],
    _MATH[3], _GSM8K[6], _GSM8K[7],
]


def list_fewshot_samples() -> list[dict]:
    return [{"question": s["question"], "solution": s["solution"], "few_shot": 1} for s in _INTERLEAVED]


# --- shared doc mapping ----------------------------------------------------------------
def doc_to_text(doc: dict) -> str:
    return doc["question"].strip()


def doc_to_target(doc: dict) -> str:
    # fewshot turns render the worked solution; eval docs just carry the gold string.
    if doc.get("few_shot"):
        return " " + doc["solution"].strip()
    return " " + str(doc["gold"]).strip()


def _score(gold: str, pred: str) -> int:
    try:
        p = parse(pred)
    except Exception:
        return 0
    gold = str(gold).strip()
    # try several gold renderings; math_verify.parse wants a math/LaTeX expr, and a bare
    # string like "[2,5)" or "-\frac{2}{3}" parses more reliably when wrapped.
    for gv in (f"\\boxed{{{gold}}}", f"${gold}$", gold):
        try:
            g = parse(gv)
            if verify(g, p):
                return 1
        except Exception:
            continue
    return 0


def process_results(doc: dict, results: list[str]) -> dict:
    pred = results[0] if results else ""
    return {"math_verify": _score(doc["gold"], pred)}


# --- per-dataset normalizers to the common {question, gold} schema ---------------------
def process_docs_gsm8k(dataset: datasets.Dataset) -> datasets.Dataset:
    def _m(doc):
        return {"question": doc["question"], "gold": doc["answer"].split("####")[-1].strip()}
    return dataset.map(_m, remove_columns=[c for c in dataset.column_names if c not in ("question",)])


def process_docs_math500(dataset: datasets.Dataset) -> datasets.Dataset:
    def _m(doc):
        return {"question": doc["problem"], "gold": doc["answer"]}
    return dataset.map(_m, remove_columns=[c for c in dataset.column_names if c not in ("problem",)])


_BOXED_INSTR = "Please output the final answer within \\boxed{}."


def process_docs_verl(dataset: datasets.Dataset) -> datasets.Dataset:
    """verl-format parquet: prompt=[{content,role}], reward_model={ground_truth}."""
    def _m(doc):
        q = doc["prompt"][0]["content"]
        if q.rstrip().endswith(_BOXED_INSTR):
            q = q.rstrip()[: -len(_BOXED_INSTR)].rstrip()
        return {"question": q, "gold": str(doc["reward_model"]["ground_truth"])}
    return dataset.map(_m, remove_columns=dataset.column_names)
