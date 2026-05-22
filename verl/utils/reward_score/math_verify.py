# Copyright 2024 Bytedance Ltd. and/or its affiliates
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

import multiprocessing
import os
import re
import threading
from concurrent.futures import ProcessPoolExecutor
from concurrent.futures import TimeoutError as FuturesTimeoutError

try:
    from math_verify.errors import TimeoutException
except ImportError:

    class TimeoutException(Exception):
        pass

    print("To use Math-Verify, please install it first by running `pip install math-verify`.")

_pool = None
_pool_lock = threading.Lock()
_BOXED_COMMAND_PATTERN = re.compile(r"\\(?:boxed|fbox)\s*")
_SAFE_BOXED_CONTENT_PATTERN = re.compile(r"^[0-9A-Za-z\\{}()[\].,;:+*/^_=<>|!&%$\s-]+$")
_ALLOWED_MATH_WORDS = {
    "begin",
    "end",
    "frac",
    "sqrt",
    "sin",
    "cos",
    "tan",
    "cot",
    "sec",
    "csc",
    "log",
    "ln",
    "exp",
    "pi",
    "theta",
    "alpha",
    "beta",
    "gamma",
    "delta",
    "pmatrix",
    "bmatrix",
    "matrix",
    "left",
    "right",
    "cdot",
    "times",
    "sum",
    "prod",
    "int",
}


def _get_pool():
    global _pool
    if _pool is None:
        with _pool_lock:
            if _pool is None:
                max_workers = int(os.getenv("VERL_MATH_VERIFY_POOL_WORKERS", "4"))
                _pool = ProcessPoolExecutor(max_workers=max_workers, mp_context=multiprocessing.get_context("spawn"))
    return _pool


def _env_flag(name: str, default: bool = False) -> bool:
    value = os.getenv(name)
    if value is None:
        return default
    return value.lower() in {"1", "true", "yes", "on"}


def _extract_fast_boxed_answer(model_output: str) -> str | None:
    """Fast guard for fresh base-model gibberish before spawning math-verify.

    The DAPO reward only accepts extracted LaTeX answers. For fresh PT rollouts,
    most long responses contain no usable boxed answer and can safely receive
    zero reward without running the expensive parser/verifier.
    """
    matches = list(_BOXED_COMMAND_PATTERN.finditer(model_output))
    if not matches:
        return None
    start = matches[-1].end()
    if start >= len(model_output):
        return None

    if model_output[start] == "{":
        depth = 0
        end = None
        for i in range(start, min(len(model_output), start + 1024)):
            if model_output[i] == "{":
                depth += 1
            elif model_output[i] == "}":
                depth -= 1
                if depth == 0:
                    end = i
                    break
        if end is None:
            return None
        content = model_output[start + 1 : end].strip()
        boxed = model_output[matches[-1].start() : end + 1]
    else:
        end = start
        while end < len(model_output) and not model_output[end].isspace() and model_output[end] not in "$\n\r":
            end += 1
        content = model_output[start:end].strip()
        boxed = model_output[matches[-1].start() : end]

    if not content or len(content) > 256:
        return None
    if not any(ch.isdigit() for ch in content):
        return None
    lowered = content.lower()
    if any(bad in lowered for bad in ("http", "www.", "text:", "@blank", "projectid")):
        return None
    if not any(ch.isdigit() or ch.isalpha() for ch in content):
        return None
    if not _SAFE_BOXED_CONTENT_PATTERN.match(content):
        return None
    words = [word.lower() for word in re.findall(r"[A-Za-z]{2,}", content)]
    if any(word not in _ALLOWED_MATH_WORDS for word in words):
        return None
    return boxed


def _has_plausible_boxed_answer(model_output: str) -> bool:
    return _extract_fast_boxed_answer(model_output) is not None


def _is_plausible_ground_truth(ground_truth: str) -> bool:
    content = ground_truth.strip()
    if not content or len(content) > 256:
        return False
    lowered = content.lower()
    if "[answer]" in lowered or lowered in {"answer", "n/a", "none"}:
        return False
    if any(bad in lowered for bad in ("http", "www.", "text:", "@blank", "projectid")):
        return False
    if not any(ch.isdigit() for ch in content):
        return False
    if not _SAFE_BOXED_CONTENT_PATTERN.match(content):
        return False
    words = [word.lower() for word in re.findall(r"[A-Za-z]{2,}", content)]
    return not any(word not in _ALLOWED_MATH_WORDS for word in words)


def _verify_in_subprocess(ground_truth_boxed: str, model_output: str) -> float:
    """Run math_verify in a subprocess where signal.alarm() works."""
    from math_verify.grader import verify
    from math_verify.parser import LatexExtractionConfig, parse

    gold_targets = (LatexExtractionConfig(),)
    # Only extract from \boxed{} to prevent reward hacking with bare numbers
    pred_targets = (LatexExtractionConfig(),)

    extracted_gold = parse(ground_truth_boxed, gold_targets)
    extracted_pred = parse(model_output, pred_targets)
    if extracted_gold and extracted_pred:
        return max(1.0 if any(verify(g, p) for g in extracted_gold) else 0.0 for p in extracted_pred)
    return 0.0


def compute_score(model_output: str, ground_truth: str, timeout_score: float = 0, timeout: float = 30.0) -> float:
    ret_score = 0.0
    ground_truth_boxed = "\\boxed{" + ground_truth + "}"
    timeout = float(os.getenv("VERL_MATH_VERIFY_TIMEOUT", timeout))
    max_chars = int(os.getenv("VERL_MATH_VERIFY_MAX_CHARS", "0"))
    if max_chars > 0 and len(model_output) > max_chars:
        model_output = model_output[-max_chars:]
    if _env_flag("VERL_MATH_VERIFY_FAST_INVALID", False):
        if not _is_plausible_ground_truth(ground_truth):
            return timeout_score
        boxed_answer = _extract_fast_boxed_answer(model_output)
        if boxed_answer is None:
            return timeout_score
        model_output = boxed_answer
        if _env_flag("VERL_MATH_VERIFY_FAST_EQUIV", False):
            from verl.utils.reward_score.math_reward import is_equiv, remove_boxed

            try:
                return 1.0 if is_equiv(remove_boxed(boxed_answer), ground_truth) else 0.0
            except Exception:
                return timeout_score
    try:
        future = _get_pool().submit(_verify_in_subprocess, ground_truth_boxed, model_output)
        ret_score = future.result(timeout=timeout)
    except (FuturesTimeoutError, TimeoutException):
        ret_score = timeout_score
    except Exception as e:
        print(f"Error in math_verify compute_score: {e}")
    return ret_score
