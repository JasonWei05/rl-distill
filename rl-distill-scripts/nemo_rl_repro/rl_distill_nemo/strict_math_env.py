"""Strict boxed-only math environment for NeMo-RL (verl reward parity).

Ports the STRICT scorer from verl/utils/reward_score/math_verify.py (this repo's
fork): a response scores 1.0 only if it contains exactly ONE \\boxed{}/\\fbox{}
whose content verifies (math_verify, LatexExtractionConfig ONLY — bare numbers
never score) against "\\boxed{ground_truth}". Anything else — no box, multiple
boxes, wrong/malformed content, parser timeout — scores 0.0.

``_BOXED_COMMAND_PATTERN``, ``_all_boxed_contents``, ``_get_pool``,
``_verify_in_subprocess``, and ``compute_score`` are copied verbatim from the
verl file (compute_score with the always-on strict-boxed path inlined and the
FAST_INVALID branch dropped; both are off/strict-on in our runs). Verification
runs in a spawn ProcessPoolExecutor exactly like verl so math-verify's
signal-based timeout works (Ray actors execute tasks off the main thread).

The environment/worker pair mirrors nemo_rl.environments.math_environment
(HFVerifyWorker / MathEnvironment); ``step`` is copied from
``MathEnvironment.step``. Registered as env "math_strict" by run_grpo_repro.py —
zero nemo_rl edits. This module only depends on math_verify, ray, and nemo_rl.
"""

import logging
import multiprocessing
import os
import re
import threading
from concurrent.futures import ProcessPoolExecutor
from concurrent.futures import TimeoutError as FuturesTimeoutError
from typing import Union

import ray
import torch
from math_verify.errors import TimeoutException

from nemo_rl.data.interfaces import LLMMessageLogType
from nemo_rl.environments.interfaces import EnvironmentReturn
from nemo_rl.environments.math_environment import (
    BaseMathEnvironment,
    MathEnvironmentMetadata,
)
from nemo_rl.environments.utils import chunk_list_to_workers

# ---------------------------------------------------------------------------
# Ported verbatim from verl/utils/reward_score/math_verify.py
# ---------------------------------------------------------------------------

_pool = None
_pool_lock = threading.Lock()
_BOXED_COMMAND_PATTERN = re.compile(r"\\(?:boxed|fbox)\s*")


def _get_pool():
    global _pool
    if _pool is None:
        with _pool_lock:
            if _pool is None:
                max_workers = int(os.getenv("VERL_MATH_VERIFY_POOL_WORKERS", "4"))
                _pool = ProcessPoolExecutor(
                    max_workers=max_workers,
                    mp_context=multiprocessing.get_context("spawn"),
                )
    return _pool


def _all_boxed_contents(model_output: str) -> list[str]:
    """Return the contents of every \\boxed{}/\\fbox{} (balanced braces) in order.

    Used for STRICT scoring: score only a single boxed answer, so bare-LaTeX echoes
    (no box) and multi-box hedging get no credit.
    """
    outs = []
    for m in _BOXED_COMMAND_PATTERN.finditer(model_output):
        start = m.end()
        if start >= len(model_output) or model_output[start] != "{":
            continue
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
        if end is not None:
            outs.append(model_output[start + 1 : end].strip())
    return outs


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
    """verl compute_score with the strict-boxed path always on (VERL_MATH_VERIFY_STRICT_BOXED=1)."""
    ret_score = 0.0
    ground_truth_boxed = "\\boxed{" + ground_truth + "}"
    timeout = float(os.getenv("VERL_MATH_VERIFY_TIMEOUT", timeout))
    max_chars = int(os.getenv("VERL_MATH_VERIFY_MAX_CHARS", "0"))
    if max_chars > 0 and len(model_output) > max_chars:
        model_output = model_output[-max_chars:]
    # STRICT boxed-only scoring: require exactly ONE \boxed{} answer and verify only
    # that box. Kills false positives from (a) bare-LaTeX echoes with no box and
    # (b) multi-box hedging.
    boxes = _all_boxed_contents(model_output)
    if len(boxes) != 1:
        return 0.0
    model_output = "\\boxed{" + boxes[0] + "}"
    try:
        future = _get_pool().submit(_verify_in_subprocess, ground_truth_boxed, model_output)
        ret_score = future.result(timeout=timeout)
    except (FuturesTimeoutError, TimeoutException):
        ret_score = timeout_score
    except Exception as e:
        print(f"Error in math_verify compute_score: {e}")
    return ret_score


# ---------------------------------------------------------------------------
# NeMo-RL environment plumbing (mirrors math_environment.HFVerifyWorker /
# MathEnvironment)
# ---------------------------------------------------------------------------


def warm_verify_pool() -> None:
    """Spawn and warm all pool workers up front.

    A spawn child re-imports this module (torch + nemo_rl chain), unlike verl's
    import-light reward module — paid lazily, that import could eat into the 30s
    verify timeout of the first real samples. Submitting max_workers concurrent
    warmups forces every child to spawn and import now.
    """
    max_workers = int(os.getenv("VERL_MATH_VERIFY_POOL_WORKERS", "4"))
    warmups = [
        _get_pool().submit(_verify_in_subprocess, "\\boxed{1}", "\\boxed{1}")
        for _ in range(max_workers)
    ]
    for warmup in warmups:
        warmup.result()


@ray.remote  # pragma: no cover
class StrictBoxedVerifyWorker:
    def __init__(self) -> None:
        logging.getLogger("math_verify").setLevel(logging.CRITICAL)
        warm_verify_pool()

    def verify(
        self,
        pred_responses: list[str],
        ground_truths: list[str],
        return_extracted_answer: bool = False,
        **kwargs,
    ) -> Union[list[float], tuple[list[float], list[str | None]]]:
        """Score each response with the strict verl scorer.

        ``**kwargs`` swallows the ``math_verify_impl`` kwarg the base step()
        forwards; this worker has exactly one implementation.
        """
        results = []
        extracted_answers: list[str | None] = []

        for response, ground_truth in zip(pred_responses, ground_truths):
            results.append(compute_score(response, ground_truth))
            if return_extracted_answer:
                boxes = _all_boxed_contents(response)
                extracted_answers.append(boxes[0] if len(boxes) == 1 else None)

        if return_extracted_answer:
            return results, extracted_answers
        else:
            return results


@ray.remote(
    max_restarts=-1, max_task_retries=-1, max_concurrency=1000
)  # pragma: no cover
class StrictMathEnvironment(BaseMathEnvironment):
    WORKER_CLASS_DICT = {
        "math": StrictBoxedVerifyWorker,
    }

    # step() copied from nemo_rl.environments.math_environment.MathEnvironment.step
    # (only the docstring is shortened) — BaseMathEnvironment provides __init__,
    # shutdown, and global_post_process_and_metrics.
    def step(
        self,
        message_log_batch: list[LLMMessageLogType],
        metadata: list[MathEnvironmentMetadata],
        return_extracted_answer: bool = False,
    ) -> EnvironmentReturn[MathEnvironmentMetadata]:
        """Grade a batch of rollouts against metadata['ground_truth']."""
        # Extract the assistant's responses from the message history
        # Each message list should have at least one assistant response
        assistant_response_batch = []
        for conversation in message_log_batch:
            assistant_responses = [
                str(interaction["content"])
                for interaction in conversation
                if interaction["role"] == "assistant"
            ]
            assistant_response_batch.append("".join(assistant_responses))

        ground_truths = [g["ground_truth"] for g in metadata]

        chunked_assistant_response_batch = chunk_list_to_workers(
            assistant_response_batch, self.num_workers
        )
        chunked_ground_truths = chunk_list_to_workers(ground_truths, self.num_workers)

        # Round-robin the starting worker index.
        # Without the rotation, all the requests will be sent to workers[0] if this function
        # is called per-sample.
        worker_index = next(self._worker_counter) % self.num_workers

        # Process each chunk in parallel
        futures = [
            self.workers[(worker_index + i) % self.num_workers].verify.remote(
                chunk,
                ground_truth_chunk,
                return_extracted_answer,
                math_verify_impl=self.cfg.get("math_verify_impl", "hf_math_verify"),
            )
            for i, (chunk, ground_truth_chunk) in enumerate(
                zip(chunked_assistant_response_batch, chunked_ground_truths)
            )
        ]

        worker_results = ray.get(futures)

        # Flatten the results and extract both scores and answers
        results = []
        extracted_answers: list[str | None] | None = (
            [] if return_extracted_answer else None
        )

        for worker_result in worker_results:
            worker_scores = worker_result
            if return_extracted_answer:
                worker_scores, worker_answers = worker_result
                extracted_answers.extend(worker_answers)
            results.extend(worker_scores)

        observations = [
            {
                "role": "environment",
                "content": "Environment: correct"
                if result
                else "Environment: incorrect",
            }
            for result in results
        ]

        # create a tensor of rewards and done flags
        rewards = torch.tensor(results).cpu()
        done = torch.ones_like(rewards).cpu()
        next_stop_strings = [None] * len(message_log_batch)

        return EnvironmentReturn(
            observations=observations,
            metadata=metadata,
            next_stop_strings=next_stop_strings,
            rewards=rewards,
            terminateds=done,
            answers=extracted_answers,
        )
