#!/usr/bin/env python3
"""Download a Gemma 3 PT model and patch its tokenizer with the IT chat template.

The Gemma 3 PT (pretrained/base) models don't ship with a chat template since
they're meant as foundations for fine-tuning. For RL training we need a chat
template, so we copy it from the IT (instruction-tuned) variant.

We also remove the strict role-alternation check from the IT template, since
verl's `initialize_system_prompt` calls apply_chat_template with two consecutive
user messages, which would fail Gemma's default validation.

Usage:
    python3 setup_model.py --size 4b
    python3 setup_model.py --size 12b
    python3 setup_model.py --size 4b --output-dir /custom/path
"""

import argparse
import os
from pathlib import Path

from huggingface_hub import snapshot_download
from transformers import AutoTokenizer


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument(
        "--size",
        required=True,
        choices=["4b", "12b", "27b"],
        help="Gemma 3 model size (4b, 12b, or 27b)",
    )
    parser.add_argument(
        "--variant",
        default="pt",
        choices=["pt", "it"],
        help="Model variant: pt (pretrained/base) or it (instruction-tuned). Default: pt",
    )
    parser.add_argument(
        "--output-dir",
        default=None,
        help="Where to save the model (default: $HOME/verl/models/gemma-3-{size}-{variant})",
    )
    parser.add_argument(
        "--it-source",
        default="google/gemma-3-4b-it",
        help="IT model to copy the chat template from (default: google/gemma-3-4b-it). "
             "All Gemma 3 IT models share the same template, so 4b-it works for any size. "
             "Only used when --variant=pt.",
    )
    args = parser.parse_args()

    repo = f"google/gemma-3-{args.size}-{args.variant}"
    if args.output_dir is None:
        args.output_dir = os.path.expanduser(f"~/verl/models/gemma-3-{args.size}-{args.variant}")
    output_dir = Path(args.output_dir)

    print(f"=== Downloading {repo} to {output_dir} ===")
    snapshot_download(repo, local_dir=str(output_dir))

    if args.variant == "pt":
        print(f"\n=== Copying chat template from {args.it_source} ===")
        it_tok = AutoTokenizer.from_pretrained(args.it_source)
        pt_tok = AutoTokenizer.from_pretrained(str(output_dir))
    else:
        print(f"\n=== Patching IT chat template (remove strict role check) ===")
        # IT model already has the template; just patch it for verl compatibility
        it_tok = AutoTokenizer.from_pretrained(str(output_dir))
        pt_tok = it_tok

    # Use the IT chat template, but remove the strict role-alternation check
    # since verl's initialize_system_prompt sends two consecutive user messages.
    template = it_tok.chat_template
    strict_check = (
        "    {%- if (message['role'] == 'user') != (loop.index0 % 2 == 0) -%}\n"
        '        {{ raise_exception("Conversation roles must alternate user/assistant/user/assistant/...") }}\n'
        "    {%- endif -%}\n"
    )
    if strict_check in template:
        template = template.replace(strict_check, "")
        print("Removed strict role-alternation check from chat template")
    else:
        print("WARNING: Did not find expected strict-check block in IT template — IT may have updated their template")

    pt_tok.chat_template = template
    pt_tok.save_pretrained(str(output_dir))

    print("\n=== Verification ===")
    tok = AutoTokenizer.from_pretrained(str(output_dir))
    print("chat_template set:", tok.chat_template is not None)

    # Normal usage
    r1 = tok.apply_chat_template(
        [{"role": "user", "content": "What is 2+2?"}], tokenize=False, add_generation_prompt=True
    )
    print(f"Normal prompt: {r1!r}")

    # Edge cases verl needs
    r2 = tok.apply_chat_template(
        [{"role": "user", "content": ""}], tokenize=False, add_generation_prompt=False
    )
    r3 = tok.apply_chat_template(
        [{"role": "user", "content": ""}] * 2, tokenize=False, add_generation_prompt=False
    )
    print(f"Empty user (system_prompt init): OK (len={len(r2)})")
    print(f"Two empty users (system_prompt init): OK (len={len(r3)})")

    print(f"\n=== Done ===")
    print(f"Model ready at: {output_dir}")
    print(f"Use it in your training script with:")
    print(f"  MODEL_PATH={output_dir}")


if __name__ == "__main__":
    main()
