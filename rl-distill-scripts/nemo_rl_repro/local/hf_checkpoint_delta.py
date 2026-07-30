#!/usr/bin/env python3
# Copyright 2026 Bytedance Ltd. and/or its affiliates
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

"""Create and reconstruct exact sparse deltas between BF16 safetensors files."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
from typing import BinaryIO

import numpy as np
import zstandard as zstd

LEGACY_CODEC = "sparse-u16-add-zstd-v1"
CODEC = "sparse-zigzag-varint-zstd-v2"
DEFAULT_CHUNK_BYTES = 32 * 1024 * 1024
DEFAULT_ZSTD_LEVEL = 19


def _read_exact(stream: BinaryIO, size: int) -> bytes:
    chunks: list[bytes] = []
    remaining = size
    while remaining:
        chunk = stream.read(remaining)
        if not chunk:
            raise EOFError(f"Expected {size} bytes, got {size - remaining}")
        chunks.append(chunk)
        remaining -= len(chunk)
    return b"".join(chunks)


def _encode_zigzag_varints(additions: np.ndarray) -> bytes:
    """Encode signed int16 additions as ordered one-to-three-byte varints."""

    signed = additions.astype(np.int32, copy=False)
    zigzag = np.where(signed >= 0, signed * 2, -signed * 2 - 1).astype(
        np.uint32,
        copy=False,
    )
    lengths = (1 + (zigzag >= 1 << 7) + (zigzag >= 1 << 14)).astype(
        np.int64,
        copy=False,
    )
    starts = np.empty(len(zigzag), dtype=np.int64)
    if len(starts) == 0:
        return b""
    starts[0] = 0
    if len(starts) > 1:
        np.cumsum(lengths[:-1], out=starts[1:])

    encoded = np.empty(int(lengths.sum()), dtype=np.uint8)
    encoded[starts] = (zigzag & 0x7F).astype(np.uint8)
    has_second = lengths >= 2
    encoded[starts[has_second]] |= 0x80
    encoded[starts[has_second] + 1] = ((zigzag[has_second] >> 7) & 0x7F).astype(np.uint8)
    has_third = lengths == 3
    encoded[starts[has_third] + 1] |= 0x80
    encoded[starts[has_third] + 2] = (zigzag[has_third] >> 14).astype(np.uint8)
    return encoded.tobytes()


def _decode_zigzag_varints(encoded: bytes, count: int) -> np.ndarray:
    """Decode exactly ``count`` ordered varints into modular uint16 additions."""

    if count == 0:
        if encoded:
            raise ValueError("Varint stream contains data for an empty chunk")
        return np.empty(0, dtype=np.uint16)

    raw = np.frombuffer(encoded, dtype=np.uint8)
    endings = np.flatnonzero((raw & 0x80) == 0)
    if len(endings) != count or endings[-1] != len(raw) - 1:
        raise ValueError("Varint stream does not contain the expected number of values")

    starts = np.empty(count, dtype=np.int64)
    starts[0] = 0
    if count > 1:
        starts[1:] = endings[:-1] + 1
    lengths = endings - starts + 1
    if int(lengths.max()) > 3:
        raise ValueError("BF16 modular additions must fit in at most three varint bytes")

    zigzag = (raw[starts] & 0x7F).astype(np.uint32)
    has_second = lengths >= 2
    zigzag[has_second] |= (raw[starts[has_second] + 1] & 0x7F).astype(np.uint32) << 7
    has_third = lengths == 3
    zigzag[has_third] |= raw[starts[has_third] + 2].astype(np.uint32) << 14
    signed = (zigzag >> 1).astype(np.int32) ^ -(zigzag & 1).astype(np.int32)
    if np.any(signed < np.iinfo(np.int16).min) or np.any(signed > np.iinfo(np.int16).max):
        raise ValueError("Decoded modular addition is outside the int16 range")
    return signed.astype(np.int16).view(np.uint16)


def create_delta(
    base_path: Path,
    target_path: Path,
    output_dir: Path,
    *,
    chunk_bytes: int = DEFAULT_CHUNK_BYTES,
    zstd_level: int = DEFAULT_ZSTD_LEVEL,
) -> dict[str, int | str | float]:
    """Write an exact sparse modular-add delta and return its manifest."""

    base_path = base_path.resolve()
    target_path = target_path.resolve()
    if base_path.stat().st_size != target_path.stat().st_size:
        raise ValueError("Base and target files must have identical sizes")
    target_size = target_path.stat().st_size
    if target_size % 2:
        raise ValueError("The BF16 safetensors file size must be divisible by two")
    if chunk_bytes <= 0 or chunk_bytes % 2:
        raise ValueError("chunk_bytes must be a positive even integer")

    output_dir.mkdir(parents=True, exist_ok=True)
    mask_path = output_dir / "changed_mask.bitset.zst"
    values_path = output_dir / "add_values.u16.zst"
    mask_compressor = zstd.ZstdCompressor(level=zstd_level, threads=4)
    values_compressor = zstd.ZstdCompressor(level=zstd_level, threads=4)
    base_hash = hashlib.sha256()
    target_hash = hashlib.sha256()
    word_count = 0
    changed_word_count = 0

    with (
        base_path.open("rb") as base_file,
        target_path.open("rb") as target_file,
        mask_path.open("wb") as mask_raw,
        values_path.open("wb") as values_raw,
        mask_compressor.stream_writer(mask_raw, closefd=False) as mask_output,
        values_compressor.stream_writer(values_raw, closefd=False) as values_output,
    ):
        while base_bytes := base_file.read(chunk_bytes):
            target_bytes = _read_exact(target_file, len(base_bytes))
            base_hash.update(base_bytes)
            target_hash.update(target_bytes)
            base_words = np.frombuffer(base_bytes, dtype="<u2")
            target_words = np.frombuffer(target_bytes, dtype="<u2")
            changed = base_words != target_words
            additions = np.subtract(
                target_words[changed],
                base_words[changed],
                dtype=np.uint16,
            )
            mask_output.write(np.packbits(changed, bitorder="little").tobytes())
            encoded_additions = _encode_zigzag_varints(additions.view(np.int16))
            values_output.write(len(encoded_additions).to_bytes(4, "little"))
            values_output.write(encoded_additions)
            word_count += len(changed)
            changed_word_count += int(changed.sum())
        if target_file.read(1):
            raise ValueError("Target file contains trailing bytes")

    return {
        "codec": CODEC,
        "chunk_bytes": chunk_bytes,
        "zstd_level": zstd_level,
        "target_bytes": target_size,
        "word_count": word_count,
        "changed_word_count": changed_word_count,
        "changed_fraction": changed_word_count / word_count,
        "value_encoding": "signed-int16-zigzag-varint",
        "mask_bytes": mask_path.stat().st_size,
        "value_bytes": values_path.stat().st_size,
        "delta_bytes": mask_path.stat().st_size + values_path.stat().st_size,
        "base_sha256": base_hash.hexdigest(),
        "target_sha256": target_hash.hexdigest(),
    }


def reconstruct_delta(
    base_path: Path,
    delta_dir: Path,
    output_path: Path,
    manifest: dict[str, int | str | float],
) -> None:
    """Reconstruct and verify the exact target safetensors file."""

    codec = manifest.get("codec")
    if codec not in {CODEC, LEGACY_CODEC}:
        raise ValueError(f"Unsupported delta codec: {codec}")
    chunk_bytes = int(manifest["chunk_bytes"])
    target_bytes = int(manifest["target_bytes"])
    if base_path.stat().st_size != target_bytes:
        raise ValueError("Base file size does not match the delta manifest")

    output_path.parent.mkdir(parents=True, exist_ok=True)
    temp_path = output_path.with_suffix(output_path.suffix + ".tmp")
    mask_decompressor = zstd.ZstdDecompressor()
    values_decompressor = zstd.ZstdDecompressor()
    output_hash = hashlib.sha256()
    changed_word_count = 0

    try:
        with (
            base_path.open("rb") as base_file,
            (delta_dir / "changed_mask.bitset.zst").open("rb") as mask_raw,
            (delta_dir / "add_values.u16.zst").open("rb") as values_raw,
            mask_decompressor.stream_reader(mask_raw) as mask_input,
            values_decompressor.stream_reader(values_raw) as values_input,
            temp_path.open("wb") as output_file,
        ):
            while base_bytes := base_file.read(chunk_bytes):
                chunk_word_count = len(base_bytes) // 2
                packed_mask = _read_exact(mask_input, (chunk_word_count + 7) // 8)
                changed = np.unpackbits(
                    np.frombuffer(packed_mask, dtype=np.uint8),
                    bitorder="little",
                    count=chunk_word_count,
                ).astype(bool, copy=False)
                count = int(changed.sum())
                if codec == LEGACY_CODEC:
                    additions = np.frombuffer(_read_exact(values_input, count * 2), dtype="<u2")
                else:
                    encoded_size = int.from_bytes(_read_exact(values_input, 4), "little")
                    additions = _decode_zigzag_varints(
                        _read_exact(values_input, encoded_size),
                        count,
                    )
                reconstructed = np.frombuffer(bytearray(base_bytes), dtype="<u2")
                reconstructed[changed] = np.add(
                    reconstructed[changed],
                    additions,
                    dtype=np.uint16,
                )
                output_bytes = reconstructed.tobytes()
                output_hash.update(output_bytes)
                output_file.write(output_bytes)
                changed_word_count += count

            if mask_input.read(1) or values_input.read(1):
                raise ValueError("Delta streams contain trailing data")

        if changed_word_count != int(manifest["changed_word_count"]):
            raise ValueError("Reconstructed changed-word count does not match the manifest")
        if output_hash.hexdigest() != manifest["target_sha256"]:
            raise ValueError("Reconstructed target SHA-256 does not match the manifest")
        os.replace(temp_path, output_path)
    finally:
        temp_path.unlink(missing_ok=True)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--base", type=Path, required=True)
    parser.add_argument("--delta-dir", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    manifest = json.loads(args.manifest.read_text())
    reconstruct_delta(args.base, args.delta_dir, args.output, manifest)
    print(f"Reconstructed and verified {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
