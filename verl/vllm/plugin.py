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

"""Register verl's vLLM-native model implementations."""


def register_gemma3_moe_model() -> None:
    """Make the Gemma3 dense-to-MoE architecture available to vLLM workers.

    Use a lazy class path so vLLM can inspect the registry in a subprocess
    without importing CUDA-dependent model code in the parent process.
    """
    from vllm import ModelRegistry

    ModelRegistry.register_model(
        "Gemma3MoeForCausalLM",
        "verl.vllm.gemma3_moe:Gemma3MoeForCausalLM",
    )
