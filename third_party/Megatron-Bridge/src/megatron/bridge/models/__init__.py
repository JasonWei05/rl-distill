# Copyright (c) 2025, NVIDIA CORPORATION.  All rights reserved.
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

import importlib
import logging

from megatron.bridge.models.conversion.auto_bridge import AutoBridge
from megatron.bridge.models.conversion.mapping_registry import MegatronMappingRegistry
from megatron.bridge.models.conversion.model_bridge import MegatronModelBridge
from megatron.bridge.models.conversion.param_mapping import (
    AutoMapping,
    ColumnParallelMapping,
    GatedMLPMapping,
    MegatronParamMapping,
    QKVMapping,
    ReplicatedMapping,
    RowParallelMapping,
)
from megatron.bridge.models.gemma import (
    Gemma2ModelProvider,
    Gemma3ModelProvider,
    Gemma3ModelProvider4B,
    Gemma3MoEModelProvider,
    Gemma3MoEModelProvider4B,
    Gemma3MoEModelProvider4B4E,
    GemmaModelProvider,
)
from megatron.bridge.models.gpt_provider import GPTModelProvider
from megatron.bridge.models.t5_provider import T5ModelProvider


logger = logging.getLogger(__name__)

__all__ = [
    "AutoBridge",
    "MegatronMappingRegistry",
    "MegatronModelBridge",
    "ColumnParallelMapping",
    "GatedMLPMapping",
    "MegatronParamMapping",
    "QKVMapping",
    "ReplicatedMapping",
    "RowParallelMapping",
    "AutoMapping",
    "GemmaModelProvider",
    "Gemma2ModelProvider",
    "Gemma3ModelProvider",
    "Gemma3ModelProvider4B",
    "Gemma3MoEModelProvider",
    "Gemma3MoEModelProvider4B",
    "Gemma3MoEModelProvider4B4E",
    "GPTModelProvider",
    "T5ModelProvider",
]

# Optional model families. Some require Megatron-LM APIs newer than the
# installed megatron-core (this fork runs against core_v0.16.0); those are
# skipped instead of breaking every megatron.bridge import.
_OPTIONAL_MODEL_IMPORTS = {
    "megatron.bridge.models.bailing": ["BailingMoeV2Bridge"],
    "megatron.bridge.models.deepseek": ["DeepSeekV2Bridge", "DeepSeekV3Bridge"],
    "megatron.bridge.models.ernie": ["Ernie45Bridge"],
    "megatron.bridge.models.ernie_vl": ["Ernie45VLBridge", "Ernie45VLModel", "Ernie45VLModelProvider"],
    "megatron.bridge.models.falcon_h1": ["FalconH1Bridge", "FalconH1ModelProvider"],
    "megatron.bridge.models.gemma_vl": [
        "Gemma3VLBridge",
        "Gemma3VLModel",
        "Gemma3VLModelProvider",
        "Gemma4VLBridge",
        "Gemma4VLModel",
        "Gemma4VLModelProvider",
    ],
    "megatron.bridge.models.glm": ["GLM45Bridge", "GLM47FlashBridge"],
    "megatron.bridge.models.glm_moe_dsa": ["GLM5Bridge"],
    "megatron.bridge.models.glm_vl": ["GLM45VBridge", "GLM45VModelProvider"],
    "megatron.bridge.models.gpt_oss": ["GPTOSSBridge"],
    "megatron.bridge.models.kimi": ["KimiK2Bridge"],
    "megatron.bridge.models.kimi_vl": ["KimiK25VLBridge", "KimiK25VLModel", "KimiK25VLModelProvider"],
    "megatron.bridge.models.llama": ["LlamaBridge"],
    "megatron.bridge.models.llama_nemotron": ["LlamaNemotronBridge", "LlamaNemotronHeterogeneousProvider"],
    "megatron.bridge.models.mamba.mamba_provider": ["MambaModelProvider"],
    "megatron.bridge.models.mimo.mimo_bridge": ["MimoBridge"],
    "megatron.bridge.models.mimo_v2_flash": ["MiMoV2FlashBridge", "MiMoV2FlashModelProvider"],
    "megatron.bridge.models.minimax_m2": ["MiniMaxM2Bridge"],
    "megatron.bridge.models.ministral3": ["Ministral3Bridge", "Ministral3Model", "Ministral3ModelProvider"],
    "megatron.bridge.models.mistral": ["MistralModelProvider"],
    "megatron.bridge.models.nemotron": ["NemotronBridge"],
    "megatron.bridge.models.nemotron_omni": ["NemotronOmniBridge", "NemotronOmniModel"],
    "megatron.bridge.models.nemotron_vl": ["NemotronVLBridge", "NemotronVLModel", "NemotronVLModelProvider"],
    "megatron.bridge.models.nemotronh": ["NemotronHBridge"],
    "megatron.bridge.models.olmoe": ["OlMoEBridge", "OlMoEModelProvider"],
    "megatron.bridge.models.qwen3_asr": ["Qwen3ASRBridge", "Qwen3ASRModel", "Qwen3ASRModelProvider"],
    "megatron.bridge.models.qwen_audio": ["Qwen2AudioBridge", "Qwen2AudioModel", "Qwen2AudioModelProvider"],
    "megatron.bridge.models.qwen_omni": [
        "Qwen3OmniBridge",
        "Qwen3OmniModel",
        "Qwen3OmniModelProvider",
        "Qwen25OmniBridge",
        "Qwen25OmniModel",
        "Qwen25OmniModelProvider",
    ],
    "megatron.bridge.models.qwen_vl": [
        "Qwen25VLBridge",
        "Qwen25VLModel",
        "Qwen25VLModelProvider",
        "Qwen35VLBridge",
        "Qwen35VLModelProvider",
        "Qwen35VLMoEBridge",
        "Qwen35VLMoEModelProvider",
    ],
    "megatron.bridge.models.qwen_vl.modelling_qwen3_vl": [
        "Qwen3VLBridge",
        "Qwen3VLModel",
        "Qwen3VLModelProvider",
        "Qwen3VLMoEBridge",
        "Qwen3VLMoEModelProvider",
    ],
    "megatron.bridge.models.sarvam": ["SarvamMLABridge", "SarvamMoEBridge"],
    "megatron.bridge.models.stepfun": ["Step35Bridge", "Step37Bridge", "Step37Model", "Step37ModelProvider"],
}

for _module_name, _symbols in _OPTIONAL_MODEL_IMPORTS.items():
    try:
        _module = importlib.import_module(_module_name)
    except ImportError as exc:
        logger.debug("Skipping %s (incompatible with installed megatron-core): %s", _module_name, exc)
        continue
    for _symbol in _symbols:
        globals()[_symbol] = getattr(_module, _symbol)
        __all__.append(_symbol)
