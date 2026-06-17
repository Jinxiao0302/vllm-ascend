#
# Copyright (c) 2025 Huawei Technologies Co., Ltd. All Rights Reserved.
# This file is a part of the vllm-ascend project.
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
#

from collections.abc import Callable
from typing import Any

import torch
import torch_npu
from vllm.config import CompilationMode, get_current_vllm_config

from vllm_ascend.ascend_config import get_ascend_config
from vllm_ascend.ascend_forward_context import _EXTRA_CTX
from vllm_ascend.flash_common3_context import get_flash_common3_context
from vllm_ascend.ops.fused_moe.experts_selector import select_experts
from vllm_ascend.ops.fused_moe.moe_runtime_args import build_fused_experts_input

from .base import AscendLinearScheme, AscendMoEScheme, QuantType, get_moe_num_logical_experts
from .registry import register_scheme


@register_scheme("W8A16_FP8", "linear")
class AscendW8A16FP8LinearMethod(AscendLinearScheme):
    """Linear method for Ascend W8A16-FP8 weight-only quantization.

    This scheme stores weights in FP8 (float8_e4m3fn) format with per-channel
    symmetric scaling, while activations remain in 16-bit precision.
    
    Implementation rationale:
    - Atlas 910B's npu_dynamic_quant does NOT support float8_e4m3fn output [^20^].
    - Therefore we use software dequantization + standard matmul for correctness.
    - Weight is dequantized on-the-fly: w_fp16 = w_fp8 * scale (per-channel).
    """

    def get_weight(self, input_size: int, output_size: int, 
                   params_dtype: torch.dtype = torch.bfloat16) -> dict[str, Any]:
        return {
            "weight": torch.empty(output_size, input_size, dtype=torch.float8_e4m3fn)
        }

    def get_perchannel_param(self, output_size: int, 
                              params_dtype: torch.dtype) -> dict[str, Any]:
        return {
            "weight_scale": torch.empty(output_size, 1, dtype=torch.float32)
        }

    def apply(self, layer: torch.nn.Module, x: torch.Tensor,
              bias: torch.Tensor | None = None, tp_rank: int | None = 0) -> torch.Tensor:
        # Software dequantization: FP8 weight -> FP16/BF16 with per-channel scale
        weight_fp16 = layer.weight.to(x.dtype) * layer.weight_scale.to(x.dtype)
        output = torch.matmul(x, weight_fp16.t())
        if bias is not None:
            output = output + bias
        return output

    def process_weights_after_loading(self, layer):
        # Transpose to column-major for matmul efficiency
        layer.weight.data = layer.weight.data.transpose(0, 1).contiguous()
        layer.weight_scale.data = torch.flatten(layer.weight_scale.data)


@register_scheme("W8A16_FP8", "moe")
class AscendW8A16FP8FusedMoEMethod(AscendMoEScheme):
    """FusedMoE method for Ascend W8A16-FP8.

    Uses NPU antiquant hardware path via npu_grouped_matmul:
    - Activations remain in FP16/BF16 (no quantization)
    - Weights are dequantized on-the-fly via antiquant_scale + antiquant_offset
    - antiquant_offset is zero tensor (symmetric quantization)
    
    Supports: fused_mc2, zero_experts, dynamic_eplb, multistream_overlap_gate.
    """

    quant_type: QuantType = QuantType.W8A16FP8  # 需先在枚举中定义

    def __init__(self):
        vllm_config = get_current_vllm_config()
        ascend_config = get_ascend_config()
        self.use_aclgraph = (
            vllm_config.compilation_config.mode == CompilationMode.VLLM_COMPILE
            and not vllm_config.model_config.enforce_eager
        )
        self.multistream_overlap_gate = ascend_config.multistream_overlap_gate
        self.dynamic_eplb = ascend_config.eplb_config.dynamic_eplb
        self.in_dtype = vllm_config.model_config.dtype

    def get_weight(self, num_experts: int, intermediate_size_per_partition: int,
                   hidden_sizes: int, params_dtype: torch.dtype) -> dict[str, Any]:
        return {
            "w13_weight": torch.empty(
                num_experts, 2 * intermediate_size_per_partition, hidden_sizes,
                dtype=torch.float8_e4m3fn
            ),
            "w2_weight": torch.empty(
                num_experts, hidden_sizes, intermediate_size_per_partition,
                dtype=torch.float8_e4m3fn
            ),
        }

    def get_dynamic_quant_param(self, num_experts: int,
                                 intermediate_size_per_partition: int,
                                 hidden_sizes: int, params_dtype: torch.dtype) -> dict[str, Any]:
        """Return weight scales and zero offsets for antiquant path."""
        return {
            "w13_weight_scale": torch.empty(
                num_experts, 2 * intermediate_size_per_partition, 1, dtype=torch.float32
            ),
            # Zero offset triggers antiquant path in npu_grouped_matmul
            "w13_weight_offset": torch.zeros(
                num_experts, 2 * intermediate_size_per_partition, 1, dtype=params_dtype
            ),
            "w2_weight_scale": torch.empty(
                num_experts, hidden_sizes, 1, dtype=torch.float32
            ),
            "w2_weight_offset": torch.zeros(
                num_experts, hidden_sizes, 1, dtype=params_dtype
            ),
        }

    def apply(self, layer, x, router_logits, top_k, renormalize, **kwargs) -> torch.Tensor:
        # Expert selection (same as scheme2)
        num_shared_experts = getattr(layer, "n_shared_experts", 0) or 0
        zero_expert_num = getattr(layer, "zero_expert_num", 0)
        zero_expert_type = getattr(layer, "zero_expert_type", None)
        mix_placement = getattr(layer, "mix_placement", False)
        
        num_logical_experts = get_moe_num_logical_experts(
            layer, kwargs.get("num_experts", -1),
            global_redundant_expert_num=kwargs.get("global_redundant_expert_num", 0),
            num_shared_experts=num_shared_experts,
        )

        # Select experts
        if self.multistream_overlap_gate:
            fc3_context = get_flash_common3_context()
            topk_weights, topk_ids = fc3_context.topk_weights, fc3_context.topk_ids
        else:
            topk_weights, topk_ids = select_experts(
                hidden_states=x, router_logits=router_logits, top_k=top_k,
                renormalize=renormalize, **{k: v for k, v in kwargs.items()
                if k in ["use_grouped_topk", "topk_group", "num_expert_group",
                         "custom_routing_function", "scoring_func",
                         "routed_scaling_factor", "e_score_correction_bias", "tid2eid"]}
            )

        # Zero experts handling
        if zero_expert_num > 0 and zero_expert_type is not None:
            topk_ids, topk_weights, zero_expert_result = zero_experts_compute(
                expert_indices=topk_ids, expert_scales=topk_weights,
                num_experts=num_logical_experts, zero_expert_type=zero_expert_type,
                hidden_states=x,
            )

        # Force load balance
        if kwargs.get("enable_force_load_balance", False):
            random_matrix = torch.rand(topk_ids.size(0), num_logical_experts, device=topk_ids.device)
            topk_ids = torch.argsort(random_matrix, dim=1)[:, :topk_ids.size(1)].to(topk_ids.dtype)

        topk_weights = topk_weights.to(self.in_dtype)

        # Fused MC2 scale preparation
        moe_comm_method = _EXTRA_CTX.moe_comm_method
        fused_scale_flag = (
            _EXTRA_CTX.moe_comm_type == MoECommType.FUSED_MC2
            and get_ascend_config().enable_fused_mc2 == 1
        )

        if self.dynamic_eplb:
            w1 = layer.w13_weight_list
            w1_scale = layer.fused_w1_scale_list if fused_scale_flag else layer.w13_weight_scale_fp32_list
            w2 = layer.w2_weight_list
            w2_scale = layer.fused_w2_scale_list if fused_scale_flag else layer.w2_weight_scale_list
        else:
            w1 = [layer.w13_weight]
            w1_scale = [layer.fused_w1_scale] if fused_scale_flag else [layer.w13_weight_scale_fp32]
            w2 = [layer.w2_weight]
            w2_scale = [layer.fused_w2_scale] if fused_scale_flag else [layer.w2_weight_scale]

        w1_scale_bias = [torch.tensor([], dtype=torch.float32)] if fused_scale_flag else None
        w2_scale_bias = [torch.tensor([], dtype=torch.float32)] if fused_scale_flag else None

        # Call fused_experts with W8A16FP8 quant_type
        final_hidden_states = moe_comm_method.fused_experts(
            fused_experts_input=build_fused_experts_input(
                hidden_states=x,
                topk_weights=topk_weights,
                topk_ids=topk_ids,
                w1=w1,
                w2=w2,
                quant_type=self.quant_type,       # W8A16FP8
                dynamic_eplb=self.dynamic_eplb,
                expert_map=kwargs.get("expert_map"),
                global_redundant_expert_num=kwargs.get("global_redundant_expert_num", 0),
                mc2_mask=kwargs.get("mc2_mask"),
                apply_router_weight_on_input=kwargs.get("apply_router_weight_on_input", False),
                log2phy=kwargs.get("log2phy"),
                pertoken_scale=kwargs.get("pertoken_scale"),
                activation=kwargs.get("activation", "silu"),
                w1_scale=w1_scale,
                w2_scale=w2_scale,
                w1_scale_bias=w1_scale_bias,
                w2_scale_bias=w2_scale_bias,
                # Key: w1_offset triggers antiquant path (activation NOT quantized)
                w1_offset=layer.w13_weight_offset,
                w2_offset=layer.w2_weight_offset,
                swiglu_limit=getattr(layer, "swiglu_limit", 0.0),
            )
        )

        if zero_expert_num > 0 and zero_expert_type is not None:
            final_hidden_states += zero_expert_result

        return final_hidden_states

    def process_weights_after_loading(self, layer):
        # Transpose for grouped matmul: (E, K, N) -> (E, N, K)
        layer.w13_weight.data = layer.w13_weight.data.transpose(1, 2).contiguous()
        layer.w2_weight.data = layer.w2_weight.data.transpose(1, 2).contiguous()
        
        # Convert to Fractal NZ format for optimal NPU performance
        layer.w13_weight.data = torch_npu.npu_format_cast(
            layer.w13_weight.data, ACL_FORMAT_FRACTAL_NZ
        )
        layer.w2_weight.data = torch_npu.npu_format_cast(
            layer.w2_weight.data, ACL_FORMAT_FRACTAL_NZ
        )
        
        # Flatten scales per expert
        layer.w13_weight_scale.data = layer.w13_weight_scale.data.view(
            layer.w13_weight_scale.data.shape[0], -1
        )
        layer.w13_weight_scale_fp32 = layer.w13_weight_scale.data.to(torch.float32)
        layer.w13_weight_offset.data = layer.w13_weight_offset.data.view(
            layer.w13_weight_offset.data.shape[0], -1
        )
        layer.w2_weight_scale.data = layer.w2_weight_scale.data.view(
            layer.w2_weight_scale.data.shape[0], -1
        )
        layer.w2_weight_scale_fp32 = layer.w2_weight_scale.data.to(torch.float32)
        layer.w2_weight_offset.data = layer.w2_weight_offset.data.view(
            layer.w2_weight_offset.data.shape[0], -1
        )

        # Fused MC2: convert scale to int64 representation
        if get_ascend_config().enable_fused_mc2 == 1:
            layer.fused_w1_scale = scale_from_float_to_int64(layer.w13_weight_scale.data)
            layer.fused_w2_scale = scale_from_float_to_int64(layer.w2_weight_scale.data)

        # Dynamic EPLB: split expert weights into lists
        if self.dynamic_eplb:
            layer.w13_weight_list = [w.clone() for w in layer.w13_weight.data.unbind(dim=0)]
            layer.w2_weight_list = [w.clone() for w in layer.w2_weight.data.unbind(dim=0)]
            layer.w13_weight_scale_fp32_list = [
                w.clone() for w in layer.w13_weight_scale_fp32.data.unbind(dim=0)
            ]
            layer.w2_weight_scale_list = [
                w.clone() for w in layer.w2_weight_scale.data.unbind(dim=0)
            ]
            if get_ascend_config().enable_fused_mc2 == 1:
                layer.fused_w1_scale_list = [
                    w.clone() for w in layer.fused_w1_scale.view(
                        len(layer.w13_weight_list), -1
                    ).data.unbind(dim=0)
                ]
                layer.fused_w2_scale_list = [
                    w.clone() for w in layer.fused_w2_scale.view(
                        len(layer.w2_weight_list), -1
                    ).data.unbind(dim=0)
                ]
            # Free original tensors to save memory
            del layer.w13_weight, layer.w2_weight
            del layer.w13_weight_scale, layer.w13_weight_scale_fp32
            del layer.w2_weight_scale
            if get_ascend_config().enable_fused_mc2 == 1:
                del layer.fused_w1_scale, layer.fused_w2_scale
            torch.npu.empty_cache()