#
# Copyright (c) 2025 Huawei Technologies Co., Ltd. All Rights Reserved.
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
# This file is a part of the vllm-ascend project.
#
# Todo: Once https://github.com/vllm-project/vllm/issues/22246 is merged in vllm. Remove this adaptor.
import json
from inspect import getattr_static
from typing import Any

import torch
import torch.distributed as dist
import torch_npu
from vllm.logger import logger

from vllm_ascend.ascend_config import get_ascend_config
from vllm_ascend.quantization.quant_type import QuantType
from vllm_ascend.utils import ACL_FORMAT_FRACTAL_ND, ACL_FORMAT_FRACTAL_NZ

EPLB_EXPERT_WEIGHT_NAMES = {
    (QuantType.NONE, False): ("w13_weight", "w2_weight"),
    (QuantType.NONE, True): ("w13_weight_list", "w2_weight_list"),
    (QuantType.W8A8, False): (
        "w13_weight_list",
        "w2_weight_list",
        "w13_weight_scale_fp32_list",
        "w2_weight_scale_list",
    ),
    (QuantType.W8A8, True): (
        "w13_weight_list",
        "w2_weight_list",
        "w13_weight_scale_fp32_list",
        "w2_weight_scale_list",
        "fused_w1_scale_list",
        "fused_w2_scale_list",
    ),
    (QuantType.W4A8, True): (
        "w13_weight_list",
        "w2_weight_list",
        "w13_weight_scale_list",
        "w2_weight_scale_list",
        "w13_scale_bias_list",
        "w2_scale_bias_list",
    ),
    (QuantType.W4A8, False): (
        "w13_weight_list",
        "w2_weight_list",
        "w13_weight_scale_list",
        "w2_weight_scale_list",
        "w13_scale_bias_list",
        "w2_scale_bias_list",
    ),
    (QuantType.W4A4MXFP, False): ("w13_weight", "w2_weight", "w13_weight_scale", "w2_weight_scale"),
    (QuantType.W4A4MXFP, True): ("w13_weight", "w2_weight", "w13_weight_scale", "w2_weight_scale"),
    (QuantType.W8A8MXFP, False): ("w13_weight", "w2_weight", "w13_weight_scale", "w2_weight_scale"),
    (QuantType.W8A8MXFP, True): ("w13_weight", "w2_weight", "w13_weight_scale", "w2_weight_scale"),
}


def _is_fractal_nz_tensor(tensor: torch.Tensor) -> bool:
    """Return whether *tensor* carries Ascend's internal FRACTAL_NZ layout."""
    if tensor.device.type != "npu":
        return False
    try:
        return int(torch_npu.get_npu_format(tensor)) == ACL_FORMAT_FRACTAL_NZ
    except (AttributeError, RuntimeError, TypeError, ValueError):
        return False


def _has_fractal_nz_storage(tensor: torch.Tensor) -> bool:
    """Return whether a tensor or its owning expert parameter uses NZ storage."""
    if _is_fractal_nz_tensor(tensor):
        return True
    parent, _ = _resolve_expert_parent(tensor)
    return parent is not None and _is_fractal_nz_tensor(parent)


def _empty_like_expert_tensor(tensor: torch.Tensor) -> torch.Tensor:
    """Allocate an EPLB receive buffer with the source tensor's internal format.

    ``torch.empty_like`` drops the NZ metadata on current torch_npu releases.
    HCCL receives the physical NZ tile stream, so an ND buffer is not a valid
    destination for an NZ expert slice.
    """
    if not _has_fractal_nz_storage(tensor):
        return torch.empty_like(tensor)
    try:
        return torch_npu.empty_with_format(
            tuple(tensor.shape),
            dtype=tensor.dtype,
            layout=tensor.layout,
            device=tensor.device,
            pin_memory=False,
            acl_format=ACL_FORMAT_FRACTAL_NZ,
        )
    except (AttributeError, RuntimeError, TypeError):
        # Keep compatibility with older torch_npu builds that do not expose
        # empty_with_format as a Python API.
        return torch_npu.npu_format_cast(torch.empty_like(tensor), ACL_FORMAT_FRACTAL_NZ)


def _copy_expert_tensor(dst: torch.Tensor, src: torch.Tensor) -> None:
    """Copy an EPLB expert row while preserving internal NPU formats."""
    if dst.device.type == "npu" and src.device.type != "npu":
        # Gloo EPLB receives into CPU staging tensors before copying back to
        # the NPU execution buffer.  Convert the source once so the NZ-aware
        # path below can use the same device-side copy primitives.
        src = src.to(device=dst.device)
    if dst.device.type == src.device.type == "npu":
        copy_memory = getattr(torch_npu, "copy_memory_", None)
        if copy_memory is not None:
            try:
                copy_memory(dst, src)
                return
            except RuntimeError:
                # If an internal-format copy is attempted, do not silently
                # fall back to aten.copy_, which is unsupported for NZ on some
                # torch_npu versions.
                if _has_fractal_nz_storage(dst) or _has_fractal_nz_storage(src):
                    dst_parent, dst_index = _resolve_expert_parent(dst)
                    src_parent, src_index = _resolve_expert_parent(src)
                    if dst_parent is None:
                        raise

                    # copy_memory_ requires storage_offset == 0.  An expert
                    # slice other than expert 0 therefore has to be rebuilt
                    # in an offset-0 ND staging tensor and copied back as one
                    # NZ tensor.  The parent parameter's storage is retained,
                    # so captured execution graphs continue to see the update.
                    parent_nd = torch_npu.npu_format_cast(dst_parent, ACL_FORMAT_FRACTAL_ND)
                    dst_row_shape = parent_nd.shape[1:]
                    if src_parent is None:
                        src_row = src
                    elif src_parent is src and src.numel() == parent_nd[dst_index].numel():
                        src_row = torch_npu.npu_format_cast(src, ACL_FORMAT_FRACTAL_ND)
                    else:
                        src_parent_nd = torch_npu.npu_format_cast(src_parent, ACL_FORMAT_FRACTAL_ND)
                        src_row = src_parent_nd[src_index]
                    # NPU does not support copy_, cat, or index updates on
                    # rank-reduced views of an internal-format tensor. Stage
                    # the semantic ND parent on CPU, update one expert row,
                    # then cast the complete parent back to NZ and commit it
                    # through the original offset-0 storage.
                    updated_cpu = parent_nd.cpu()
                    if src_row.device.type == "npu":
                        if _has_fractal_nz_storage(src_row):
                            src_row = torch_npu.npu_format_cast(src_row, ACL_FORMAT_FRACTAL_ND)
                        src_row = src_row.cpu()
                    updated_cpu[dst_index].copy_(src_row.reshape(dst_row_shape).contiguous())
                    updated_nd = updated_cpu.to(device=dst.device)
                    parent_nz = torch_npu.npu_format_cast(updated_nd, ACL_FORMAT_FRACTAL_NZ)
                    copy_memory(dst_parent, parent_nz)
                    return
    dst.copy_(src)


def prepare_expert_tensor_for_send(tensor: torch.Tensor) -> torch.Tensor:
    """Materialize an offset-0 send tensor when an expert is a view of NZ storage."""
    if not _has_fractal_nz_storage(tensor):
        return tensor
    if tensor.storage_offset() == 0:
        return tensor

    parent, expert_index = _resolve_expert_parent(tensor)
    if parent is None:
        if tensor.storage_offset() != 0:
            raise ValueError("An EPLB FRACTAL_NZ send tensor must have storage_offset == 0.")
        return tensor
    parent_nd = torch_npu.npu_format_cast(parent, ACL_FORMAT_FRACTAL_ND)
    expert_nd = parent_nd[expert_index].clone()
    return torch_npu.npu_format_cast(expert_nd, ACL_FORMAT_FRACTAL_NZ)


def _resolve_expert_parent(tensor: torch.Tensor) -> tuple[torch.Tensor | None, int]:
    """Find the owning tensor and expert row for a view of an expert tensor."""
    # Start at the first base tensor.  An expert slice is itself usually a
    # rank-reduced view (for example [K, N]), so treating the slice as its own
    # E-dimensional parent would resolve every update to row zero.
    current = getattr(tensor, "_base", None)
    tensor_offset = tensor.storage_offset()
    while current is not None:
        if current.ndim > tensor.ndim and current.shape[0] > 0:
            stride = current.stride(0)
            offset = tensor_offset - current.storage_offset()
            if stride > 0 and offset >= 0:
                expert_index, remainder = divmod(offset, stride)
                if remainder == 0 and expert_index < current.shape[0]:
                    return current, expert_index
        current = getattr(current, "_base", None)
    return None, -1


class VllmEplbAdaptor:
    _registered_moe_layers: list["torch.nn.Module"] = []

    @staticmethod
    def register_layer(layer: "torch.nn.Module") -> None:
        """Register a MoE layer for EPLB. Called during layer initialization.

        Only real layers call this; PPMissingLayer won't, so the registry
        naturally contains only layers on this PP rank.
        """
        VllmEplbAdaptor._registered_moe_layers.append(layer)

    def __init__(self, model, **args):
        super().__init__(**args)
        if hasattr(model, "language_model"):
            self.model = model.language_model
            self.config = model.config.text_config
        else:
            self.model = model
            self.config = model.config
        self.rank_id = dist.get_rank()
        self.world_size = dist.get_world_size()
        self.num_dense_layers = getattr(self.config, "first_k_dense_replace", 0)

        self.moe_layers = VllmEplbAdaptor._registered_moe_layers
        self.num_moe_layers = len(self.moe_layers)

        self.expert_map_per_layer_cpu = dict()  # copy of expert map on CPU to avoid device synchronize frequently

        # Get num_local_experts from first real MoE layer
        first_layer = self.moe_layers[0]
        self.num_local_experts = first_layer.local_num_experts
        self.ep_rank = first_layer.ep_rank

        self.expert_param_per_layer = dict()
        self.expert_weight_key_per_layer = dict()
        self.init_expert_param_per_layer()

        num_buffer_tensor = self.num_local_experts
        self.buffer_tensor_list: dict[Any, list[list[Any]]] = dict()
        self.init_buffer_tensor(num_buffer_tensor)

        self.log2phy_map_per_layer = dict()
        for local_idx, layer in enumerate(self.moe_layers):
            self.log2phy_map_per_layer[local_idx] = layer.get_log2phy_map()

    def init_buffer_tensor(self, num_buffer_tensor):
        buffer_tensor_shapes: dict[Any, list[torch.Size]] = dict()
        for local_idx, _ in enumerate(self.moe_layers):
            expert_weight_key = self.expert_weight_key_per_layer[local_idx]
            expert_weight_names = EPLB_EXPERT_WEIGHT_NAMES[expert_weight_key]
            expert_tensors = [self.param_dict[f"{local_idx}.{name}"][0] for name in expert_weight_names]
            expert_tensor_shapes = [tensor.shape for tensor in expert_tensors]
            if expert_weight_key in self.buffer_tensor_list:
                assert expert_tensor_shapes == buffer_tensor_shapes[expert_weight_key], (
                    f"EPLB expert weight shapes mismatch for {expert_weight_key}: "
                    f"expected {buffer_tensor_shapes[expert_weight_key]}, got {expert_tensor_shapes}"
                )
                continue
            buffer_tensor_shapes[expert_weight_key] = expert_tensor_shapes
            self.buffer_tensor_list[expert_weight_key] = [[] for _ in range(num_buffer_tensor)]
            for buffer_id in range(num_buffer_tensor):
                for expert_tensor in expert_tensors:
                    buffer_tensor = _empty_like_expert_tensor(expert_tensor)
                    self.buffer_tensor_list[expert_weight_key][buffer_id].append(buffer_tensor)

    def init_expert_param_per_layer(self):
        self.param_dict = dict()

        for local_idx, layer in enumerate(self.moe_layers):
            quant_type = QuantType.NONE if self.model.quant_config is None else layer.quant_type
            expert_weight_key = (quant_type, get_ascend_config().enable_fused_mc2 == 1)
            if expert_weight_key[0] == QuantType.W4A8MXFP:
                raise RuntimeError(f"EPLB not support {quant_type}")
            if expert_weight_key not in EPLB_EXPERT_WEIGHT_NAMES:
                raise ValueError(f"EPLB not support {quant_type} with fused MC2 {expert_weight_key[1]}")
            expert_weight_names = EPLB_EXPERT_WEIGHT_NAMES[expert_weight_key]
            self.expert_weight_key_per_layer[local_idx] = expert_weight_key
            self.expert_param_per_layer[local_idx] = list()
            for name in expert_weight_names:
                param_key = f"{local_idx}.{name}"
                # Inspect the attribute without invoking __getattr__. Besides
                # avoiding side effects from dynamic proxies, this prevents an
                # unspecced MagicMock from fabricating the optional accessor.
                # The refactored MoERunner declares the accessor because its
                # RoutedExperts child owns weights; legacy layers expose them
                # directly.
                get_parameter = (
                    layer.get_eplb_parameter if getattr_static(layer, "get_eplb_parameter", None) is not None else None
                )
                self.param_dict[param_key] = get_parameter(name) if get_parameter is not None else getattr(layer, name)
            for local_expert_id in range(self.num_local_experts):
                per_expert_param = list()
                for name in expert_weight_names:
                    per_expert_param.append(self.param_dict[f"{local_idx}.{name}"][local_expert_id])
                self.expert_param_per_layer[local_idx].append(per_expert_param)

    def get_rank_expert_workload(self) -> torch.Tensor:
        loads = [layer.moe_load for layer in self.moe_layers]
        self.moe_load = torch.stack(loads, dim=0) if loads else torch.empty(0)
        return self.moe_load

    def clear_all_moe_loads(self):
        for layer in self.moe_layers:
            layer.clear_moe_load()

    def _export_tensor_to_file(self, expert_maps, expert_map_record_path: str):
        if self.rank_id == 0:
            num_local_experts = expert_maps.max() + 1

            expert_maps_list = expert_maps.tolist()
            record: dict[str, Any] = {"moe_layer_count": len(expert_maps_list), "layer_list": []}

            for layer_idx, layer_data in enumerate(expert_maps_list):
                layer_record: dict[str, Any] = {
                    "layer_id": layer_idx,
                    "device_count": len(layer_data),
                    "device_list": [],
                }

                for device_idx, experts in enumerate(layer_data):
                    placement = [experts.index(i) for i in range(num_local_experts)]
                    device_record = {"device_id": device_idx, "device_expert": placement}
                    layer_record["device_list"].append(device_record)

                record["layer_list"].append(layer_record)

            with open(expert_map_record_path, "w") as f:
                json.dump(record, f, indent=4)

    def do_update_expert_map(self, layer_id, updated_expert_map):
        self.expert_map_per_layer_cpu[layer_id].copy_(updated_expert_map)

    def do_update_expert_weight(self, layer_id, local_expert_to_replace, buffer_tensor_id):
        expert_weight_key = self.expert_weight_key_per_layer[layer_id]
        for expert_tensor, buffer_tensor in zip(
            self.expert_param_per_layer[layer_id][local_expert_to_replace],
            self.buffer_tensor_list[expert_weight_key][buffer_tensor_id],
        ):
            _copy_expert_tensor(expert_tensor, buffer_tensor)
            logger.debug("Expert tensor shape is :%s", expert_tensor.shape)

    def do_update_log2phy_map(self, layer_id, updated_log2phy_map):
        if self.log2phy_map_per_layer[layer_id] is not None:
            self.log2phy_map_per_layer[layer_id].copy_(updated_log2phy_map)

    def get_global_expert_map(self):
        all_layer_global_expert_map = []
        for local_idx, layer in enumerate(self.moe_layers):
            map_cpu = layer.global_expert_map.cpu()
            all_layer_global_expert_map.append(map_cpu)
            self.expert_map_per_layer_cpu[local_idx] = map_cpu[self.ep_rank]

        return torch.stack(all_layer_global_expert_map)
