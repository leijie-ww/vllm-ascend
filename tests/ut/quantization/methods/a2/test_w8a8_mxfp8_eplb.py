import pytest
import torch
import torch_npu
from vllm.distributed.eplb import rebalance_execute
from vllm.distributed.eplb.rebalance_execute import TransferMetadata, move_from_buffer

from vllm_ascend.device.hardware import AscendDeviceType, device_type_from_runtime_soc
from vllm_ascend.eplb.adaptor.vllm_adaptor import (
    VllmEplbAdaptor,
    _copy_expert_tensor,
    _empty_like_expert_tensor,
    prepare_expert_tensor_for_send,
)
from vllm_ascend.quantization.methods.w8a8.w8a8_mxfp8 import AscendW8A8MXFP8DynamicFusedMoEMethod
from vllm_ascend.quantization.quant_type import QuantType
from vllm_ascend.utils import ACL_FORMAT_FRACTAL_ND, ACL_FORMAT_FRACTAL_NZ


def _is_a5_runtime() -> bool:
    try:
        if not torch.npu.is_available():
            return False
        return device_type_from_runtime_soc(torch.npu.get_soc_version()) == AscendDeviceType.A5
    except (AttributeError, RuntimeError, TypeError, ValueError):
        return False


pytestmark = pytest.mark.skipif(not _is_a5_runtime(), reason="W8A8 MXFP8 requires Ascend 950")

_WEIGHT_NAMES = (
    "w13_weight",
    "w2_weight",
    "w13_weight_scale",
    "w2_weight_scale",
)


def _fp8_tensor(shape: tuple[int, ...]) -> torch.Tensor:
    return torch.randint(0, 255, shape, dtype=torch.uint8, device="npu").view(torch.float8_e4m3fn)


def _byte_equal(left: torch.Tensor, right: torch.Tensor) -> bool:
    return torch.equal(left.view(torch.uint8).cpu(), right.view(torch.uint8).cpu())


def _as_nd(tensor: torch.Tensor) -> torch.Tensor:
    if int(torch_npu.get_npu_format(tensor)) == ACL_FORMAT_FRACTAL_NZ:
        return torch_npu.npu_format_cast(tensor, ACL_FORMAT_FRACTAL_ND)
    return tensor


def _make_processed_layer() -> torch.nn.Module:
    num_experts = 2
    hidden_size = 64
    intermediate_size = 128
    group_size = 32
    layer = torch.nn.Module()
    layer.w13_weight = torch.nn.Parameter(
        _fp8_tensor((num_experts, 2 * intermediate_size, hidden_size)), requires_grad=False
    )
    layer.w2_weight = torch.nn.Parameter(
        _fp8_tensor((num_experts, hidden_size, intermediate_size)), requires_grad=False
    )
    layer.w13_weight_scale = torch.nn.Parameter(
        torch.randint(
            0,
            255,
            (num_experts, 2 * intermediate_size, hidden_size // group_size),
            dtype=torch.uint8,
            device="npu",
        ),
        requires_grad=False,
    )
    layer.w2_weight_scale = torch.nn.Parameter(
        torch.randint(
            0,
            255,
            (num_experts, hidden_size, intermediate_size // group_size),
            dtype=torch.uint8,
            device="npu",
        ),
        requires_grad=False,
    )
    method = AscendW8A8MXFP8DynamicFusedMoEMethod.__new__(AscendW8A8MXFP8DynamicFusedMoEMethod)
    method.process_weights_after_loading(layer)
    layer.quant_method = method
    return layer


def test_mxfp8_eplb_migration_updates_nz_execution_weights_in_place():
    layer = _make_processed_layer()
    views = layer.quant_method.get_eplb_weight_views(layer)

    for name, view in zip(_WEIGHT_NAMES, views):
        parameter = getattr(layer, name)
        assert view is parameter
        assert view.view(2, -1).untyped_storage().data_ptr() == parameter.untyped_storage().data_ptr()
    assert int(torch_npu.get_npu_format(layer.w13_weight)) == ACL_FORMAT_FRACTAL_NZ
    assert int(torch_npu.get_npu_format(layer.w2_weight)) == ACL_FORMAT_FRACTAL_NZ
    send_tensor = prepare_expert_tensor_for_send(layer.w13_weight[1])
    assert send_tensor is not layer.w13_weight[1]
    assert send_tensor.storage_offset() == 0
    assert int(torch_npu.get_npu_format(send_tensor)) == ACL_FORMAT_FRACTAL_NZ
    source_nd = torch_npu.npu_format_cast(layer.w13_weight, ACL_FORMAT_FRACTAL_ND)
    send_nd = torch_npu.npu_format_cast(send_tensor, ACL_FORMAT_FRACTAL_ND)
    assert _byte_equal(send_nd, source_nd[1])

    key = (QuantType.W8A8MXFP, False)
    adaptor = VllmEplbAdaptor.__new__(VllmEplbAdaptor)
    adaptor.moe_layers = [layer]
    adaptor.expert_weight_key_per_layer = {0: key}
    adaptor.param_dict = {f"0.{name}": getattr(layer, name) for name in _WEIGHT_NAMES}
    adaptor.buffer_tensor_list = {}
    adaptor.init_buffer_tensor(2)
    adaptor.expert_param_per_layer = {
        0: [[getattr(layer, name)[expert_id] for name in _WEIGHT_NAMES] for expert_id in range(2)]
    }

    buffers = adaptor.buffer_tensor_list[key][0]
    assert all(int(torch_npu.get_npu_format(buffer)) == ACL_FORMAT_FRACTAL_NZ for buffer in buffers[:2])

    original_first_experts = [_as_nd(getattr(layer, name))[0].cpu() for name in _WEIGHT_NAMES]
    expected_second_experts = []
    for target, buffer in zip(adaptor.expert_param_per_layer[0][1], buffers):
        if target.dtype == torch.float8_e4m3fn:
            payload_nd = _fp8_tensor(tuple(target.shape))
            payload = torch_npu.npu_format_cast(payload_nd, ACL_FORMAT_FRACTAL_NZ)
        else:
            payload_nd = torch.randint(0, 255, target.shape, dtype=target.dtype, device="npu")
            payload = payload_nd
        torch_npu.copy_memory_(buffer, payload)
        expected_second_experts.append(payload_nd.cpu())

    data_ptrs = [getattr(layer, name).data_ptr() for name in _WEIGHT_NAMES]
    adaptor.do_update_expert_weight(layer_id=0, local_expert_to_replace=1, buffer_tensor_id=0)
    torch.npu.synchronize()

    assert [getattr(layer, name).data_ptr() for name in _WEIGHT_NAMES] == data_ptrs
    for name, original_first, expected_second in zip(_WEIGHT_NAMES, original_first_experts, expected_second_experts):
        actual = _as_nd(getattr(layer, name))
        assert _byte_equal(actual[0], original_first)
        assert _byte_equal(actual[1], expected_second)


def test_upstream_eplb_move_from_buffer_updates_flattened_nz_views():
    """Exercise the V2 EPLB move path against real Ascend NZ storage."""
    layer = _make_processed_layer()
    weights = [weight.view(2, -1) for weight in AscendW8A8MXFP8DynamicFusedMoEMethod.get_eplb_weight_views(layer)]
    buffers = [_empty_like_expert_tensor(weight) for weight in weights]

    expected_second = []
    raw_weights = AscendW8A8MXFP8DynamicFusedMoEMethod.get_eplb_weight_views(layer)
    for raw_weight, weight, buffer in zip(raw_weights, weights, buffers):
        payload_nd = (
            _fp8_tensor(tuple(weight.shape))
            if weight.dtype == torch.float8_e4m3fn
            else torch.randint(0, 255, weight.shape, dtype=weight.dtype, device="npu")
        )
        payload = (
            torch_npu.npu_format_cast(payload_nd, ACL_FORMAT_FRACTAL_NZ)
            if int(torch_npu.get_npu_format(buffer)) == ACL_FORMAT_FRACTAL_NZ
            else payload_nd
        )
        torch_npu.copy_memory_(buffer, payload)
        expected_second.append(payload_nd[1].reshape(raw_weight.shape[1:]).cpu())

    rebalance_execute.set_eplb_tensor_hooks(
        empty_like=_empty_like_expert_tensor,
        copy_tensor=_copy_expert_tensor,
        prepare_send=prepare_expert_tensor_for_send,
    )
    metadata = TransferMetadata(
        is_unchanged=torch.tensor([True, False], dtype=torch.bool).cpu().numpy(),
        is_received_locally=torch.tensor([True, True], dtype=torch.bool).cpu().numpy(),
        recv_primary_mask=torch.tensor([False, False], dtype=torch.bool).cpu().numpy(),
        recv_count=0,
        recv_expert_ids=torch.full((2,), -1, dtype=torch.int64).numpy(),
        recv_dst_rows=torch.full((2,), -1, dtype=torch.int32).numpy(),
    )
    data_ptrs = [
        layer_weight.data_ptr()
        for layer_weight in AscendW8A8MXFP8DynamicFusedMoEMethod.get_eplb_weight_views(layer)
    ]
    move_from_buffer(weights, buffers, metadata, new_indices=torch.tensor([0, 1]).numpy(), ep_rank=0)
    torch.npu.synchronize()

    assert [
        layer_weight.data_ptr()
        for layer_weight in AscendW8A8MXFP8DynamicFusedMoEMethod.get_eplb_weight_views(layer)
    ] == data_ptrs
    for raw_weight, expected in zip(raw_weights, expected_second):
        actual = _as_nd(raw_weight)
        assert _byte_equal(actual[1], expected)


def test_gloo_cpu_staging_copy_updates_nonzero_nz_expert_slice():
    """The V2 Gloo receive path must write CPU staging data into NZ storage."""
    layer = _make_processed_layer()
    target = layer.w13_weight[1]
    payload = _fp8_tensor(tuple(target.shape)).cpu()

    _copy_expert_tensor(target, payload)
    torch.npu.synchronize()

    actual = _as_nd(layer.w13_weight)[1]
    assert _byte_equal(actual, payload)
