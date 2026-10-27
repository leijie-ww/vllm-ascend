"""C8 MXFP bulk MemCache admission, layout and key compatibility."""

import ctypes
from dataclasses import replace
from types import SimpleNamespace

import pytest

import tests.ut.distributed.ascend_store._mock_deps  # noqa: F401, E402

import torch
from vllm.v1.kv_cache_interface import FullAttentionSpec, KVCacheGroupSpec, MambaSpec, UniformTypeKVCacheSpecs

from vllm_ascend.ascend_config import AscendConfig
from vllm_ascend.attention.attention_c8_mxfp import mxfp_packet_size_bytes, mxfp_paged_cache_views
from vllm_ascend.distributed.kv_transfer.kv_pool.ascend_store.backend.memcache_backend import (
    MemcacheBackend,
    MmcDirect,
)
from vllm_ascend.distributed.kv_transfer.kv_pool.ascend_store.coordinator import _uses_reachable_mask
from vllm_ascend.distributed.kv_transfer.kv_pool.ascend_store.metadata import (
    _C8_MXFP_CACHE_FAMILY,
    ChunkedTokenDatabase,
    KeyMetadata,
    PoolKey,
    infer_group_cache_families,
)
from vllm_ascend.distributed.kv_transfer.kv_pool.ascend_store.pool_worker import KVPoolWorker
from vllm_ascend.utils import uses_c8_mxfp_sync_memcache


def _transfer(connector="AscendStoreConnector", role="kv_both", **extra):
    return SimpleNamespace(
        kv_connector=connector,
        kv_role=role,
        kv_connector_extra_config={"backend": "memcache", **extra},
    )


def _config(transfer, dtype="mxfp8"):
    return SimpleNamespace(cache_config=SimpleNamespace(cache_dtype=dtype), kv_transfer_config=transfer)


def test_direct_sync_bulk_memcache_is_admitted():
    transfer = _transfer()
    assert uses_c8_mxfp_sync_memcache(transfer)
    AscendConfig._check_c8_mxfp_kv_transfer(_config(transfer))
    AscendConfig._check_c8_mxfp_kv_transfer(_config(None))
    AscendConfig._check_c8_mxfp_kv_transfer(_config(_transfer("OtherConnector"), dtype="auto"))
    AscendConfig._check_c8_mxfp_kv_transfer(_config(_transfer("MooncakeConnectorV1")))


@pytest.mark.parametrize(
    "transfer",
    [
        _transfer("MultiConnector"),
        _transfer("MultiConnector", connectors=[{"kv_connector": "MooncakeConnectorV1"}]),
        _transfer("MooncakeConnectorStoreV1"),
        _transfer(role="kv_consumer"),
        _transfer(backend="mooncake"),
        _transfer(use_layerwise=True),
        _transfer(load_async=True),
        _transfer(discard_partial_chunks=False),
        _transfer(save_decode_cache=True),
        SimpleNamespace(
            kv_connector="AscendStoreConnector",
            kv_role="kv_both",
            kv_connector_module_path="custom.connector",
            kv_connector_extra_config={"backend": "memcache"},
        ),
    ],
)
def test_unsupported_c8_transfer_is_rejected_at_config_init(transfer):
    assert not uses_c8_mxfp_sync_memcache(transfer)
    with pytest.raises(ValueError, match="C8_MXFP KV transfer requires"):
        AscendConfig._check_c8_mxfp_kv_transfer(_config(transfer))


def test_only_c8_full_attention_keys_receive_layout_version():
    attention = FullAttentionSpec(block_size=2048, num_kv_heads=4, head_size=256, dtype=torch.float8_e4m3fn)
    state = MambaSpec(block_size=2048, shapes=((8,),), dtypes=(torch.float32,), mamba_cache_mode="align")
    groups = [
        KVCacheGroupSpec(["model.layers.0.attn"], attention),
        KVCacheGroupSpec(["model.layers.0.mamba"], state),
        KVCacheGroupSpec(
            ["model.mtp.0.attn"],
            UniformTypeKVCacheSpecs(block_size=2048, kv_cache_specs={"model.mtp.0.attn": attention}),
        ),
    ]
    assert infer_group_cache_families(groups, None, c8_mxfp=True) == [
        _C8_MXFP_CACHE_FAMILY,
        "default",
        _C8_MXFP_CACHE_FAMILY,
    ]
    assert infer_group_cache_families(groups, None) == ["default"] * 3
    assert infer_group_cache_families(None, None, c8_mxfp=True) == [_C8_MXFP_CACHE_FAMILY]
    assert _uses_reachable_mask(_C8_MXFP_CACHE_FAMILY)
    old_meta = KeyMetadata("model", 0, 0, 0)
    new_meta = replace(old_meta, cache_family=_C8_MXFP_CACHE_FAMILY)
    assert PoolKey(old_meta, "hash").to_string() != PoolKey(new_meta, "hash").to_string()


def _worker_for_c8_views(raw):
    worker = object.__new__(KVPoolWorker)
    worker.num_blocks = 3
    worker.use_layerwise = False
    worker.hf_config = SimpleNamespace(num_hidden_layers=1)
    worker.num_layers = 1
    worker.kv_caches = {
        "model.layers.0.attn": mxfp_paged_cache_views(
            raw[137:-97], 12, 4, 256, 256, 512, page_stride_bytes=mxfp_packet_size_bytes(4, 512, 256, 256) + 256
        )
    }
    worker._global_to_local_layer = {}
    for name in (
        "group_kv_caches_base_addr",
        "group_block_len",
        "group_block_stride",
        "group_layer_cache_entry_offsets",
        "group_cache_entry_segments",
        "group_num_layers",
    ):
        setattr(worker, name, {})
    worker._infer_cache_group_metadata(0, list(worker.kv_caches))
    database = ChunkedTokenDatabase([], [2048], None)
    database.set_group_buffers(
        worker.group_kv_caches_base_addr,
        worker.group_block_len,
        worker.group_block_stride,
        group_cache_families={0: _C8_MXFP_CACHE_FAMILY},
        group_num_layers=worker.group_num_layers,
        group_layer_cache_entry_offsets=worker.group_layer_cache_entry_offsets,
        group_cache_entry_segments=worker.group_cache_entry_segments,
    )
    return worker, database


def test_four_view_padded_scheduler_page_roundtrip_and_partial_rejection():
    packet = mxfp_packet_size_bytes(4, 512, 256, 256)
    pitch = packet + 256
    total = 137 + 12 * pitch + 97
    source = torch.arange(total, dtype=torch.int32).remainder(251).to(torch.int8)
    target = torch.full((total,), -7, dtype=torch.int8)
    source_worker, source_db = _worker_for_c8_views(source)
    target_worker, target_db = _worker_for_c8_views(target)
    source_addresses, source_sizes, _ = source_db.prepare_value(2048, 4096, [0, 1])
    target_addresses, target_sizes, _ = target_db.prepare_value(2048, 4096, [0, 1])
    assert len(source_addresses) == len(target_addresses) == 16
    assert source_sizes == target_sizes == source_worker.group_block_len[0] == target_worker.group_block_len[0]
    assert sum(source_sizes) == 4 * packet == 4_325_376

    class InMemoryStore:
        def __init__(self):
            self.objects = {}

        def batch_put_from_layers(self, keys, addresses, sizes, direction):
            assert direction == MmcDirect.COPY_L2G.value
            for key, key_addresses, key_sizes in zip(keys, addresses, sizes):
                self.objects[key] = [
                    ctypes.string_at(address, size) for address, size in zip(key_addresses, key_sizes)
                ]
            return [0] * len(keys)

        def batch_get_into_layers(self, keys, addresses, sizes, direction):
            assert direction == MmcDirect.COPY_G2L.value
            for key, key_addresses, key_sizes in zip(keys, addresses, sizes):
                payloads = self.objects[key]
                assert key_sizes == [len(payload) for payload in payloads]
                for address, payload in zip(key_addresses, payloads):
                    ctypes.memmove(address, payload, len(payload))
            return [0] * len(keys)

    backend = object.__new__(MemcacheBackend)
    backend.store = InMemoryStore()
    backend._lazy_init = False
    backend._store_initialized = True
    assert backend.put(["c8-page"], [source_addresses], [source_sizes]) is True
    assert backend.get(["c8-page"], [target_addresses], [target_sizes]) == [0]
    expected = torch.full_like(target, -7)
    for physical_block in range(4, 8):
        start = 137 + physical_block * pitch
        expected[start : start + packet] = source[start : start + packet]
    assert torch.equal(target, expected)
    with pytest.raises(ValueError, match="complete aligned scheduler block"):
        source_db.prepare_value(2048, 4095, [0, 1])
    with pytest.raises(ValueError, match="complete aligned scheduler block"):
        target_db.prepare_value(2049, 4097, [0, 1])
