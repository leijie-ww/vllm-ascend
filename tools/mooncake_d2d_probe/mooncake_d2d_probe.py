#!/usr/bin/env python3
import argparse
import json
import os
import socket
import sys
import time
from pathlib import Path


def _import_runtime():
    try:
        import torch
        from mooncake.engine import TransferEngine
    except Exception as exc:
        print(f"failed to import torch/mooncake runtime: {exc!r}", file=sys.stderr)
        raise
    return torch, TransferEngine


def _set_device(torch):
    if not hasattr(torch, "npu"):
        raise RuntimeError("torch.npu is not available in this Python environment")
    torch.npu.set_device(0)
    return torch.device("npu:0")


def _init_engine(TransferEngine, host: str, device_name: str):
    engine = TransferEngine()
    ret = engine.initialize(host, "P2PHANDSHAKE", "ascend", device_name)
    if ret != 0:
        raise RuntimeError(
            f"TransferEngine.initialize({host!r}, 'P2PHANDSHAKE', 'ascend', {device_name!r}) failed: {ret}"
        )
    return engine


def _register(engine, ptr: int, nbytes: int, name: str):
    ret = engine.register_memory(ptr, nbytes)
    if ret != 0:
        raise RuntimeError(f"register_memory({name}, ptr={ptr}, nbytes={nbytes}) failed: {ret}")


def run_server(args):
    torch, TransferEngine = _import_runtime()
    dev = _set_device(torch)
    engine = _init_engine(TransferEngine, args.host, args.device_name)

    src = torch.empty(args.bytes, dtype=torch.uint8, device=dev)
    src.fill_(args.pattern & 0xFF)
    torch.npu.synchronize()

    _register(engine, src.data_ptr(), src.numel() * src.element_size(), "src")
    rpc_port = int(engine.get_rpc_port())
    meta = {
        "host": args.host,
        "rpc_port": rpc_port,
        "session": f"{args.host}:{rpc_port}",
        "addr": int(src.data_ptr()),
        "bytes": int(src.numel() * src.element_size()),
        "pattern": int(args.pattern & 0xFF),
        "hostname": socket.gethostname(),
        "pid": os.getpid(),
        "visible_devices": os.getenv("ASCEND_RT_VISIBLE_DEVICES", ""),
        "device_name": args.device_name,
        "created_at": time.strftime("%Y-%m-%d %H:%M:%S"),
    }
    Path(args.meta).write_text(json.dumps(meta, indent=2, sort_keys=True) + "\n")
    print(json.dumps(meta, indent=2, sort_keys=True), flush=True)
    print("server is holding the registered NPU buffer; keep this process running during client test", flush=True)

    try:
        while True:
            time.sleep(args.heartbeat)
            print(f"alive pid={os.getpid()} session={meta['session']} addr={meta['addr']} bytes={meta['bytes']}", flush=True)
    except KeyboardInterrupt:
        print("server stopped by KeyboardInterrupt", flush=True)


def run_client(args):
    torch, TransferEngine = _import_runtime()
    dev = _set_device(torch)
    engine = _init_engine(TransferEngine, args.host, args.device_name)

    meta = json.loads(Path(args.meta).read_text())
    session = meta["session"]
    src_addr = int(meta["addr"])
    nbytes = int(args.bytes or meta["bytes"])
    expected = int(meta.get("pattern", args.pattern)) & 0xFF

    dst = torch.empty(nbytes, dtype=torch.uint8, device=dev)
    _register(engine, dst.data_ptr(), dst.numel() * dst.element_size(), "dst")

    print(
        f"client local_host={args.host} local_rpc_port={engine.get_rpc_port()} "
        f"remote_session={session} remote_addr={src_addr} local_addr={int(dst.data_ptr())} bytes={nbytes}",
        flush=True,
    )

    failures = 0
    total_ms = 0.0
    for i in range(args.iters):
        dst.zero_()
        torch.npu.synchronize()
        t0 = time.perf_counter()
        ret = engine.batch_transfer_sync_read(session, [src_addr], [int(dst.data_ptr())], [nbytes])
        torch.npu.synchronize()
        elapsed_ms = (time.perf_counter() - t0) * 1000.0
        total_ms += elapsed_ms
        if ret != 0:
            failures += 1
            print(f"iter={i} ret={ret} elapsed_ms={elapsed_ms:.3f} status=TRANSFER_FAILED", flush=True)
            continue
        sample = dst[: min(nbytes, args.check_bytes)].cpu()
        ok = bool((sample == expected).all().item())
        status = "OK" if ok else "DATA_MISMATCH"
        if not ok:
            failures += 1
        print(f"iter={i} ret={ret} elapsed_ms={elapsed_ms:.3f} status={status}", flush=True)
        if args.interval > 0:
            time.sleep(args.interval)

    avg_ms = total_ms / max(args.iters, 1)
    print(f"summary iters={args.iters} failures={failures} avg_ms={avg_ms:.3f}", flush=True)
    return 1 if failures else 0


def main():
    parser = argparse.ArgumentParser(description="Mooncake ascend device-to-device read probe")
    sub = parser.add_subparsers(dest="role", required=True)

    common = argparse.ArgumentParser(add_help=False)
    common.add_argument("--host", required=True, help="local business IP, e.g. 80.5.17.106 or 80.5.17.108")
    common.add_argument("--device-name", default="", help="Mooncake device_name. Empty string matches vLLM-Ascend connector.")
    common.add_argument("--meta", default="/mnt/share/tools/mooncake_d2d_meta.json")
    common.add_argument("--bytes", type=int, default=64 * 1024 * 1024)
    common.add_argument("--pattern", type=int, default=90)

    p_server = sub.add_parser("server", parents=[common])
    p_server.add_argument("--heartbeat", type=float, default=30.0)

    p_client = sub.add_parser("client", parents=[common])
    p_client.add_argument("--iters", type=int, default=20)
    p_client.add_argument("--interval", type=float, default=0.2)
    p_client.add_argument("--check-bytes", type=int, default=4096)

    args = parser.parse_args()
    if args.role == "server":
        run_server(args)
        return 0
    if args.role == "client":
        return run_client(args)
    raise AssertionError(args.role)


if __name__ == "__main__":
    raise SystemExit(main())
