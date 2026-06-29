# Mooncake D2D Probe

该目录用于定位 **prefill 节点 device8 -> decode 节点 device8** 的
Mooncake / ADXL / HCCL one-sided read 通路是否正常。

探测脚本会直接调用 `mooncake.engine.TransferEngine`，初始化参数与当前
vLLM Ascend connector 的调用方式保持一致：

- protocol: `P2PHANDSHAKE`
- transport: `ascend`
- `device_name`: 空字符串 `""`

测试方向是 decode 侧通过 Mooncake one-sided read 从 prefill 侧注册的 NPU
buffer 读取数据，并校验读取到的数据模式是否正确。

## 文件说明

| 文件 | 说明 |
| --- | --- |
| `mooncake_d2d_probe.py` | 主探测程序，包含 `server` 和 `client` 两种角色。 |
| `p_mooncake_d2d_probe.sh` | prefill 节点启动脚本，绑定 `ASCEND_RT_VISIBLE_DEVICES=8`，启动 server。 |
| `d_mooncake_d2d_probe.sh` | decode 节点启动脚本，绑定 `ASCEND_RT_VISIBLE_DEVICES=8`，启动 client。 |
| `mooncake_d2d_meta_device8.json` | prefill server 写出的远端 buffer 元信息，decode client 读取该文件后发起 one-sided read。 |

## 启动方式

需要先启动 prefill 侧 server，并保持进程运行；再启动 decode 侧 client。

### 1. 启动 prefill server

在 prefill `80.5.17.106` 节点容器内执行：

```bash
cd /mnt/share/tools
bash p_mooncake_d2d_probe.sh
```

该脚本会：

- 设置 `ASCEND_RT_VISIBLE_DEVICES=8`
- 设置 HCCL / Ascend 调试日志环境变量
- 使用 `--host 80.5.17.106` 初始化 `TransferEngine`
- 在 NPU device8 上分配并注册 64 MiB 源 buffer
- 将远端 session、buffer 地址、字节数等信息写入：

```bash
/mnt/share/tools/mooncake_d2d_meta_device8.json
```

server 启动后会周期性打印类似信息：

```text
alive pid=... session=80.5.17.106:<rpc_port> addr=... bytes=67108864
```

保持该进程运行，不要退出。

### 2. 启动 decode client

在 decode `80.5.17.108` 节点容器内执行：

```bash
cd /mnt/share/tools
bash d_mooncake_d2d_probe.sh
```

该脚本会：

- 设置 `ASCEND_RT_VISIBLE_DEVICES=8`
- 设置 HCCL / Ascend 调试日志环境变量
- 使用 `--host 80.5.17.108` 初始化本地 `TransferEngine`
- 读取 `mooncake_d2d_meta_device8.json`
- 调用 `batch_transfer_sync_read()` 从 prefill 侧 NPU buffer 读取到 decode 侧 NPU buffer
- 校验读取到的数据是否符合预期 pattern

## 预期结果

decode client 正常时，每轮会输出 `status=OK`，最后 `failures=0`：

```text
client local_host=80.5.17.108 local_rpc_port=... remote_session=80.5.17.106:... remote_addr=... local_addr=... bytes=67108864
iter=0 ret=0 elapsed_ms=... status=OK
iter=1 ret=0 elapsed_ms=... status=OK
...
summary iters=50 failures=0 avg_ms=...
```

如果出现以下结果，说明通路或数据正确性异常：

- `status=TRANSFER_FAILED`: `batch_transfer_sync_read()` 返回非 0。
- `status=DATA_MISMATCH`: 传输返回成功，但读取数据与 prefill 侧写入 pattern 不一致。
- `summary ... failures>0`: 本轮探测存在失败。

## 日志位置

prefill 侧脚本日志：

```bash
/mnt/share/log/mooncake_probe_prefill_<timestamp>.log
/mnt/share/log/ascend_log_probe_p_<timestamp>
```

decode 侧脚本日志：

```bash
/mnt/share/log/mooncake_probe_decode_<timestamp>.log
/mnt/share/log/ascend_log_probe_d_<timestamp>
```

## 手工运行

如需绕过 shell 脚本手工执行，可使用以下命令。

prefill server：

```bash
export ASCEND_RT_VISIBLE_DEVICES=8
python3 /mnt/share/tools/mooncake_d2d_probe.py server \
  --host 80.5.17.106 \
  --meta /mnt/share/tools/mooncake_d2d_meta_device8.json \
  --bytes $((64 * 1024 * 1024)) \
  --heartbeat 10
```

decode client：

```bash
export ASCEND_RT_VISIBLE_DEVICES=8
python3 /mnt/share/tools/mooncake_d2d_probe.py client \
  --host 80.5.17.108 \
  --meta /mnt/share/tools/mooncake_d2d_meta_device8.json \
  --iters 50 \
  --interval 0.2
```

## 参数说明

| 参数 | 说明 |
| --- | --- |
| `server` | prefill 侧角色，分配并注册远端 NPU buffer，写出 meta 文件。 |
| `client` | decode 侧角色，读取 meta 文件并发起 one-sided read。 |
| `--host` | 本节点业务 IP，用于初始化 `TransferEngine`。 |
| `--device-name` | Mooncake `device_name`，默认空字符串，保持与 vLLM Ascend connector 一致。 |
| `--meta` | server 写出、client 读取的元信息文件路径。 |
| `--bytes` | 传输 buffer 大小，默认 64 MiB。 |
| `--pattern` | server 写入源 buffer 的 uint8 pattern，默认 `90`。 |
| `--heartbeat` | server 保活日志间隔。 |
| `--iters` | client 传输迭代次数。 |
| `--interval` | client 每轮传输之间的等待时间。 |
| `--check-bytes` | client 每轮校验的前 N 字节数，默认 `4096`。 |

## 排障要点

1. 确认 prefill server 先启动，并持续运行。
2. 确认 decode client 读取到的 `mooncake_d2d_meta_device8.json` 是本次 server 新写出的文件。
3. 确认两侧都设置了 `ASCEND_RT_VISIBLE_DEVICES=8`。
4. 确认 `HCCL_IF_IP` 分别为对应节点 IP：
   - prefill: `80.5.17.106`
   - decode: `80.5.17.108`
5. 确认 `HCCL_SOCKET_IFNAME`、`GLOO_SOCKET_IFNAME`、`TP_SOCKET_IFNAME` 与当前网络接口一致。
6. 如果 client 返回 `TRANSFER_FAILED`，优先查看两侧 Mooncake / HCCL / Ascend 日志。
7. 如果 client 返回 `DATA_MISMATCH`，说明读请求完成但数据内容不符合预期，需要继续排查 device-to-device read 数据路径。
