ts=$(date +%m%d%H%M%S)
unset http_proxy https_proxy ftp_proxy

export ASCEND_RT_VISIBLE_DEVICES=8
export ASCEND_PROCESS_LOG_PATH=/mnt/share/log/ascend_log_probe_p_$ts
export ASCEND_HOST_LOG_FILE_NUM=50
export ASCEND_GLOBAL_LOG_LEVEL=0
export ASCEND_SLOG_PRINT_TO_STDOUT=0
export HCCL_LOG_LEVEL=0
export HCCL_DEBUG=DEBUG
export HCCL_DEBUG_LEVEL=3
export HCCL_IF_IP=80.5.17.106
export HCCL_SOCKET_IFNAME=enp48s3u1u1
export GLOO_SOCKET_IFNAME=enp48s3u1u1
export TP_SOCKET_IFNAME=enp48s3u1u1
export ASCEND_CONNECT_TIMEOUT=10000
export ASCEND_TRANSFER_TIMEOUT=10000

python3 /mnt/share/tools/mooncake_d2d_probe.py server \
  --host 80.5.17.106 \
  --meta /mnt/share/tools/mooncake_d2d_meta_device8.json \
  --bytes $((64 * 1024 * 1024)) \
  --heartbeat 10 \
  | tee /mnt/share/log/mooncake_probe_prefill_$ts.log
