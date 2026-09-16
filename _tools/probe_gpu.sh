#!/bin/bash
set -e
export NVIDIA_DRIVER_CAPABILITIES=compute,utility
export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0,1,3,4,5,7}
echo "HOST=$(hostname)"
echo "PWD=$(pwd)"
which nvidia-smi || ls -la /usr/bin/nvidia-smi /usr/local/nvidia/bin/nvidia-smi 2>/dev/null
NVIDIA_SMI=$(command -v nvidia-smi || echo /usr/local/nvidia/bin/nvidia-smi)
$NVIDIA_SMI --query-gpu=index,memory.used,utilization.gpu --format=csv
echo "==== TORCH ===="
PY=${PY:-/root/miniforge3/bin/python}
$PY - <<'PY'
import torch
print("torch", torch.__version__)
print("cuda_available", torch.cuda.is_available())
print("device_count", torch.cuda.device_count())
if torch.cuda.is_available():
    print("name0", torch.cuda.get_device_name(0))
    x = torch.randn(1024, 1024, device="cuda")
    y = x @ x
    print("matmul_ok", float(y.mean()))
PY
