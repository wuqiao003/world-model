#!/bin/bash
set -e
export NVIDIA_DRIVER_CAPABILITIES=compute,utility
export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0,1,3,4,5,7}
echo "HOST=$(hostname) POD_IP=${POD_IP}"
/usr/bin/nvidia-smi --query-gpu=index,memory.used,utilization.gpu --format=csv
echo "==== PY ===="
/usr/bin/python3 - <<'PY'
import sys
print(sys.version)
try:
    import torch
    print("torch", torch.__version__)
    print("cuda_available", torch.cuda.is_available())
    print("device_count", torch.cuda.device_count())
    if torch.cuda.is_available():
        print("name0", torch.cuda.get_device_name(0))
        x = torch.randn(2048, 2048, device="cuda")
        y = (x @ x).mean().item()
        print("matmul_ok", y)
except Exception as e:
    print("TORCH_ERR", type(e).__name__, e)
PY
echo "==== MNT ===="
ls /mnt/group/jxdong 2>/dev/null || true
ls /mnt/public/Wan2.2_models 2>/dev/null || true
df -h /mnt/group /mnt/public /tmp 2>/dev/null | head
