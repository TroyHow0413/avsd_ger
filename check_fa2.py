import torch
import sys
import subprocess
import os

print("="*50)
print(" Flash Attention 2 环境检测工具 ")
print("="*50)

# 1. Python 版本
print(f"\n[1] Python 版本: {sys.version}")
if sys.version_info >= (3, 8) and sys.version_info <= (3, 11):
    print("✅ Python 版本 OK")
else:
    print("❌ Python 版本 不支持 (需要 3.8~3.11)")

# 2. PyTorch 版本
print(f"\n[2] PyTorch 版本: {torch.__version__}")
if torch.__version__ >= "2.0":
    print("✅ PyTorch >= 2.0 OK")
else:
    print("❌ PyTorch 太旧，需要 >= 2.0")

# 3. PyTorch CUDA 版本
pt_cuda = torch.version.cuda
print(f"\n[3] PyTorch 编译的 CUDA: {pt_cuda}")

# 4. 系统 CUDA 版本
try:
    out = subprocess.check_output(["nvcc", "--version"]).decode()
    sys_cuda = out.split("release ")[-1].split(",")[0]
    print(f"[4] 系统 CUDA Toolkit: {sys_cuda}")
except:
    sys_cuda = "未安装"
    print(f"[4] 系统 CUDA Toolkit: ❌ 未安装 (必须安装!)")

if sys_cuda != "未安装" and sys_cuda == pt_cuda:
    print("✅ CUDA 版本完全匹配！")
else:
    print("❌ CUDA 版本不匹配 → flash-attn 无法编译")

# 5. GPU 架构
cap = torch.cuda.get_device_capability()
cap_score = cap[0] + cap[1]/10
print(f"\n[5] GPU 算力: {cap_score}")
if cap_score >= 8.0:
    print("✅ GPU 支持 Flash Attention 2")
else:
    print("❌ GPU 太旧，不支持")

# 6. GCC 版本
try:
    out = subprocess.check_output(["g++", "--version"]).decode()
    gcc_version = out.split("\n")[0].split()[-1]
    print(f"\n[6] GCC 版本: {gcc_version}")
    if gcc_version.startswith("13"):
        print("❌ GCC 13 太高！CUDA 12.0 只支持 <13 的版本")
    elif float(gcc_version[:2]) >= 6:
        print("✅ GCC 版本 OK")
except:
    print("\n[6] GCC 未安装")

print("\n" + "="*50)