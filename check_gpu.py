#!/usr/bin/env python3
"""
GPU 診斷腳本 - 檢查 RTX 4070 是否正確配置
"""
import sys

print("=" * 70)
print("🔍 GPU 診斷工具")
print("=" * 70)

# 1. 檢查 PyTorch
print("\n1️⃣ 檢查 PyTorch...")
try:
    import torch
    print(f"   ✅ PyTorch 版本: {torch.__version__}")
    print(f"   ✅ CUDA 可用: {torch.cuda.is_available()}")
    
    if torch.cuda.is_available():
        print(f"   ✅ CUDA 版本: {torch.version.cuda}")
        print(f"   ✅ GPU 數量: {torch.cuda.device_count()}")
        for i in range(torch.cuda.device_count()):
            props = torch.cuda.get_device_properties(i)
            print(f"   ✅ GPU {i}: {props.name}")
            print(f"      記憶體: {props.total_memory / 1024**3:.2f} GB")
            print(f"      計算能力: {props.major}.{props.minor}")
        
        # 測試 GPU 計算
        print("\n   🧪 測試 GPU 計算...")
        x = torch.randn(1000, 1000).cuda()
        y = torch.randn(1000, 1000).cuda()
        z = torch.matmul(x, y)
        print(f"   ✅ GPU 計算測試成功！")
    else:
        print("   ❌ PyTorch 無法檢測到 CUDA")
        print("\n   💡 解決方案：")
        print("      1. 檢查是否安裝了 PyTorch CPU 版本")
        print("      2. 重新安裝 PyTorch CUDA 版本：")
        print("         conda install pytorch torchvision torchaudio pytorch-cuda=12.1 -c pytorch -c nvidia")
        print("      或")
        print("         pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121")
        
except ImportError:
    print("   ❌ PyTorch 未安裝")
    sys.exit(1)

# 2. 檢查 accelerate
print("\n2️⃣ 檢查 Accelerate 庫...")
try:
    from accelerate import Accelerator
    accel = Accelerator()
    print(f"   ✅ Accelerate 已安裝")
    
    # 檢查 accelerate 配置
    import os
    config_path = os.path.expanduser("~/.cache/huggingface/accelerate/default_config.yaml")
    if os.path.exists(config_path):
        print(f"   ✅ Accelerate 配置文件存在: {config_path}")
        with open(config_path, 'r') as f:
            print(f"   📄 配置內容:\n{f.read()}")
    else:
        print(f"   ⚠️ Accelerate 配置文件不存在")
        print(f"   💡 運行 'accelerate config' 來配置")
        
except ImportError:
    print("   ❌ Accelerate 未安裝")
    print("   💡 安裝: pip install accelerate")

# 3. 檢查 NVIDIA 驅動
print("\n3️⃣ 檢查 NVIDIA 驅動...")
import subprocess
try:
    result = subprocess.run(['nvidia-smi'], capture_output=True, text=True, timeout=5)
    if result.returncode == 0:
        print("   ✅ NVIDIA 驅動正常")
        print(f"\n{result.stdout}")
    else:
        print("   ❌ nvidia-smi 執行失敗")
except FileNotFoundError:
    print("   ❌ nvidia-smi 未找到（NVIDIA 驅動可能未安裝）")
except subprocess.TimeoutExpired:
    print("   ⚠️ nvidia-smi 執行超時")

# 4. 檢查 diffusers
print("\n4️⃣ 檢查 Diffusers 庫...")
try:
    import diffusers
    print(f"   ✅ Diffusers 版本: {diffusers.__version__}")
except ImportError:
    print("   ❌ Diffusers 未安裝")

print("\n" + "=" * 70)
print("診斷完成！")
print("=" * 70)
