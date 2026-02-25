#!/bin/bash
# 修復 mediapipe 兼容性問題

echo "=========================================="
echo "修復 mediapipe 兼容性問題"
echo "=========================================="
echo ""

if [ -z "$CONDA_DEFAULT_ENV" ]; then
    echo "⚠️ 請先激活 conda 環境："
    echo "   conda activate openarch"
    exit 1
fi

echo "當前環境: $CONDA_DEFAULT_ENV"
echo ""

echo "方案 1: 安裝兼容版本的 mediapipe (0.10.x)..."
echo "這是最推薦的解決方案"
read -p "是否繼續？(y/n): " confirm1

if [ "$confirm1" = "y" ] || [ "$confirm1" = "Y" ]; then
    pip uninstall -y mediapipe
    pip install mediapipe==0.10.9
    echo "✅ 已安裝 mediapipe 0.10.9"
fi

echo ""
echo "方案 2: 更新 controlnet_aux 到最新版本..."
read -p "是否繼續？(y/n): " confirm2

if [ "$confirm2" = "y" ] || [ "$confirm2" = "Y" ]; then
    pip install -U controlnet-aux
    echo "✅ 已更新 controlnet_aux"
fi

echo ""
echo "=========================================="
echo "修復完成！"
echo "=========================================="
echo ""
echo "請重新運行程序測試："
echo "  python sdxl_turbo_app.py"
