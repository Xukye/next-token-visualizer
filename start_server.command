#!/bin/bash

# 获取当前脚本所在目录
DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
cd "$DIR"

echo "========================================="
echo "   Starting Qwen Next-Token Visualizer / 启动 Qwen 下一个 Token 可视化"
echo "========================================="
echo "Loading the model into memory. This may take about 10-30 seconds."
echo "正在加载大模型到内存/显存中，这可能需要 10-30 秒左右..."
echo "The browser will open automatically when the server is ready."
echo "加载完成后会自动打开网页，请稍作等待。"

# 在后台循环检测服务器是否启动，一旦连通立刻打开网页
(
  while ! curl -s http://localhost:5050 > /dev/null; do
      sleep 1
  done
  open http://localhost:5050
) &

# 启动 Flask 服务器
python3 app.py
