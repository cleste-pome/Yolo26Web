#!/usr/bin/env bash
# ============================================================
# YOLO26 — 双击启动（macOS）
# 首次使用：先运行一次 ./launch.sh --setup
# ============================================================
set -e
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
cd "$SCRIPT_DIR"

echo "========================================"
echo "  🚀 YOLO26 视觉 AI 工作站"
echo "========================================"
echo ""

# 检查虚拟环境
VENV_DIR="$SCRIPT_DIR/venv"
if [ ! -d "$VENV_DIR" ]; then
    echo "📦 首次运行，正在创建虚拟环境..."
    python3 -m venv "$VENV_DIR"
    source "$VENV_DIR/bin/activate"
    pip install --upgrade pip -q
    pip install flask flask-cors ultralytics
    echo "✅ 环境配置完成"
else
    source "$VENV_DIR/bin/activate"
fi

echo "🔧 启动后端服务器..."
python server.py --port 8050 &
SERVER_PID=$!

# 等待服务器就绪
for i in $(seq 1 15); do
    sleep 1
    if curl -s http://localhost:8050/api/health > /dev/null 2>&1; then
        echo "✅ 后端就绪"
        break
    fi
done

echo "🌐 打开浏览器..."
open http://localhost:8050

echo ""
echo "========================================"
echo "  后端运行中：http://localhost:8050"
echo "  关闭此窗口将停止服务器"
echo "========================================"

trap "kill $SERVER_PID 2>/dev/null; echo '👋 已停止'" EXIT
wait $SERVER_PID
