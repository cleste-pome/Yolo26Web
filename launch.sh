#!/usr/bin/env bash
# ============================================================
# YOLO26 纯净态管理系统 — 一键启动脚本
# 首次使用: ./launch.sh --setup
# 正常启动: ./launch.sh
# ============================================================
set -e

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
cd "$SCRIPT_DIR"
VENV_DIR="$SCRIPT_DIR/venv"

# ── 首次设置 ──────────────────────────────────
if [ "$1" = "--setup" ]; then
    echo "========================================"
    echo "  🔧 YOLO26 纯净态系统 — 环境配置"
    echo "========================================"
    echo ""

    # 创建虚拟环境
    if [ ! -d "$VENV_DIR" ]; then
        echo "📦 创建 Python 虚拟环境..."
        python3 -m venv "$VENV_DIR"
    else
        echo "✅ 虚拟环境已存在"
    fi

    # 激活并安装
    echo ""
    echo "📥 安装依赖..."
    source "$VENV_DIR/bin/activate"
    pip install --upgrade pip -q
    pip install flask flask-cors ultralytics
    echo ""
    echo "========================================"
    echo "  ✅ 环境配置完成！"
    echo "  运行 ./launch.sh 启动系统"
    echo "========================================"
    exit 0
fi

# ── 检查虚拟环境 ──────────────────────────────
if [ ! -d "$VENV_DIR" ]; then
    echo "❌ 虚拟环境未找到，请先运行: ./launch.sh --setup"
    exit 1
fi

# ── 启动 ──────────────────────────────────────
echo "========================================"
echo "  🚀 YOLO26 纯净态管理系统"
echo "========================================"
echo ""

source "$VENV_DIR/bin/activate"

echo "🔧 启动后端服务器 (http://localhost:8050)..."
python server.py --port 8050 &
SERVER_PID=$!

# 等待服务器就绪
echo "⏳ 等待后端就绪..."
for i in $(seq 1 15); do
    sleep 1
    if curl -s http://localhost:8050/api/health > /dev/null 2>&1; then
        echo "✅ 后端服务器就绪"
        break
    fi
    echo "   等待中... ($i/15)"
done

echo ""
echo "🌐 打开前端仪表盘..."
open http://localhost:8050

echo ""
echo "========================================"
echo "  ✅ 系统已启动"
echo ""
echo "  后端 API : http://localhost:8050"
echo "  前端页面 : 已自动打开浏览器"
echo ""
echo "  按 Ctrl+C 停止服务器"
echo "========================================"

# 等待服务器进程
trap "kill $SERVER_PID 2>/dev/null; echo '👋 服务器已停止'" EXIT
wait $SERVER_PID
