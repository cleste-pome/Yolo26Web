#!/usr/bin/env bash
# ============================================================
# YOLO26 — macOS 双击启动器
# Finder 中双击此文件即可一键启动 YOLO26 视觉 AI 工作站
# 自动完成：环境检查 → 后端启动 → 打开浏览器
# 首次使用会先创建 Python 虚拟环境并安装依赖
# ============================================================
set -e  # 遇到错误立即退出

# 获取脚本所在目录的绝对路径，并切换到此目录
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
cd "$SCRIPT_DIR"

# ── 打印启动 Banner ──
echo "========================================"
echo "  🚀 YOLO26 视觉 AI 工作站"
echo "========================================"
echo ""

# ── 检查/创建 Python 虚拟环境 ──
VENV_DIR="$SCRIPT_DIR/venv"
if [ ! -d "$VENV_DIR" ]; then
    # 首次运行，自动创建虚拟环境并安装依赖
    echo "📦 首次运行，正在创建虚拟环境..."
    python3 -m venv "$VENV_DIR"
    source "$VENV_DIR/bin/activate"
    pip install --upgrade pip -q
    pip install flask flask-cors ultralytics
    echo "✅ 环境配置完成"
else
    # 已有虚拟环境，直接激活
    source "$VENV_DIR/bin/activate"
fi

# ── 启动 Flask 后端服务器（端口 8050）──
echo "🔧 启动后端服务器..."
python server.py --port 8050 &
SERVER_PID=$!  # 记录后端进程号，退出时自动清理

# ── 轮询等待后端就绪（最长 15 秒）──
for i in $(seq 1 15); do
    sleep 1
    if curl -s http://localhost:8050/api/health > /dev/null 2>&1; then
        echo "✅ 后端就绪"
        break
    fi
done

# ── 在默认浏览器中打开前端界面 ──
echo "🌐 打开浏览器..."
open http://localhost:8050

# ── 打印运行状态提示 ──
echo ""
echo "========================================"
echo "  后端运行中：http://localhost:8050"
echo "  关闭此窗口将停止服务器"
echo "========================================"

# ── 注册退出清理钩子：关闭终端时自动杀掉后端进程 ──
trap "kill $SERVER_PID 2>/dev/null; echo '👋 已停止'" EXIT
wait $SERVER_PID
