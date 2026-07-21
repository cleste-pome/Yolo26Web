#!/usr/bin/env bash
# ============================================================
# YOLO26 纯净态管理系统 — 终端一键启动脚本
# 首次使用: bash launch.sh --setup   # 初始化 Python 虚拟环境
# 正常启动: bash launch.sh           # 启动后端 + 打开浏览器
# ============================================================
set -e  # 遇到非零返回值时立即退出，防止静默失败

# ── 获取项目根目录绝对路径并切换 ──
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
cd "$SCRIPT_DIR"
VENV_DIR="$SCRIPT_DIR/venv"  # Python 虚拟环境路径

# ── 首次设置模式 ──────────────────────────────────
# 用法：bash launch.sh --setup
# 功能：创建虚拟环境 + 安装 Flask/CORS/ultralytics 依赖
if [ "$1" = "--setup" ]; then
    echo "========================================"
    echo "  🔧 YOLO26 纯净态系统 — 环境配置"
    echo "========================================"
    echo ""

    # 创建 Python 虚拟环境（如已存在则跳过）
    if [ ! -d "$VENV_DIR" ]; then
        echo "📦 创建 Python 虚拟环境..."
        python3 -m venv "$VENV_DIR"
    else
        echo "✅ 虚拟环境已存在"
    fi

    # 激活虚拟环境并安装项目依赖
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

# ── 启动前检查 ──────────────────────────────────
if [ ! -d "$VENV_DIR" ]; then
    echo "❌ 虚拟环境未找到，请先运行: ./launch.sh --setup"
    exit 1
fi

# ── 启动后端服务 ────────────────────────────────
echo "========================================"
echo "  🚀 YOLO26 纯净态管理系统"
echo "========================================"
echo ""

# 激活虚拟环境中的 Python 解释器
source "$VENV_DIR/bin/activate"

# 在后台启动 Flask 服务器（端口 8050）
echo "🔧 启动后端服务器 (http://localhost:8050)..."
python server.py --port 8050 &
SERVER_PID=$!  # 保存进程号，用于后续清理

# ── 轮询等待后端就绪（最长 15 秒）──
# 通过 /api/health 端点判断服务是否已启动
echo "⏳ 等待后端就绪..."
for i in $(seq 1 15); do
    sleep 1
    if curl -s http://localhost:8050/api/health > /dev/null 2>&1; then
        echo "✅ 后端服务器就绪"
        break
    fi
    echo "   等待中... ($i/15)"
done

# ── 打开浏览器进入前端界面 ──
echo ""
echo "🌐 打开前端仪表盘..."
open http://localhost:8050

# ── 打印运行状态信息 ──
echo ""
echo "========================================"
echo "  ✅ 系统已启动"
echo ""
echo "  后端 API : http://localhost:8050"
echo "  前端页面 : 已自动打开浏览器"
echo ""
echo "  按 Ctrl+C 停止服务器"
echo "========================================"

# ── 注册退出钩子：按 Ctrl+C 或退出终端时自动杀掉后端 ──
trap "kill $SERVER_PID 2>/dev/null; echo '👋 服务器已停止'" EXIT
wait $SERVER_PID
