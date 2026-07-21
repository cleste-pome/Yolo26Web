@echo off
REM ============================================================
REM YOLO26 纯净态管理系统 — Windows 一键启动脚本
REM 首次使用: launch.bat --setup   # 初始化 Python 虚拟环境
REM 正常启动: launch.bat           # 启动后端 + 打开浏览器
REM ============================================================

setlocal enabledelayedexpansion

REM ── 获取脚本所在目录并切换 ──
set "SCRIPT_DIR=%~dp0"
cd /d "%SCRIPT_DIR%"

REM ── Python 虚拟环境路径 ──
set "VENV_DIR=%SCRIPT_DIR%venv"

REM ── 首次设置模式 ──────────────────────────────────
REM 用法：launch.bat --setup
REM 功能：创建虚拟环境 + 安装 Flask/CORS/ultralytics 依赖
if "%~1"=="--setup" (
    echo ========================================
    echo   YOLO26 纯净态系统 - 环境配置
    echo ========================================
    echo.

    REM 检查并创建 Python 虚拟环境
    if not exist "%VENV_DIR%" (
        echo 创建 Python 虚拟环境...
        python -m venv "%VENV_DIR%"
    ) else (
        echo 虚拟环境已存在
    )

    REM 激活虚拟环境并安装 pip 依赖
    echo.
    echo 安装依赖...
    call "%VENV_DIR%\Scripts\activate.bat"
    pip install --upgrade pip -q
    pip install flask flask-cors ultralytics
    echo.
    echo ========================================
    echo   环境配置完成！运行 launch.bat 启动系统
    echo ========================================
    pause
    exit /b 0
)

REM ── 启动前检查：确保已执行过 --setup ──
if not exist "%VENV_DIR%" (
    echo 虚拟环境未找到，请先运行: launch.bat --setup
    pause
    exit /b 1
)

REM ── 启动后端服务 ────────────────────────────────
echo ========================================
echo   YOLO26 纯净态管理系统
echo ========================================
echo.

REM 激活虚拟环境中的 Python
call "%VENV_DIR%\Scripts\activate.bat"

REM 在后台启动 Flask 服务器（端口 8050）
echo 启动后端服务器 (http://localhost:8050)...
start "YOLO26 Backend" python server.py --port 8050

REM 等待服务器启动
echo 等待后端就绪...
timeout /t 5 /nobreak >nul

REM ── 打开默认浏览器进入前端界面 ──
echo.
echo 打开前端仪表盘...
start http://localhost:8050

REM ── 打印运行状态信息 ──
echo.
echo ========================================
echo   系统已启动
echo   后端 API: http://localhost:8050
echo   前端页面: 已自动打开浏览器
echo   关闭此窗口或 Ctrl+C 停止服务器
echo ========================================
pause
