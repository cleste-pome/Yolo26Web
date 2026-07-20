#!/usr/bin/env bash
# Windows 用户一键启动脚本
# 首次使用: launch.bat --setup
# 正常启动: launch.bat

@echo off
setlocal enabledelayedexpansion
set "SCRIPT_DIR=%~dp0"
cd /d "%SCRIPT_DIR%"
set "VENV_DIR=%SCRIPT_DIR%venv"

if "%~1"=="--setup" (
    echo ========================================
    echo   YOLO26 纯净态系统 - 环境配置
    echo ========================================
    echo.

    if not exist "%VENV_DIR%" (
        echo 创建 Python 虚拟环境...
        python -m venv "%VENV_DIR%"
    ) else (
        echo 虚拟环境已存在
    )

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

if not exist "%VENV_DIR%" (
    echo 虚拟环境未找到，请先运行: launch.bat --setup
    pause
    exit /b 1
)

echo ========================================
echo   YOLO26 纯净态管理系统
echo ========================================
echo.

call "%VENV_DIR%\Scripts\activate.bat"

echo 启动后端服务器 (http://localhost:5000)...
start "YOLO26 Backend" python server.py --port 5000

echo 等待后端就绪...
timeout /t 5 /nobreak >nul

echo.
echo 打开前端仪表盘...
start index.html

echo.
echo ========================================
echo   系统已启动
echo   后端 API: http://localhost:5000
echo   前端页面: 已自动打开浏览器
echo   关闭此窗口或 Ctrl+C 停止服务器
echo ========================================
pause
