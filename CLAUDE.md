# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## 项目概述

YOLO26 Web — 基于浏览器的 YOLO 视觉 AI 工作站。单文件前端（Yolo26.html）+ Flask 后端（server.py），支持目标检测、实例分割、姿态估计、分类、旋转框检测、目标跟踪六种任务。

## 启动命令

```bash
# 双击启动（macOS）
双击 Yolo26.command

# 终端启动
bash launch.sh                # 完整启动（含环境配置）
python server.py --port 8050  # 仅启动后端
```

## 架构

```
浏览器 (Yolo26.html) ──fetch──> Flask (server.py:8050) ──> ultralytics YOLO ──> weights/*.pt
```

- **前端**：单文件 `Yolo26.html`，无框架依赖。Canvas 渲染检测框，左侧导航多页切换（首页/检测工作室/模型管理/训练/测试/历史）
- **后端**：`server.py`，Flask + flask_cors，ultralytics Python API 直接调用
- **数据**：`weights/` 存 .pt 模型文件（不上传 git），`history/` 存检测历史（图片+JSON），`states.db` / `experiments.db` 存运行时状态

## 关键文件

| 文件 | 作用 |
|------|------|
| `Yolo26.html` | 完整前端，760+ 行，包含所有页面、Canvas 渲染、API 调用 |
| `server.py` | Flask 后端，API 路由、模型管理、标注绘制 |
| `state_manager.py` | 模型版本纯净态管理 |
| `launch.sh` / `launch.bat` / `Yolo26.command` | 启动脚本 |

## API 路由

| 路由 | 用途 |
|------|------|
| `POST /api/predict` | 推理（base64 图/文件路径） |
| `GET /api/models/available` | 模型列表（含 downloaded 状态） |
| `GET /api/models/info` | 分组模型列表（供下载网格） |
| `POST /api/models/download` | 下载模型 |
| `GET /api/system/info` | 设备/内存信息 |
| `GET /api/health` | 健康检查 |

## 标注系统

`draw_tech_boxes()` 在 server.py 中负责后端标注：
- 图片缩放到最大 1920px，坐标同步缩放
- Apple 风格：圆角细线框 + 标签贴在框内顶部
- 字号/线宽按图片尺寸等比缩放（基准 640px）
- 标签自动排版：横排 → 竖排（类别上、概率下）→ 仅类别名
- 前端摄像头：Canvas 实时绘制，读 `stuShowLabels`/`stuShowConf` 复选框

## Python 依赖

Flask、flask-cors、ultralytics、Pillow、torch。首次运行 `bash launch.sh --setup` 自动创建 venv 并安装。

## 注意事项

- 默认识别模型为 Large（yolo26l.pt），非 Nano
- Apple Silicon >8GB 内存时默认推理尺寸 1280
- 模型权重 `weights/*.pt` 在 .gitignore 中排除
- 前端 API 地址硬编码在 `const API='http://localhost:8050'`
- 检测框坐标：后端用 YOLO 返回的 `xyxy`（原图坐标），标注前坐标随图片缩放同步变换
