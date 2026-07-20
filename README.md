# YOLO26 纯净态管理系统

> **AI 模型状态管理的 Git 式范式** — 让每一个 `best.pt` 都有来处，有归途，有完整的谱系。

## 项目概述

纯净态（Pure State）系统为 YOLO 模型提供完整的**三态管理体系**：

```
纯净态 (Pure)  ──训练──▶  训练态 (Trained)  ──导出──▶  导出态 (Export)
不可变基准                 完整训练谱系                  部署就绪
```

### 核心能力

- 🔒 **SHA256 完整性验证** — 每个模型文件都有内容指纹
- 🌳 **Git 式分支管理** — 从任意状态 fork 并行实验
- 📊 **实验追踪** — 自动记录所有训练参数、指标和日志
- 🔍 **状态对比** — 任意两个模型状态的指标 diff
- 📜 **谱系追溯** — 从部署态一路回溯到纯净态
- 🎨 **可视化仪表盘** — 网页端一站式管理

## 项目结构

```
yolo26-purestate/
├── server.py           # Flask 后端服务器（REST API）
├── state_manager.py    # 纯净态管理器核心模块
├── index.html          # 前端仪表盘（单文件，直接用浏览器打开）
├── docs/
│   └── pure-state.md   # 纯净态概念完整文档
├── requirements.txt    # Python 依赖
└── README.md           # 本文件
```

## 快速开始

### 1. 安装依赖

```bash
cd yolo26-purestate
pip install -r requirements.txt
```

### 2. 启动后端

```bash
python server.py --port 5000
```

启动后自动扫描当前目录的 YOLO 模型文件并注册为纯净态。

### 3. 打开前端

在浏览器中打开 `index.html`，即可看到完整仪表盘。

> 前端会自动连接 `localhost:5000` 后端。如果后端未启动，将使用模拟数据展示所有功能。

## API 端点速查

| 类别 | 端点 | 方法 | 说明 |
|------|------|------|------|
| 系统 | `/api/health` | GET | 健康检查 |
| 系统 | `/api/stats` | GET | 系统统计 |
| 实验 | `/api/experiments` | GET | 实验列表 |
| 实验 | `/api/experiments/<id>` | GET | 实验详情 |
| 实验 | `/api/experiments/<id>` | DELETE | 删除实验 |
| 训练 | `/api/train` | POST | 启动训练 |
| 训练 | `/api/train/status/<id>` | GET | 训练进度 |
| 验证 | `/api/val` | POST | 运行验证 |
| 推理 | `/api/predict` | POST | 运行推理 |
| 导出 | `/api/export` | POST | 导出模型 |
| 模型 | `/api/models` | GET | 可用模型列表 |
| 指标 | `/api/metrics/<id>` | GET | 训练指标数据 |
| 纯净态 | `/api/pure-states` | GET | 纯净态列表 |
| 纯净态 | `/api/pure-states/register` | POST | 注册纯净态 |
| 训练态 | `/api/trained-states` | GET | 训练态列表 |
| 导出态 | `/api/export-states` | GET | 导出态列表 |
| 谱系 | `/api/states/lineage/<id>` | GET | 完整谱系 |
| 谱系 | `/api/states/lineage-tree/<id>` | GET | 谱系树 |
| 对比 | `/api/states/compare` | POST | 状态对比 |
| 验证 | `/api/states/verify` | GET | 完整性验证 |
| 分支 | `/api/branches` | GET/POST | 分支管理 |

## CLI 使用

```bash
# 注册纯净态
python state_manager.py register --name "YOLO26n" --variant yolo26n --file yolo26n.pt

# 查看谱系树
python state_manager.py tree --id 1

# 对比两个状态
python state_manager.py compare --id1 1 --id2 2

# 完整性检查
python state_manager.py verify

# 垃圾回收
python state_manager.py gc
```

## 文档

详细概念文档：[docs/pure-state.md](docs/pure-state.md)

## 许可

MIT
