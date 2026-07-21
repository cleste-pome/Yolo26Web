# YOLO26 视觉 AI 前端管理平台 🚀

> **YOLO26 Web — 基于浏览器的 YOLO 视觉 AI 工作站，支持目标检测、实例分割、姿态估计、分类、旋转框检测与目标跟踪。**

---

## 📖 写在前面

2022 年冬天，我开始做本科毕业设计，课题是小样本目标检测。那时候还是 YOLOv5 的天下，我每天窝在实验室里，对着黑漆漆的终端敲 `python train.py`，等上几个小时，看一串数字跳出来告诉我 mAP 提高了 0.3 个百分点。说实话，那时候我心里一直有个念头——为什么这些东西非要用命令行呢？为什么不能像 Photoshop 一样，左边选工具、右边调参数、中间是画布，点一下就能看到检测结果？为什么训练一个模型要手动改 YAML 文件里的参数，而不是在网页上点几个按钮就搞定？

我当时在毕业设计的「未来展望」章节里写过：希望有一天能做一个可视化的平台，把模型的训练、测试、检测都搬到网页上。但那时候我只是个本科生，会写 Python 脚本，会调参，会看论文，但不会写前端。那些想法只能留在 Word 文档的「展望」章节里，像一个遥远的梦。

然后到了 2026 年。Claude Code 来了。坐在电脑前，我打开这个尘封已久的项目文件夹，这一次我不再是一个人——我有了一个几乎无所不能的 AI 搭档。我说「我想要一个检测工作室，左边是工具栏，中间是画布，右边是参数面板」，它就开始写代码；我说「这里面的参数要完全对应 YOLO 的 default.yaml」，它就读了那个文件，把 68 个参数一个一个地搬到了网页上；我说「终端输出要实时显示在页面上」，它就打通了后端和前端。几百轮对话下来，我本科毕业设计里写下的每一个「未来展望」，都变成了可以点击的按钮、可以拖动的滑块、可以实时查看的画面。

这个项目既是我对计算机视觉的热爱，也是一个时代的见证——AI 不是来取代我们的，是来帮我们实现那些曾经「做不到」的事情的。希望你喜欢这个平台，也希望你也有一个被 AI 圆满的梦想。

📂 **GitHub**: [https://github.com/cleste-pome/Yolo26Web](https://github.com/cleste-pome/Yolo26Web)

---

## 📑 目录

- [项目背景](#项目背景)
- [项目架构](#项目架构)
- [快速开始](#快速开始)
- [功能概览](#功能概览)
  - [首页总览](#-首页总览)
  - [检测工作室](#-检测工作室)
  - [模型管理](#-模型管理)
  - [训练控制台](#-训练控制台)
  - [测试控制台](#-测试控制台)
  - [集成生态](#-集成生态)
- [YOLO 任务支持](#-yolo-任务支持)
- [模型体系](#-模型体系)
- [纯净态概念](#-纯净态概念)
- [API 参考](#-api-参考)
- [技术栈](#-技术栈)
- [项目结构](#-项目结构)
- [开发日志](#-开发日志)

---

## 项目背景

本项目始于本科毕业设计——研究 YOLO 系列模型在多种计算机视觉任务上的表现。在完成核心算法研究后，借助 **Claude Code**（Anthropic 的 AI 编程助手）进行了大规模的 Web 前端开发，将以下能力全部集成到了一个统一的 Web 平台中：

| 阶段 | 工具 | 产出 |
|------|------|------|
| 算法研究 | Python + PyTorch | YOLO 模型训练、推理脚本 |
| 后端开发 | Flask + ultralytics | REST API 服务 |
| 前端开发 | HTML/CSS/JS + Claude Code | 完整 Web 平台 |

> 💡 **AI 辅助开发体验**：从零开始，通过与 Claude Code 数百轮交互，逐步搭建出了一个具有 Photoshop 风格面板布局、三主题切换、实时摄像头检测、完整参数面板等功能的专业级 Web 应用。

---

## 项目架构

```
┌─────────────────────────────────────────────────────────────┐
│                    浏览器 (Frontend)                          │
│  ┌─────────┐  ┌──────────┐  ┌──────────┐  ┌──────────────┐ │
│  │ 首页总览  │  │ 检测工作室 │  │ 模型管理  │  │ 训练/测试台   │ │
│  └─────────┘  └──────────┘  └──────────┘  └──────────────┘ │
│                         │ fetch() API                         │
├─────────────────────────┼────────────────────────────────────┤
│              Flask Server (localhost:8050)                     │
│  ┌──────────┐  ┌──────────┐  ┌──────────┐  ┌──────────────┐ │
│  │ /api/     │  │ /api/     │  │ /api/     │  │ /api/         │ │
│  │ predict   │  │ train     │  │ models    │  │ history        │ │
│  └──────────┘  └──────────┘  └──────────┘  └──────────────┘ │
│                         │                                     │
│              ultralytics Python API                            │
│                         │                                     │
│              YOLO26 Models (weights/)                          │
└─────────────────────────────────────────────────────────────┘
```

---

## 快速开始

### 环境要求

| 依赖 | 版本 |
|------|------|
| Python | ≥ 3.8 |
| PyTorch | ≥ 1.8 |
| ultralytics | ≥ 8.4.0 |
| Flask | ≥ 3.0 |
| 浏览器 | Chrome / Safari / Edge (需支持 WebRTC) |

> ⚠️ **平台建议**：本项目在 **MacBook（macOS）** 上开发与测试，Safari / Chrome 下布局、字体、动画均为完整体验。Windows 系统下，不同浏览器的默认字体渲染、滚动条样式、表单控件尺寸存在差异，可能导致部分 UI 元素错位、面板折叠动画卡顿、以及 CSS Grid / Flexbox 表现不一致。建议优先在 Mac 上使用以获得最佳效果。

### 安装 & 启动

**🥇 推荐方式 — 双击启动（macOS）**

Finder 中双击 `Yolo26.command`，自动完成环境配置、启动后端、打开浏览器。

> 首次双击如提示「无法打开」，**右键 → 打开** 即可（macOS Gatekeeper 安全机制）。换一台新电脑 clone 后同样操作，只需一次。

**🥈 终端方式**

```bash
# 1. 克隆项目
git clone https://github.com/cleste-pome/Yolo26Web.git
cd Yolo26Web

# 2. 一键环境配置（仅首次）
bash launch.sh --setup

# 3. 启动（自动打开浏览器）
bash launch.sh
```

> 提示：如果 `./launch.sh` 报权限错误，改用 `bash launch.sh` 即可，无需 chmod。

启动后访问 **http://localhost:8050**，左侧导航切换各功能页面。

**🥉 直接打开了 HTML？**

如果直接双击 `Yolo26.html`，页面会弹出全屏遮罩提示后端未连接——这是正常的，因为后端服务还没启动。按上述方式启动即可，页面会自动连接。

> **Windows 用户**: 运行 `launch.bat --setup` 然后 `launch.bat`

### 模型权重

首次使用需下载模型权重（约 5-150 MB/个）。在「📥 模型管理」页面一键下载，或等待后端自动下载。

```bash
# 也可以手动下载到 weights/ 目录
# 从 https://github.com/ultralytics/assets/releases 下载 .pt 文件
mkdir -p weights
# 将 .pt 文件放入 weights/ 目录
```

---

## 功能概览

### 🏠 首页总览

项目的入口页面，提供快速导航和各功能模块的概览。

- 核心性能指标展示：57.5 mAP / 1.7ms GPU / 80+ 类别 / 20+ 格式
- 六种视觉任务卡片，点击直达检测工作室
- 一键进入核心功能

### 🎯 检测工作室

**Photoshop 风格三栏布局** — 左侧工具面板 + 中央画布 + 右侧参数面板。

| 功能 | 说明 |
|------|------|
| 📹 实时摄像头检测 | 每 100-120ms 抓帧 → 后端 YOLO 推理 → 实时叠加锚框 |
| 🖼️ 图片上传检测 | 拖拽或点击上传 → 自动推理 → 标注图显示 |
| 🔄 多任务切换 | 6 种任务独立模型，切换后自动重新检测 |
| 📋 检测记录 | 自动保存检测结果到磁盘（`history/`），支持回放和删除 |
| 💻 终端输出 | 实时显示 YOLO 推理日志、速度指标 |
| 🎛️ 可折叠参数面板 | 基础参数 + 高级参数 + 显示选项，共 15+ 可调参数 |

**检测参数：**

| 类别 | 参数 |
|------|------|
| 📐 基础 | 模型选择、推理设备、置信度、IoU、推理尺寸、最大检测数 |
| 🔧 高级 | 类别过滤、视频帧间隔、agnostic NMS、TTA、FP16、retina_masks、线宽 |
| 👁️ 显示 | 显示标签、显示置信度、显示锚框、线宽滑块、字号滑块 |

**三种在线主题：** ☀️ 浅色 → 🍂 暖色 → 🌙 深色，一键切换。

### 📊 模型管理

管理 25 个 YOLO26 预训练模型权重。

| 功能 | 说明 |
|------|------|
| 模型对比表 | 检测/分割/姿态三种任务的性能对比（mAP、速度、参数量） |
| 模型下载 | 25 个模型一键下载，自动保存到 `weights/` 目录 |
| 下载状态 | 实时显示每个模型的下载状态（✅ 已下载 / 📥 待下载） |
| 按任务筛选 | 全部/检测/分割/姿态/分类/OBB |

### ⚡ 训练控制台

完整的 YOLO 训练参数面板，**68 个可调参数**，完全对应 `ultralytics/cfg/default.yaml`。

```
┌──────────────────────┐  ┌────────────────────────────┐
│   Python 代码预览     │  │  📐 基础设置 (10 参数)       │
│                      │  │  🎮 训练控制 (18 参数)       │
│  from ultralytics    │  │  🔬 超参数 (7 参数)          │
│  import YOLO         │  │  📏 损失权重 (4 参数)        │
│                      │  │  🎨 数据增强 (15 参数)       │
│  model = YOLO(...)   │  │                            │
│  model.train(...)    │  │  [▶ 启动训练]               │
│                      │  │  ████████░░ 80%            │
└──────────────────────┘  └────────────────────────────┘
```

**5 个可折叠参数面板：**

| 面板 | 参数数 | 包含 |
|------|--------|------|
| 📐 基础设置 | 10 | 模型权重、数据集、轮数、尺寸、批次、设备、项目目录、实验名称、接着训练、覆盖结果 |
| 🎮 训练控制 | 18 | patience、save_period、cache、workers、pretrained、optimizer、seed、AMP、compile、fraction、freeze、multi_scale、cos_lr、close_mosaic、deterministic、single_cls、rect、verbose |
| 🔬 超参数 | 7 | lr0、lrf、momentum、weight_decay、warmup_epochs、warmup_momentum、warmup_bias_lr |
| 📏 损失权重 | 4 | box、cls、dfl、nbs |
| 🎨 数据增强 | 15 | HSV-H/S/V、degrees、translate、scale、shear、fliplr、flipud、mosaic、mixup、cutmix、copy_paste、erasing、BGR |

点击「▶ 启动训练」→ 左侧代码面板显示完整的 Python 调用 + 实时训练进度日志（YOLO 风格输出）。

### 🧪 测试控制台

独立于训练的测试页面，支持三种模式：

| Tab | 功能 | 参数数 |
|-----|------|--------|
| `val()` 验证 | 在数据集上评估模型性能 | 10 |
| `predict()` 推理 | 对图像执行推理 | 12 |
| `export()` 导出 | 导出为 12 种部署格式 | 9 |

每种模式有专属的参数面板和代码预览，点击执行按钮输出结果日志。

### 🔗 集成生态

展示 YOLO 与主流 AI 平台的集成能力：

| 集成 | 说明 |
|------|------|
| 📊 Weights & Biases | 实验追踪、超参优化、模型注册 |
| ☄️ Comet ML | 实验管理、可视化、生产监控 |
| 🤖 Roboflow | 数据标注、预处理、数据集管理 |
| 🔷 Intel OpenVINO | Intel 硬件优化推理 |
| 🔗 ONNX | 跨平台模型部署 |
| 🚀 TensorRT | NVIDIA GPU 加速 |
| 🍎 CoreML | Apple 生态部署 |
| 🌀 MLflow | ML 生命周期管理 |

支持 **21 种模型导出格式**：ONNX、TensorRT、CoreML、TFLite、OpenVINO、TorchScript、TF.js、PaddlePaddle、NCNN、MNN、RKNN、QNN、Hailo 等。

---

## 🎯 YOLO 任务支持

| 任务 | 英文 | 模型后缀 | 数据集 | 指标 |
|------|------|---------|--------|------|
| 🔍 目标检测 | Detection | `yolo26n.pt` ~ `yolo26x.pt` | COCO (80类) | mAP 40.9–57.5 |
| 🎯 实例分割 | Segmentation | `yolo26n-seg.pt` ~ `yolo26x-seg.pt` | COCO-Seg | Box 39.6–56.5 / Mask 33.9–47.0 |
| 🧍 姿态估计 | Pose | `yolo26n-pose.pt` ~ `yolo26x-pose.pt` | COCO-Pose (17关键点) | mAP 57.2–71.6 |
| 🏷️ 图像分类 | Classification | `yolo26n-cls.pt` ~ `yolo26x-cls.pt` | ImageNet (1000类) | Top-1 71.4%–79.9% |
| 📐 旋转框 | OBB | `yolo26n-obb.pt` ~ `yolo26x-obb.pt` | DOTAv1 (15类) | mAP 52.4–56.7 |
| 👣 目标跟踪 | Tracking | 复用检测模型 | — | BoT-SORT / ByteTrack |

---

## 📊 模型体系

```
                    ┌──────────┐
                    │  YOLO26   │
                    └────┬─────┘
           ┌─────────────┼─────────────┐
           │             │             │
      ┌────▼────┐  ┌────▼────┐  ┌────▼────┐
      │ Detect  │  │Segment  │  │  Pose   │  ...
      └────┬────┘  └────┬────┘  └────┬────┘
     ┌─────┼─────┐      │           │
     │     │     │      │           │
    Nano Small Medium Large    X-Large
    (2.4M)(9.5M)(20.4M)(24.8M)(55.7M)
```

**5 种规模 × 5 种任务 + 1 种跟踪 = 25 个预训练模型**，全部存储在 `weights/` 目录。

### 检测模型性能

| 变体 | mAP 50-95 | T4 速度 | CPU 速度 | 参数量 | 计算量 |
|------|-----------|---------|----------|--------|--------|
| **YOLO26n** | 40.9 | 1.7ms | 38.9ms | 2.4M | 5.4B |
| **YOLO26s** | 48.6 | 2.5ms | 87.2ms | 9.5M | 20.7B |
| **YOLO26m** | 53.1 | 4.7ms | 220ms | 20.4M | 68.2B |
| **YOLO26l** | 55.0 | 6.2ms | 286ms | 24.8M | 86.4B |
| **YOLO26x** | 57.5 | 11.8ms | 526ms | 55.7M | 193.9B |

---

## 💎 纯净态概念

本项目实现了 Git 式的模型版本管理——**「纯净态（Pure State）」**。

```
纯净态(Pure) ──训练──▶ 训练态(Trained) ──导出──▶ 导出态(Export)
yolo26n.pt             best.pt                  model.onnx
SHA256: abc123...      SHA256: def456...        SHA256: ghi789...
不可变基准             完整训练谱系              部署就绪
```

| 概念 | 类比 | 说明 |
|------|------|------|
| **纯净态** | `git init` | 预训练权重的原始快照，SHA256 校验 |
| **训练态** | `git commit` | 训练后的权重 + 超参数 + 指标 |
| **导出态** | `git tag` | 导出为 ONNX/TensorRT 等部署格式 |
| **分支** | `git branch` | 从任意状态分叉，并行实验 |

模块 `state_manager.py` 提供完整的 CLI 和 Python API 进行状态管理。

```bash
# 注册纯净态
python state_manager.py register --name "YOLO26n" --variant yolo26n --file weights/yolo26n.pt

# 查看谱系树
python state_manager.py tree --id 1

# 对比两个训练态
python state_manager.py compare --id1 1 --id2 2

# 垃圾回收
python state_manager.py gc
```

---

## 📡 API 参考

| 方法 | 端点 | 说明 |
|------|------|------|
| GET | `/api/health` | 健康检查 |
| POST | `/api/predict` | 图片推理（支持 base64/URL/文件路径） |
| POST | `/api/train` | 启动训练任务 |
| GET | `/api/train/status/<id>` | 训练进度查询 |
| GET | `/api/models/available` | 模型下载状态列表 |
| GET | `/api/models/info` | 模型详细信息 |
| POST | `/api/models/download` | 下载模型权重 |
| POST | `/api/cli` | 执行 yolo CLI 命令 |
| POST | `/api/export` | 导出模型 |
| POST | `/api/history/save` | 保存检测历史 |
| GET | `/api/history/list` | 历史记录列表 |
| GET | `/api/history/image/<id>` | 获取历史图片 |
| DELETE | `/api/history/delete/<id>` | 删除历史记录 |

### 推理请求示例

```bash
curl -X POST http://localhost:8050/api/predict \
  -H "Content-Type: application/json" \
  -d '{"model":"yolo26n.pt","task":"detect","conf":0.3,"source":"image.jpg"}'
```

响应包含：检测框坐标、类别、置信度、推理速度、YOLO 标注图（base64）、终端输出。

---

## 🛠 技术栈

| 层级 | 技术 |
|------|------|
| 前端 | HTML5 + CSS3 + Vanilla JavaScript（无框架） |
| 后端 | Python Flask + flask-cors |
| AI 引擎 | ultralytics (PyTorch) |
| 数据库 | SQLite (state_manager) |
| 存储 | localStorage + 磁盘文件 (history/) |
| 字体 | Inter + Noto Sans SC + JetBrains Mono |
| 部署 | 单文件 HTML + Python 脚本，一键启动 |

---

## 📂 项目结构

```
Yolo26Web/
├── Yolo26.html         # 前端主页面（单文件，完整多页应用）
├── server.py           # Flask 后端服务器
├── state_manager.py    # 纯净态模型版本管理
├── launch.sh           # macOS/Linux 一键启动脚本
├── launch.bat          # Windows 一键启动脚本
├── requirements.txt    # Python 依赖
├── README.md           # 本文件
├── .gitignore
├── experiments.db      # 实验记录数据库（SQLite）
├── states.db           # 纯净态状态数据库（SQLite）
├── datasets/           # 示例数据集（coco8）
├── uploads/            # 用户上传图片的临时目录
├── weights/            # 模型权重目录（.pt 文件不上传 git）
│   └── .gitkeep
├── history/            # 检测历史记录（图片 + JSON）
├── runs/               # 推理输出目录
│   └── detect/predict/
└── yolo26/             # YOLO 核心库（ultralytics）
```

> **注意**：`weights/*.pt` 模型权重文件不会上传到 GitHub（已在 `.gitignore` 中排除），请通过项目内置的「模型管理」页面下载。

---

## 🔧 开发日志

| 时间 | 里程碑 |
|------|--------|
| 2022 年冬 | 🎓 本科毕业设计——YOLO 多任务视觉算法研究。在「未来展望」里写下：希望有一天做一个可视化平台 |
| 2026 年夏 | 💡 重新打开项目。这一次，有了 Claude Code |
| 1 | 🏗️ 搭建 Flask 后端 + YOLO API 封装，跑通第一个实时推理 |
| 2 | 🎨 前端从零开始——首页、侧边栏、检测工作室的三栏布局 |
| 3 | 📹 实时摄像头检测 + 图片上传推理，解决坐标映射、Canvas 渲染 |
| 4 | ⚡ 训练控制台 + 测试控制台，68 个参数逐一从 default.yaml 迁移 |
| 5 | 🔄 多任务切换、检测历史存储、Photoshop 风格折叠面板 |
| 6 | 🎨 三主题切换、文件路径选择器、API 全面测试 |
| 7 | 📖 写 README
| 2026.07.20 | 🚀 GitHub 开源发布——梦想成真 |

---

## 📄 许可证

本项目代码采用 MIT 许可证。

YOLO26 模型权重版权归 [Ultralytics](https://www.ultralytics.com/) 所有，采用 AGPL-3.0 许可证。商业使用需获取 [Ultralytics Enterprise License](https://www.ultralytics.com/license)。

---

> *"四年前写在毕业设计未来展望里的梦想，被 AI 一行一行地写成了现实。"*
