# 🎯 YOLO Web 项目构建教程

> *手把手教你从零搭建一个 YOLO 视觉 AI 网页平台*

---

## 📑 目录

- [你需要准备什么](#-你需要准备什么)
- [第一步：后端 — 搭建 Flask 服务器](#-第一步后端--搭建-flask-服务器)
- [第二步：后端 — 接入 YOLO 推理](#-第二步后端--接入-yolo-推理)
- [第三步：前端 — 创建页面骨架](#-第三步前端--创建页面骨架)
- [第四步：前端 — Canvas 画布的魔法](#-第四步前端--canvas-画布的魔法)
- [第五步：前后端打通](#-第五步前后端打通)
- [第六步：训练与测试控制台](#-第六步训练与测试控制台)
- [第七步：打磨细节](#-第七步打磨细节)
- [部署与分享](#-部署与分享)
- [常见问题](#-常见问题)

---

## 🛠 你需要准备什么

### 硬件

| 配置 | 最低要求 | 推荐配置 |
|------|---------|---------|
| 内存 | 4GB | 16GB+（Apple Silicon 统一内存） |
| 显卡 | CPU 即可 | MPS/CUDA GPU 加速 |
| 系统 | macOS / Windows / Linux | macOS（前端布局最完整） |

### 软件

```bash
# Python 环境（必须）
python3 --version   # ≥ 3.8
pip3 --version      # Python 包管理器

# 浏览器（任选）
Chrome / Safari / Edge  # 需支持 WebRTC（摄像头功能）
```

### 技术基础

你不需要是前端专家，但建议对以下概念有基本了解：

- **Python 基础**：会写函数、会用 pip 安装包
- **HTML/CSS 基础**：知道标签、选择器是什么
- **JavaScript 基础**：知道 `fetch`、`async/await`、DOM 操作
- **终端基础**：会运行命令、会用 `cd`

> 💡 **完全零基础？** 没关系！现代 AI 编程助手（如 Claude Code）可以帮你写出大部分代码，你只需要理解架构就好。

---

## 🏗 第一步：后端 — 搭建 Flask 服务器

### 1.1 初始化项目目录

```bash
# 创建项目文件夹
mkdir yolo-web-app
cd yolo-web-app

# 创建子目录
mkdir weights         # 模型权重文件
mkdir history         # 检测历史记录
mkdir uploads         # 图片上传临时目录
mkdir datasets        # 数据集

# 创建 Python 虚拟环境
python3 -m venv venv

# 激活虚拟环境
source venv/bin/activate   # macOS/Linux
# venv\Scripts\activate    # Windows

# 安装关键依赖
pip install flask flask-cors ultralytics pillow
```

### 1.2 编写最简单的后端入口

创建一个 `server.py`，这是整个项目的后端核心。它的结构就像餐厅的厨房——前端（服务员）把请求送过来，后端做好菜（推理结果）再送回去。

```python
#!/usr/bin/env python3
"""
YOLO Web 后端服务
功能：接收前端图片 → YOLO 推理 → 返回检测结果 + 标注图
"""

import base64, io, json
from pathlib import Path
from flask import Flask, jsonify, request
from flask_cors import CORS

# 创建 Flask 应用
app = Flask(__name__)
CORS(app)  # 允许前端跨域请求

@app.route("/api/health")
def health():
    """健康检查——前端用这个接口判断后端是否在线"""
    return jsonify({"status": "ok"})

if __name__ == "__main__":
    # 启动服务器，监听所有网络接口（可通过局域网访问）
    app.run(host="0.0.0.0", port=8050)
```

### 1.3 测试服务器

```bash
python server.py
```

打开浏览器访问 `http://localhost:8050/api/health`，你会看到：

```json
{"status": "ok"}
```

> ✅ **第一个里程碑完成了！** 你已经有了一个运行中的后端服务。

---

## 🔌 第二步：后端 — 接入 YOLO 推理

### 2.1 了解 YOLO 的 Python API

Ultralytics 提供了极其简洁的 Python 接口。在 Python 交互环境中试一下：

```python
from ultralytics import YOLO

# 加载模型（会自动下载权重）
model = YOLO("yolo26n.pt")

# 推理一张图片
results = model("bus.jpg")

# 提取检测结果
boxes = results[0].boxes
for i in range(len(boxes)):
    cls_id = int(boxes.cls[i])      # 类别 ID
    conf = float(boxes.conf[i])     # 置信度
    xyxy = boxes.xyxy[i].tolist()   # 框坐标 [x1,y1,x2,y2]
    name = results[0].names[cls_id] # 类别名称（如 "person"）
    
    print(f"{name}: {conf:.0%} at {xyxy}")
```

> 📌 **关键理解**：`results[0].boxes.xyxy` 返回的坐标是相对于**原始图片尺寸**的，不是模型内部处理后的尺寸。

### 2.2 封装推理 API

我们把上面的逻辑封装成一个完整的 `/api/predict` 接口：

```python
# ── 全局模型缓存（避免每次请求都重新加载模型）──
MODEL_CACHE = {}

def get_model(name):
    """加载/缓存 YOLO 模型"""
    if name not in MODEL_CACHE:
        MODEL_CACHE[name] = YOLO(name)
    return MODEL_CACHE[name]


@app.route("/api/predict", methods=["POST"])
def predict():
    """
    图片推理接口
    请求体：{"source": "base64编码的图片", "model": "yolo26n.pt"}
    返回：检测框坐标、类别、置信度、标注图
    """
    data = request.get_json()
    source = data.get("source")   # base64 图片数据
    model_name = data.get("model", "yolo26n.pt")
    
    # 步骤 1：解码 base64 图片 → PIL Image
    header, encoded = source.split(",", 1)
    img_bytes = base64.b64decode(encoded)
    img = Image.open(io.BytesIO(img_bytes))
    
    # 步骤 2：加载模型并推理
    model = get_model(model_name)
    results = model(img)
    r = results[0]
    
    # 步骤 3：提取检测结果
    detections = []
    if r.boxes is not None:
        for i in range(len(r.boxes)):
            cls_id = int(r.boxes.cls[i])
            conf = float(r.boxes.conf[i])
            xyxy = r.boxes.xyxy[i].tolist()
            detections.append({
                "bbox": [round(x, 1) for x in xyxy],
                "class_id": cls_id,
                "class_name": r.names.get(cls_id, f"cls_{cls_id}"),
                "confidence": round(conf, 4),
            })
    
    return jsonify({"detections": detections, "model": model_name})
```

### 2.3 生成标注图（在图上画框）

只返回坐标数据是不够的——用户想看到框画在图上是什么效果：

```python
def draw_boxes(pil_img, result):
    """在 PIL 图片上绘制检测框，返回标注后的图片"""
    from PIL import ImageDraw, ImageFont
    
    img = pil_img.copy().convert("RGB")
    draw = ImageDraw.Draw(img)
    boxes = result.boxes
    if boxes is None:
        return img
    
    # 配色方案（Apple 系统色）
    COLORS = [(0, 122, 255), (52, 199, 89), (255, 149, 0)]
    
    for i in range(len(boxes)):
        box = boxes.xyxy[i].cpu().numpy()
        cls_id = int(boxes.cls[i])
        conf = float(boxes.conf[i])
        name = result.names[cls_id]
        color = COLORS[cls_id % len(COLORS)]
        
        x1, y1, x2, y2 = [int(v) for v in box]
        
        # 画圆角矩形框
        draw.rounded_rectangle([x1, y1, x2, y2], radius=8, 
                               outline=color, width=2)
        
        # 画标签背景 + 文字
        label = f" {name} {conf:.0%} "
        draw.text((x1 + 4, y1 + 4), label, fill="white")
    
    return img
```

### 2.4 标注图转 base64 返回给前端

```python
def pil_to_b64(pil_img):
    """PIL Image → base64 字符串，方便在网页中显示"""
    buf = io.BytesIO()
    pil_img.save(buf, format="JPEG", quality=90)
    return base64.b64encode(buf.getvalue()).decode()

# 在 predict() 函数末尾添加：
# annotated = draw_boxes(img, results[0])
# result["annotated_b64"] = pil_to_b64(annotated)
```

> ✅ **第二个里程碑！** 后端已经可以接收图片、调用 YOLO 推理、返回框坐标和标注图了。

---

## 🎨 第三步：前端 — 创建页面骨架

### 3.1 理解单文件架构

本项目前端采用**单文件架构**（`Yolo26.html`），所有 HTML/CSS/JS 写在一个文件中。为什么？

| 好处 | 坏处 |
|------|------|
| 无需构建工具 | 文件较大（~800 行） |
| 双击即可运行 | 多人协作较困难 |
| 零依赖 | 代码组织较扁平 |

对于学习项目来说，单文件是**最友好的起点**——你不需要理解 Webpack、Vite、npm 等工具链。

### 3.2 基础 HTML 骨架

```html
<!DOCTYPE html>
<html lang="zh-CN">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>YOLO Web 视觉平台</title>
    <style>
        /* 所有样式写在这里 */
    </style>
</head>
<body>
    <!-- 侧边栏导航 -->
    <nav class="sidebar">
        <div class="sidebar-logo">YOLO</div>
        <button class="nav-item" data-page="home">🏠 首页</button>
        <button class="nav-item" data-page="studio">🎯 检测工作室</button>
    </nav>
    
    <!-- 主内容区 -->
    <div class="main">
        <div class="page active" id="page-home"><!-- 首页 --></div>
        <div class="page" id="page-studio"><!-- 检测工作室 --></div>
    </div>
    
    <script>
        /* 所有 JavaScript 写在这里 */
    </script>
</body>
</html>
```

### 3.3 页面切换逻辑

```javascript
// 当前显示的页面
let curPage = "home";

function switchPage(page) {
    // 隐藏所有页面
    document.querySelectorAll(".page").forEach(p => {
        p.classList.remove("active");
    });
    // 显示目标页面
    document.getElementById("page-" + page).classList.add("active");
    curPage = page;
}

// 点击侧边栏切换页面
document.querySelectorAll(".nav-item").forEach(btn => {
    btn.addEventListener("click", function() {
        switchPage(this.dataset.page);
    });
});
```

### 3.4 CSS 布局要点

本项目使用 CSS Grid + Flexbox 实现三栏布局：

```css
/* 检测工作室三栏布局 */
.studio-wrap {
    display: flex;
    height: 100%;
}

.studio-canvas {
    flex: 1;              /* 中间画布占据全部剩余空间 */
    display: flex;
    align-items: center;
    justify-content: center;
}

.studio-sidebar {
    width: 300px;          /* 右侧面板固定宽度 */
    border-left: 1px solid #30363d;
}
```

> 🌟 **关键理解**：`flex: 1` 是响应式布局的核心——它让画布自动填满窗口宽度的变化。

---

## 🖼 第四步：前端 — Canvas 画布的魔法

### 4.1 为什么用 Canvas？

用一个 `<img>` 标签显示图片最简单，但 Canvas 允许你在图片上**动态绘制**框、文字、掩码、骨架。它是「实时检测」功能的基础。

### 4.2 基础 Canvas 操作

```html
<canvas id="myCanvas"></canvas>
```

```javascript
const canvas = document.getElementById("myCanvas");
const ctx = canvas.getContext("2d");

// 设置画布尺寸
canvas.width = 800;
canvas.height = 600;

// 画一个矩形框
ctx.strokeStyle = "#007AFF";    // 蓝色线条
ctx.lineWidth = 2;
ctx.strokeRect(100, 100, 200, 150);

// 写文字
ctx.fillStyle = "white";
ctx.font = "14px sans-serif";
ctx.fillText("person 91%", 104, 114);

// 画图像
const img = new Image();
img.onload = function() {
    ctx.drawImage(img, 0, 0, canvas.width, canvas.height);
};
img.src = "data:image/jpeg;base64,...";
```

### 4.3 保持图片不变形（关键！）

这是初学者最常踩的坑。Canvas 尺寸和图片尺寸比例不一致时，图片会被**拉伸**。

```javascript
function drawImageContain(canvas, img) {
    const ctx = canvas.getContext("2d");
    
    // 设置 canvas 尺寸 = 容器尺寸
    const rect = canvas.parentElement.getBoundingClientRect();
    canvas.width = Math.floor(rect.width);
    canvas.height = Math.floor(rect.height);
    
    // 计算保持比例的缩放尺寸
    const scale = Math.min(
        canvas.width / img.width, 
        canvas.height / img.height
    );
    const drawW = Math.round(img.width * scale);
    const drawH = Math.round(img.height * scale);
    
    // 居中绘制
    const offsetX = Math.round((canvas.width - drawW) / 2);
    const offsetY = Math.round((canvas.height - drawH) / 2);
    
    // 清空画布 → 填充黑色背景 → 绘制图片
    ctx.fillStyle = "#000";
    ctx.fillRect(0, 0, canvas.width, canvas.height);
    ctx.drawImage(img, offsetX, offsetY, drawW, drawH);
}
```

### 4.4 坐标映射：后端框 → 前端画布

后端返回的框坐标是**原图尺寸**，但前端画布可能比原图小（或大）。需要做坐标换算：

```javascript
// 计算缩放比例
const scaleW = drawW / img.naturalWidth;   // = scale
const scaleH = drawH / img.naturalHeight;  // = scale（宽高比一致）

// 换算框坐标
const box = { x1: 222, y1: 405, x2: 345, y2: 862 };  // 来自后端
const bx = offsetX + box.x1 * scaleW;
const by = offsetY + box.y1 * scaleH;
const bw = (box.x2 - box.x1) * scaleW;
const bh = (box.y2 - box.y1) * scaleH;

// 现在 bx, by, bw, bh 就是在画布上的正确位置
ctx.strokeRect(bx, by, bw, bh);
```

### 4.5 图片上传功能

```html
<input type="file" id="imageInput" accept="image/*" hidden>
<button onclick="document.getElementById('imageInput').click()">
    上传图片
</button>
```

```javascript
document.getElementById("imageInput").onchange = function(e) {
    const file = e.target.files[0];
    if (!file) return;
    
    const reader = new FileReader();
    reader.onload = function(ev) {
        const img = new Image();
        img.onload = function() {
            drawImageContain(canvas, img);
            // 保存原始图片数据，用于后续推理
            cachedImage = ev.target.result;
        };
        img.src = ev.target.result;
    };
    reader.readAsDataURL(file);
};
```

---

## 🔗 第五步：前后端打通

### 5.1 前端调用后端 API

```javascript
const API_BASE = "http://localhost:8050";

async function callAPI(method, path, body) {
    const options = {
        method,
        headers: { "Content-Type": "application/json" },
    };
    if (body) options.body = JSON.stringify(body);
    
    const resp = await fetch(API_BASE + path, options);
    return await resp.json();
}

// 示例：调用推理
async function runPrediction() {
    const result = await callAPI("POST", "/api/predict", {
        model: "yolo26n.pt",
        source: cachedImage,   // 之前保存的 base64 图片
        conf: 0.25,
        imgsz: 640,
    });
    
    if (result.error) {
        alert("推理失败: " + result.error);
        return;
    }
    
    // 显示标注图
    if (result.annotated_b64) {
        const img = new Image();
        img.onload = () => drawImageContain(canvas, img);
        img.src = "data:image/jpeg;base64," + result.annotated_b64;
    }
    
    // 更新检测列表
    updateDetectionList(result.detections);
}
```

### 5.2 检测后端是否在线

```javascript
async function checkBackend() {
    try {
        const resp = await fetch(API_BASE + "/api/health", {
            signal: AbortSignal.timeout(5000)  // 5秒超时
        });
        const data = await resp.json();
        backendOnline = data.status === "ok";
    } catch (e) {
        backendOnline = false;
    }
    
    // 更新界面状态
    document.getElementById("status").textContent = 
        backendOnline ? "✅ 在线" : "❌ 离线";
}

// 页面加载时检查，之后每12秒检查一次
checkBackend();
setInterval(checkBackend, 12000);
```

### 5.3 离线遮罩层

如果后端没启动，前端应该给用户清晰的提示：

```html
<div class="overlay" id="overlay">
    <div class="overlay-card">
        <h2>后端未连接</h2>
        <p>请先启动后端服务：</p>
        <code>python server.py --port 8050</code>
        <button onclick="checkBackend()">🔄 重新连接</button>
        <button onclick="closeOverlay()">✕ 关闭</button>
    </div>
</div>
```

---

## ⚡ 第六步：训练与测试控制台

### 6.1 训练参数面板

YOLO 的训练配置有 68+ 个参数。把它们全部可视化到网页上：

```javascript
// 参数面板配置
const TRAIN_PANELS = [
    {
        title: "📐 基础设置",
        fields: [
            { id: "tm", label: "模型权重", type: "select", 
              options: ["yolo26n.pt", "yolo26s.pt", "yolo26m.pt", "yolo26l.pt", "yolo26x.pt"],
              default: "yolo26n.pt" },
            { id: "te", label: "训练轮数", type: "range", 
              min: 1, max: 1000, default: 100 },
            // ... 更多参数
        ]
    },
    // ... 更多面板
];

function renderTrainPanel() {
    let html = "";
    TRAIN_PANELS.forEach(panel => {
        html += `<div class="panel">`;
        html += `<div class="panel-title">${panel.title}</div>`;
        panel.fields.forEach(f => {
            if (f.type === "select") {
                html += `<label>${f.label}</label>`;
                html += `<select id="${f.id}">`;
                f.options.forEach(o => {
                    html += `<option ${o === f.default ? "selected" : ""}>${o}</option>`;
                });
                html += `</select>`;
            }
            // ... range/checkbox/text 等类型
        });
        html += `</div>`;
    });
    document.getElementById("trainConfig").innerHTML = html;
}
```

### 6.2 实时训练日志

启动训练后，后端把 stdout 写入文件，前端轮询读取：

```python
# 后端：训练日志写入文件
def start_training(data):
    import sys, io
    old_stdout = sys.stdout
    tee = io.StringIO()
    sys.stdout = tee
    
    model = YOLO(data["model"])
    model.train(...)
    
    sys.stdout = old_stdout
    # 保存日志
    Path(f"runs/{job_id}.log").write_text(tee.getvalue())
```

```javascript
// 前端：每秒轮询日志
async function pollTrainingLog(jobId) {
    const timer = setInterval(async () => {
        const status = await callAPI("GET", `/api/train/status/${jobId}`);
        if (status.log) {
            // 追加新日志到界面
            document.getElementById("trainLog").textContent += status.log;
        }
        if (status.status === "finished") {
            clearInterval(timer);
            alert("训练完成！");
        }
    }, 1000);
}
```

> 💡 **训练耗时很长**，所以用轮询而不是等待 HTTP 响应返回。

---

## ✨ 第七步：打磨细节

### 7.1 Apple 风格检测框

一个美观的检测框 = 圆角 + 细线 + 半透明标签：

```python
# 后端标注：圆角框
draw.rounded_rectangle([x1, y1, x2, y2], radius=8, outline=color, width=2)

# 毛玻璃标签
draw.rounded_rectangle([lx, ly, lx+lw, ly+lh], radius=4, fill=(28, 28, 30))

# 文字
draw.text((lx+4, ly+3), label, fill=(255, 255, 255))
```

### 7.2 自适应标签

框太窄时，自动切换排版：

```
宽框: person 91%          → 横排
窄框: person              → 竖排
      91%
极窄: 仅显示 person       → 保留类别名
```

```python
# 判断框宽
if box_width >= label_width + 10:
    # 横排
    draw.text(lx, ly, f"{name} {conf:.0%}")
elif box_height >= label_height * 2 + 6:
    # 竖排
    draw.text(lx, ly, name)
    draw.text(lx, ly + height, f"{conf:.0%}")
```

### 7.3 主题切换

```css
/* 使用 CSS 变量实现主题 */
:root, [data-theme="light"] {
    --bg: #ffffff;
    --fg: #1d1d1f;
    --brand: #007AFF;
}

[data-theme="dark"] {
    --bg: #0d1117;
    --fg: #c9d1d9;
    --brand: #58a6ff;
}

[data-theme="warm"] {
    --bg: #faf8f5;
    --fg: #3d352e;
    --brand: #c07c4a;
}
```

```javascript
function toggleTheme() {
    const themes = ["light", "dark", "warm"];
    const current = document.documentElement.dataset.theme || "light";
    const next = themes[(themes.indexOf(current) + 1) % themes.length];
    document.documentElement.dataset.theme = next;
    localStorage.setItem("theme", next);
}
```

### 7.4 分割掩码和姿态骨架

```python
# 分割掩码：半透明多边形
if result.masks is not None:
    polygon = result.masks.xy[i]
    draw.polygon(polygon, fill=color + (25,), outline=color + (60,))

# 姿态骨架：关键点连线
SKELETON = [(0,1), (0,2), (1,3), (2,4), ...]  # COCO 17 关键点骨架
for a, b in SKELETON:
    if pt_a_valid and pt_b_valid:
        draw.line([pt_a, pt_b], fill=color, width=2)
```

### 7.5 模型下载状态

后端检查本地文件是否存在来判定下载状态：

```python
def is_model_downloaded(name):
    """检查模型权重是否已下载"""
    weights_path = Path("weights") / name
    if weights_path.exists():
        return True
    # 检查 ultralytics 缓存
    cache_path = Path.home() / ".cache" / "torch" / "hub" / "ultralytics" / name
    return cache_path.exists()
```

---

## 🚀 部署与分享

### 本地使用

```bash
# 1. 克隆/下载项目
# 2. 启动后端
python server.py --port 8050
# 3. 浏览器访问
open http://localhost:8050
```

### 局域网分享

在同一 WiFi 下，局域网内的设备也可以访问：

```bash
# 启动时默认监听 0.0.0.0（所有网络接口）
python server.py --port 8050

# 在另一台设备上访问
http://你的IP地址:8050
# 例如：http://192.168.1.100:8050
```

### 生产环境部署

对于生产环境，建议使用 Gunicorn 替换 Flask 自带的开发服务器：

```bash
pip install gunicorn
gunicorn -w 4 -b 0.0.0.0:8050 server:app
```

---

## ❓ 常见问题

### Q: 为什么双击 HTML 文件后显示「后端未连接」？

因为 YOLO 推理需要 Python 后端在后台运行。你必须先启动 `server.py`，再打开网页。

**正确做法**：双击 `Yolo26.command`（macOS）或运行 `bash launch.sh`。

### Q: 图片检测后变形了？

**原因**：画布拉伸了图片，没有保持宽高比。

**解决**：参考 4.3 节的 `drawImageContain()` 函数，用 `Math.min(width/imgW, height/imgH)` 计算等比缩放。

### Q: 检测框和图片位置对不上？

**原因**：后端标注图被缩放后，坐标没有同步缩放。

**解决**：标注函数内部先缩图再画框，框坐标乘以缩放比：`x1 * scale`。

### Q: 为什么模型下载后还显示「未下载」？

**原因**：前端检查的是后端 API 返回的 `downloaded` 字段，而后端没正确检测文件是否存在。

**解决**：确保 `/api/models/available` 接口返回的每个模型都带有 `downloaded: true/false`。

### Q: 摄像头检测很卡？

**原因**：每次抓帧都发送到后端推理（耗时 30-200ms），帧率受限。

**优化**：降低推理频率（约 100ms 一次），缩小推理尺寸（640→320），使用 GPU 加速。

### Q: 如何在 Windows 上运行？

确保 Python 安装后，运行：
```bash
launch.bat --setup   # 首次配置
launch.bat           # 启动
```

---

## 📚 推荐学习路径

| 阶段 | 学习内容 | 预计时间 |
|------|---------|---------|
| 1 | 跟着教程搭建基础后端+前端 | 1-2 小时 |
| 2 | 理解 Canvas 绘制和坐标映射 | 2-3 小时 |
| 3 | 添加图片上传和摄像头功能 | 2-3 小时 |
| 4 | 搭建训练控制台参数面板 | 3-4 小时 |
| 5 | 打磨 UI 细节和用户体验 | 4-8 小时 |
| 6 | 添加分割/姿态/追踪等高级功能 | 8-16 小时 |

> 💡 **最重要的原则**：先跑通最简功能，再逐步完善。不要试图一次性写出完美代码。

---

## 🔗 参考资源

- [Ultralytics 官方文档](https://docs.ultralytics.com/)
- [Flask 中文文档](https://dormousehole.readthedocs.io/)
- [MDN Canvas 教程](https://developer.mozilla.org/zh-CN/docs/Web/API/Canvas_API/Tutorial)
- [本项目 GitHub](https://github.com/cleste-pome/Yolo26Web)

---

> *"从论文公式到开源框架，从 GPU 算力到大模型涌现，每一个字节都承载着无数研究者的灵光一现。我们站在巨人的肩膀上，用 AI 驱动的 AI，把曾经遥不可及的梦想，变成了每个人浏览器里的一次点击。"*
