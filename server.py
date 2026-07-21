#!/usr/bin/env python3
"""
YOLO26 后端服务 — 直接调用 ultralytics Python API
支持：模型下载、目标检测、姿态估计、分割、训练、导出
启动：python server.py --port 8050
"""

import base64, io, json, logging, os, sys, time, threading, uuid, re, traceback
from datetime import datetime, timezone
from pathlib import Path
from flask import Flask, jsonify, request, send_file
from flask_cors import CORS

app = Flask(__name__)
CORS(app)

# ── 全局模型缓存 ──
MODEL_CACHE = {}
TRAIN_JOBS = {}
job_lock = threading.Lock()

MODEL_VARIANTS = {
    "yolo26n": "YOLO26 Nano (2.4M params, 40.9 mAP)",
    "yolo26s": "YOLO26 Small (9.5M, 48.6 mAP)",
    "yolo26m": "YOLO26 Medium (20.4M, 53.1 mAP)",
    "yolo26l": "YOLO26 Large (24.8M, 55.0 mAP)",
    "yolo26x": "YOLO26 X-Large (55.7M, 57.5 mAP)",
}
TASK_VARIANTS = {
    "detect": "yolo26n.pt yolo26s.pt yolo26m.pt yolo26l.pt yolo26x.pt",
    "segment": "yolo26n-seg.pt yolo26s-seg.pt yolo26m-seg.pt yolo26l-seg.pt yolo26x-seg.pt",
    "pose": "yolo26n-pose.pt yolo26s-pose.pt yolo26m-pose.pt yolo26l-pose.pt yolo26x-pose.pt",
    "classify": "yolo26n-cls.pt yolo26s-cls.pt yolo26m-cls.pt yolo26l-cls.pt yolo26x-cls.pt",
    "obb": "yolo26n-obb.pt yolo26s-obb.pt yolo26m-obb.pt yolo26l-obb.pt yolo26x-obb.pt",
}

def now_iso():
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")

# ── 模型加载 ────────────────────────────────────────
def resolve_model_path(model_name: str) -> str:
    """在 weights/ 和当前目录中查找模型文件"""
    if not model_name or not model_name.strip() or not model_name.endswith('.pt'):
        return "yolo26n.pt"  # 回退到默认模型
    # 如果已包含路径前缀，直接用
    if "/" in model_name or "\\" in model_name:
        return model_name
    # 优先查找 weights/ 目录
    weights_path = Path("weights") / model_name
    if weights_path.exists():
        return str(weights_path)
    # 回退到当前目录
    if Path(model_name).exists():
        return model_name
    # 都不存在，让 YOLO 自动下载到 weights/
    return str(weights_path)

def get_model(model_name: str, task: str = None):
    """加载 YOLO 模型（带缓存）"""
    from ultralytics import YOLO
    actual_path = resolve_model_path(model_name)
    cache_key = f"{model_name}_{task or 'auto'}"
    if cache_key not in MODEL_CACHE:
        try:
            MODEL_CACHE[cache_key] = YOLO(actual_path, task=task, verbose=False)
            print(f"[模型] 已加载: {model_name} → {actual_path} (task={task or 'auto'})")
        except Exception as e:
            raise RuntimeError(f"模型加载失败 {model_name}: {e}")
    return MODEL_CACHE[cache_key]

# ── 辅助 ────────────────────────────────────────────
def pil_to_b64(pil_img, fmt="JPEG"):
    """PIL Image → base64"""
    buf = io.BytesIO()
    pil_img.save(buf, format=fmt)
    return base64.b64encode(buf.getvalue()).decode()

def results_to_json(results, model_key="", task="detect"):
    """将 YOLO Results 转为 JSON"""
    out = {"detections": [], "task": task, "model": model_key, "speed_ms": None}

    if results and len(results) > 0:
        r = results[0]
        # 速度
        if r.speed:
            speeds = r.speed
            out["speed_ms"] = {
                "preprocess": round(speeds.get("preprocess", 0), 2),
                "inference": round(speeds.get("inference", 0), 2),
                "postprocess": round(speeds.get("postprocess", 0), 2),
            }

        # 分类任务：使用 probs
        if task == "classify" and r.probs is not None:
            probs = r.probs
            top5_idx = probs.top5 if hasattr(probs, 'top5') else []
            top5_conf = probs.top5conf if hasattr(probs, 'top5conf') else []
            for i in range(len(top5_idx)):
                cls_id = int(top5_idx[i])
                conf = float(top5_conf[i]) if i < len(top5_conf) else 0
                name = r.names.get(cls_id, f"class_{cls_id}") if r.names else f"cls_{cls_id}"
                out["detections"].append({
                    "class_id": cls_id, "class_name": str(name),
                    "confidence": round(conf, 4),
                    "bbox": [0, 0, 0, 0],
                })

        # 检测/分割/姿态/OBB：使用 boxes
        elif r.boxes is not None:
            boxes = r.boxes
            for i in range(len(boxes)):
                cls_id = int(boxes.cls[i]) if boxes.cls is not None else 0
                conf = float(boxes.conf[i]) if boxes.conf is not None else 0
                xyxy = boxes.xyxy[i].tolist() if boxes.xyxy is not None else [0,0,0,0]
                name = r.names.get(cls_id, f"class_{cls_id}") if r.names else f"cls_{cls_id}"
                out["detections"].append({
                    "class_id": cls_id, "class_name": str(name),
                    "confidence": round(conf, 4),
                    "bbox": [round(x, 1) for x in xyxy],
                })

        # 关键点 (pose)
        if r.keypoints is not None:
            kpts = r.keypoints
            if kpts.xy is not None and len(kpts.xy) > 0:
                for i in range(min(len(kpts.xy), len(out["detections"]))):
                    out["detections"][i]["keypoints"] = kpts.xy[i].tolist() if hasattr(kpts.xy[i], 'tolist') else kpts.xy[i]
                    if kpts.conf is not None and i < len(kpts.conf):
                        out["detections"][i]["kp_conf"] = kpts.conf[i].tolist() if hasattr(kpts.conf[i], 'tolist') else kpts.conf[i]

        # 掩码 (segment)
        if r.masks is not None:
            masks = r.masks
            if masks.xy is not None and len(masks.xy) > 0:
                for i in range(min(len(masks.xy), len(out["detections"]))):
                    poly = masks.xy[i].tolist() if hasattr(masks.xy[i], 'tolist') else masks.xy[i]
                    out["detections"][i]["segmentation_polygon"] = [[round(p, 1) for p in pt] for pt in poly]

        # 标注图像
        try:
            annotated = r.plot(conf=True, labels=True, boxes=True)
            out["annotated_b64"] = pil_to_b64(annotated)
        except Exception:
            pass

    return out

# ══════════════════════════════════════════════════════
#  API 路由
# ══════════════════════════════════════════════════════

@app.route("/api/health")
def health():
    return jsonify({"status": "ok", "timestamp": now_iso(), "cached_models": list(MODEL_CACHE.keys())})

# ── 模型下载 / 列表 ─────────────────────────────────
@app.route("/api/models/available")
def available_models():
    """列出所有 YOLO26 变体及下载状态"""
    models = []
    for task, variants_str in TASK_VARIANTS.items():
        for v in variants_str.split():
            # Check weights/ first, then current dir
            path = Path("weights") / v
            if not path.exists():
                path = Path(v)
            exists = path.exists()
            size_mb = round(path.stat().st_size / 1e6, 2) if exists else None
            models.append({"name": v, "task": task, "downloaded": exists, "size_mb": size_mb, "path": str(path)})
    return jsonify(models)

@app.route("/api/models/download", methods=["POST"])
def download_model():
    """下载模型权重（YOLO 自动从 GitHub Releases 下载到 weights/）"""
    data = request.get_json() or {}
    model_name = data.get("model", "yolo26n.pt")
    task = data.get("task")
    try:
        actual_path = resolve_model_path(model_name)
        get_model(model_name, task=task)
        path = Path(actual_path)
        size_mb = round(path.stat().st_size / 1e6, 2) if path.exists() else None
        return jsonify({"success": True, "model": model_name, "size_mb": size_mb, "path": str(path), "message": f"模型 {model_name} 已就绪"})
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route("/api/models/cached")
def cached_models():
    return jsonify(list(MODEL_CACHE.keys()))

# ── Apple 液态玻璃标注 ──────────────────────────
def draw_tech_boxes(pil_img, result):
    """Apple 液态玻璃风格：细线圆角框、标签嵌在框内、渐变层次感"""
    from PIL import Image, ImageDraw, ImageFont

    img = pil_img.copy().convert("RGBA")
    w, h = img.size

    APPLE = [
        (0, 122, 255), (52, 199, 89), (255, 149, 0), (255, 59, 48),
        (175, 82, 222), (90, 200, 250), (255, 204, 0), (255, 45, 85),
    ]

    try:
        font = ImageFont.truetype("/System/Library/Fonts/Helvetica.ttc", 14)
    except Exception:
        try:
            font = ImageFont.truetype("/System/Library/Fonts/Supplemental/Arial.ttf", 14)
        except Exception:
            font = ImageFont.load_default()

    boxes = result.boxes
    if boxes is None:
        return img.convert("RGB")

    # 玻璃标签单独图层
    glass = Image.new("RGBA", img.size, (0, 0, 0, 0))
    gd = ImageDraw.Draw(glass)
    draw = ImageDraw.Draw(img)

    for i in range(len(boxes)):
        box = boxes.xyxy[i].cpu().numpy()
        cls_id = int(boxes.cls[i]) if boxes.cls is not None else 0
        conf = float(boxes.conf[i]) if boxes.conf is not None else 0
        name = (result.names or {}).get(cls_id, f"cls_{cls_id}")
        color = APPLE[i % len(APPLE)]
        x1, y1, x2, y2 = [int(v) for v in box]
        bw, bh = x2 - x1, y2 - y1
        r = 10  # 框圆角

        # ── 框体：外发光 + 细线 ──
        # 发光层
        draw.rounded_rectangle([x1-1, y1-1, x2+1, y2+1], radius=r,
                               outline=color + (35,), width=4)
        # 主框线
        draw.rounded_rectangle([x1, y1, x2, y2], radius=r,
                               outline=color, width=2)
        # 微弱填充
        draw.rounded_rectangle([x1+3, y1+3, x2-3, y2-3], radius=max(0, r-3),
                               fill=color + (6,))

        # ── 液态玻璃标签（嵌在框内左上角）──
        label = f"{name}  {conf:.0%}"
        tb = draw.textbbox((0, 0), label, font=font)
        tw, th = tb[2] - tb[0], tb[3] - tb[1]
        px, py = 10, 5
        lx, ly = x1 + 4, y1 + 4
        lw_label, lh_label = tw + px * 2, th + py * 2

        if bw > lw_label + 16 and bh > lh_label + 16:
            # 底层暗色
            gd.rounded_rectangle(
                [lx, ly, lx + lw_label, ly + lh_label], radius=6,
                fill=(10, 12, 18, 200))
            # 中间层（稍亮，模拟玻璃折射）
            gd.rounded_rectangle(
                [lx + 1, ly + 1, lx + lw_label - 1, ly + lh_label - 1], radius=5,
                fill=(30, 32, 40, 160))
            # 顶部高光线
            gd.rectangle(
                [lx + 4, ly + 1, lx + lw_label - 4, ly + 2],
                fill=(255, 255, 255, 18))
            # 左侧色条
            gd.rectangle([lx + 5, ly + 4, lx + 7, ly + lh_label - 4],
                         fill=color + (230,))
            # 文字
            gd.text((lx + px + 4, ly + py), label, fill=(255, 255, 255, 250), font=font)

    img = Image.alpha_composite(img, glass)
    return img.convert("RGB")


# ── 推理 ────────────────────────────────────────────
@app.route("/api/predict", methods=["POST"])
def predict():
    """图片推理：上传 base64 图片或 URL，返回检测结果"""
    data = request.get_json() or {}
    model_name = data.get("model", "yolo26n.pt")
    task = data.get("task", "detect")
    conf = float(data.get("conf", 0.25))
    iou = float(data.get("iou", 0.7))
    imgsz = int(data.get("imgsz", 640))
    max_det = int(data.get("max_det", 300))
    device = data.get("device", "mps")  # MPS加速(Mac) / CUDA / CPU
    source = data.get("source")  # base64 image 或 URL 或文件路径

    try:
        model = get_model(model_name, task=task if task != "detect" else None)
    except Exception as e:
        return jsonify({"error": str(e)}), 500

    # 如果没传 source，使用默认测试图片
    if not source:
        from ultralytics.utils import ASSETS
        source = str(Path(ASSETS) / "bus.jpg")

    # 处理不同来源
    from PIL import Image, ImageOps
    try:
        if source.startswith("data:image"):
            header, encoded = source.split(",", 1)
            img_bytes = base64.b64decode(encoded)
            img = Image.open(io.BytesIO(img_bytes))
            img = ImageOps.exif_transpose(img)  # 修正 EXIF 旋转
            results = model(img, conf=conf, iou=iou, imgsz=imgsz, max_det=max_det, device=device, verbose=False)
        else:
            results = model(source, conf=conf, iou=iou, imgsz=imgsz, max_det=max_det, device=device, verbose=False)
    except Exception as e:
        return jsonify({"error": f"推理失败: {e}"}), 500

    # 用检测结果生成真实终端风格的输出
    r0 = results[0] if results and len(results) > 0 else None
    sp = r0.speed if r0 and r0.speed else {}
    n_det = len(r0.boxes) if r0 and r0.boxes else 0
    src_name = "webcam_frame" if (source and source.startswith("data:image")) else (source or "image")
    if len(src_name) > 60: src_name = src_name[:57] + "..."
    img_shape = f"{r0.orig_shape[1]}x{r0.orig_shape[0]}" if r0 and r0.orig_shape else "?"
    labels = {}
    if r0 and r0.boxes:
        for i in range(min(n_det, 20)):
            cls_id = int(r0.boxes.cls[i])
            name = r0.names.get(cls_id, f"cls_{cls_id}") if r0.names else f"cls_{cls_id}"
            labels[name] = labels.get(name, 0) + 1
    label_str = ", ".join(f"{v} {k}{'s' if v>1 else ''}" for k, v in labels.items())

    terminal_lines = [
        f"Ultralytics 8.4.102 🚀 Python torch-2.13.0 {device.upper()}",
        f"{model_name} summary: {r0.names.__len__() if r0 and r0.names else '?'} classes, {n_det} objects detected",
        f"",
        f"image 1/1 {src_name}: {img_shape} {label_str}, {sp.get('inference', '?')}ms",
        f"Speed: {sp.get('preprocess', '?')}ms preprocess, {sp.get('inference', '?')}ms inference, {sp.get('postprocess', '?')}ms postprocess",
    ]
    terminal_output = "\n".join(terminal_lines)

    out = results_to_json(results, model_name, task)
    out["terminal"] = terminal_output[-4000:] if terminal_output else ""
    # 在原图上绘制科技风检测框（保持原始宽高比，不变形）
    if results and len(results) > 0:
        try:
            # 重新加载原始图像
            if source.startswith("data:image"):
                header, encoded = source.split(",", 1)
                img_bytes = base64.b64decode(encoded)
                orig_img = ImageOps.exif_transpose(Image.open(io.BytesIO(img_bytes))).convert("RGB")
            else:
                orig_img = ImageOps.exif_transpose(Image.open(source)).convert("RGB")
            annotated = draw_tech_boxes(orig_img, results[0])
            out["annotated_b64"] = pil_to_b64(annotated)
        except Exception as e:
            import traceback
            print(f"[WARN] annotation failed: {e}")
            traceback.print_exc()
    return jsonify(out)

# ── 训练 ────────────────────────────────────────────
@app.route("/api/train", methods=["POST"])
def start_training():
    """启动训练任务"""
    data = request.get_json() or {}
    model_name = data.get("model", "yolo26n.pt")
    dataset = data.get("data", "coco8.yaml")
    epochs = int(data.get("epochs", 100))
    imgsz = int(data.get("imgsz", 640))
    device = data.get("device", "mps")  # MacBook MPS 加速（Intel Mac 自动回退 CPU）
    task = data.get("task", "detect")
    batch = int(data.get("batch", 16))
    name = data.get("name", f"train_{int(time.time())}")

    job_id = str(uuid.uuid4())[:8]
    with job_lock:
        TRAIN_JOBS[job_id] = {
            "id": job_id, "model": model_name, "status": "starting",
            "epoch": 0, "epochs": epochs, "progress": 0.0,
            "metrics": {}, "log": [], "started_at": now_iso(), "error": None,
        }

    def train_thread():
        from ultralytics import YOLO
        try:
            with job_lock:
                TRAIN_JOBS[job_id]["status"] = "running"
            model = YOLO(model_name, verbose=False)
            # 训练（verbose=True 让日志输出到控制台，我们捕获难，用文件监控）
            model.train(
                data=dataset, epochs=epochs, imgsz=imgsz, device=device,
                task=task, batch=batch, name=name, exist_ok=True,
                verbose=False,
            )
            with job_lock:
                TRAIN_JOBS[job_id]["status"] = "completed"
                TRAIN_JOBS[job_id]["progress"] = 100.0
                TRAIN_JOBS[job_id]["epoch"] = epochs
        except Exception as e:
            with job_lock:
                TRAIN_JOBS[job_id]["status"] = "failed"
                TRAIN_JOBS[job_id]["error"] = str(e)
            traceback.print_exc()

    threading.Thread(target=train_thread, daemon=True).start()
    return jsonify({"job_id": job_id, "status": "starting"}), 201

@app.route("/api/train/status/<job_id>")
def train_status(job_id):
    with job_lock:
        job = TRAIN_JOBS.get(job_id)
    if not job:
        return jsonify({"error": "任务未找到"}), 404
    return jsonify(job)

@app.route("/api/train/jobs")
def train_jobs():
    with job_lock:
        return jsonify(list(TRAIN_JOBS.values()))

# ── 导出 ────────────────────────────────────────────
@app.route("/api/export", methods=["POST"])
def export_model():
    data = request.get_json() or {}
    model_name = data.get("model", "yolo26n.pt")
    fmt = data.get("format", "onnx")
    try:
        model = get_model(model_name)
        export_path = model.export(format=fmt, verbose=False)
        return jsonify({"success": True, "format": fmt, "output": str(export_path)})
    except Exception as e:
        return jsonify({"error": str(e)}), 500

# ── CLI 命令执行 ──────────────────────────────────────
@app.route("/api/cli", methods=["POST"])
def run_cli():
    """在终端执行 yolo 命令并返回输出"""
    data = request.get_json() or {}
    command = data.get("command", "").strip()
    if not command:
        return jsonify({"error": "需要 command 参数"}), 400

    # 安全限制：只允许 yolo 相关命令
    if not command.startswith(("predict","train","val","export","track","detect","segment","pose","classify","obb","mode=")):
        return jsonify({"error": "请使用 yolo 子命令: predict, train, val, export, track"}), 400

    import subprocess
    try:
        result = subprocess.run(
            ["yolo"] + command.split(),
            capture_output=True, text=True, timeout=120, cwd=str(Path.cwd())
        )
        output = (result.stdout + result.stderr)[-5000:]
        return jsonify({"output": output, "success": result.returncode == 0, "model": "yolo26n.pt"})
    except subprocess.TimeoutExpired:
        return jsonify({"error": "命令超时（120秒）"}), 408
    except Exception as e:
        return jsonify({"error": str(e)}), 500

# ── 文件上传 ─────────────────────────────────────────
UPLOAD_DIR = Path("uploads")
UPLOAD_DIR.mkdir(exist_ok=True)

@app.route("/api/upload", methods=["POST"])
def upload_file():
    """上传图片文件"""
    if "file" not in request.files:
        return jsonify({"error": "缺少 file"}), 400
    f = request.files["file"]
    fname = f"{uuid.uuid4().hex}_{f.filename}"
    fpath = UPLOAD_DIR / fname
    f.save(str(fpath))
    return jsonify({"filename": fname, "path": str(fpath.resolve()), "size": fpath.stat().st_size})

# ── 检测历史存储 ──────────────────────────────────────
HISTORY_DIR = Path("history")
HISTORY_DIR.mkdir(exist_ok=True)

@app.route("/api/history/save", methods=["POST"])
def save_history():
    """保存检测结果到磁盘（图片 + JSON）"""
    data = request.get_json() or {}
    b64img = data.get("image", "")
    meta = data.get("meta", {})
    if not b64img:
        return jsonify({"error": "需要 image 字段"}), 400

    entry_id = str(int(time.time() * 1000))
    img_path = HISTORY_DIR / f"{entry_id}.jpg"
    meta_path = HISTORY_DIR / f"{entry_id}.json"

    try:
        # 保存标注图
        if b64img.startswith("data:image"):
            header, encoded = b64img.split(",", 1)
            with open(img_path, "wb") as f:
                f.write(base64.b64decode(encoded))
        # 保存元数据
        with open(meta_path, "w") as f:
            json.dump(meta, f, ensure_ascii=False)
        return jsonify({"id": entry_id, "success": True})
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route("/api/history/list", methods=["GET"])
def list_history():
    """列出所有历史检测记录"""
    entries = []
    for meta_file in sorted(HISTORY_DIR.glob("*.json"), reverse=True):
        try:
            with open(meta_file) as f:
                meta = json.load(f)
            entry_id = meta_file.stem
            img_file = HISTORY_DIR / f"{entry_id}.jpg"
            meta["id"] = entry_id
            meta["has_image"] = img_file.exists()
            meta["image_url"] = f"/api/history/image/{entry_id}"
            entries.append(meta)
        except Exception:
            pass
    return jsonify(entries[:100])  # 最多100条

@app.route("/api/history/image/<entry_id>", methods=["GET"])
def get_history_image(entry_id):
    """获取历史检测图片"""
    img_path = HISTORY_DIR / f"{entry_id}.jpg"
    if img_path.exists():
        from flask import send_file
        return send_file(str(img_path), mimetype="image/jpeg")
    return jsonify({"error": "图片未找到"}), 404

@app.route("/api/history/delete/<entry_id>", methods=["DELETE"])
def delete_history(entry_id):
    """删除历史记录"""
    deleted = 0
    for ext in [".jpg", ".json"]:
        p = HISTORY_DIR / f"{entry_id}{ext}"
        if p.exists():
            p.unlink()
            deleted += 1
    return jsonify({"deleted": deleted > 0})

# ── 硬件检测 ──────────────────────────────────────────
@app.route("/api/system/info")
def system_info():
    """检测硬件设备，返回推荐的推理设备"""
    import torch, platform
    info = {
        "platform": platform.system(),
        "machine": platform.machine(),
        "python": platform.python_version(),
        "pytorch": torch.__version__,
    }
    # 检测最佳设备
    if torch.cuda.is_available():
        info["device"] = "cuda"
        info["device_name"] = torch.cuda.get_device_name(0)
        info["gpu_count"] = torch.cuda.device_count()
    elif torch.backends.mps.is_available():
        info["device"] = "mps"
        info["device_name"] = "Apple MPS (Metal Performance Shaders)"
    else:
        info["device"] = "cpu"
        info["device_name"] = platform.processor() or "CPU"
    return jsonify(info)

# ── 模型信息 ─────────────────────────────────────────
@app.route("/api/models/info")
def model_info():
    """返回所有 YOLO26 变体信息"""
    return jsonify({
        "variants": {
            "detect": [
                {"name": "yolo26n.pt", "size": "nano", "params_m": 2.4, "map": 40.9, "speed_ms": 1.7, "flops_b": 5.4},
                {"name": "yolo26s.pt", "size": "small", "params_m": 9.5, "map": 48.6, "speed_ms": 2.5, "flops_b": 20.7},
                {"name": "yolo26m.pt", "size": "medium", "params_m": 20.4, "map": 53.1, "speed_ms": 4.7, "flops_b": 68.2},
                {"name": "yolo26l.pt", "size": "large", "params_m": 24.8, "map": 55.0, "speed_ms": 6.2, "flops_b": 86.4},
                {"name": "yolo26x.pt", "size": "xlarge", "params_m": 55.7, "map": 57.5, "speed_ms": 11.8, "flops_b": 193.9},
            ],
            "segment": [
                {"name": "yolo26n-seg.pt", "size": "nano", "params_m": 2.7, "box_map": 39.6, "mask_map": 33.9, "speed_ms": 2.1},
                {"name": "yolo26s-seg.pt", "size": "small", "params_m": 10.4, "box_map": 47.3, "mask_map": 40.0, "speed_ms": 3.3},
                {"name": "yolo26m-seg.pt", "size": "medium", "params_m": 23.6, "box_map": 52.5, "mask_map": 44.1, "speed_ms": 6.7},
                {"name": "yolo26l-seg.pt", "size": "large", "params_m": 28.0, "box_map": 54.4, "mask_map": 45.5, "speed_ms": 8.0},
                {"name": "yolo26x-seg.pt", "size": "xlarge", "params_m": 62.8, "box_map": 56.5, "mask_map": 47.0, "speed_ms": 16.4},
            ],
            "pose": [
                {"name": "yolo26n-pose.pt", "size": "nano", "params_m": 2.9, "map": 57.2, "speed_ms": 1.8},
                {"name": "yolo26s-pose.pt", "size": "small", "params_m": 10.4, "map": 63.0, "speed_ms": 2.7},
                {"name": "yolo26m-pose.pt", "size": "medium", "params_m": 21.5, "map": 68.8, "speed_ms": 5.0},
                {"name": "yolo26l-pose.pt", "size": "large", "params_m": 25.9, "map": 70.4, "speed_ms": 6.5},
                {"name": "yolo26x-pose.pt", "size": "xlarge", "params_m": 57.6, "map": 71.6, "speed_ms": 12.2},
            ],
            "classify": [
                {"name": "yolo26n-cls.pt", "size": "nano", "params_m": 2.8, "top1": 71.4, "top5": 90.1, "speed_ms": 1.1},
                {"name": "yolo26s-cls.pt", "size": "small", "params_m": 6.7, "top1": 76.0, "top5": 92.9, "speed_ms": 1.3},
                {"name": "yolo26m-cls.pt", "size": "medium", "params_m": 11.6, "top1": 78.1, "top5": 94.2, "speed_ms": 2.0},
                {"name": "yolo26l-cls.pt", "size": "large", "params_m": 14.1, "top1": 79.0, "top5": 94.6, "speed_ms": 2.8},
                {"name": "yolo26x-cls.pt", "size": "xlarge", "params_m": 29.6, "top1": 79.9, "top5": 95.0, "speed_ms": 3.8},
            ],
            "obb": [
                {"name": "yolo26n-obb.pt", "size": "nano", "params_m": 2.5, "map": 52.4, "speed_ms": 2.8},
                {"name": "yolo26s-obb.pt", "size": "small", "params_m": 9.8, "map": 54.8, "speed_ms": 4.9},
                {"name": "yolo26m-obb.pt", "size": "medium", "params_m": 21.2, "map": 55.3, "speed_ms": 10.2},
                {"name": "yolo26l-obb.pt", "size": "large", "params_m": 25.6, "map": 56.2, "speed_ms": 13.0},
                {"name": "yolo26x-obb.pt", "size": "xlarge", "params_m": 57.6, "map": 56.7, "speed_ms": 30.5},
            ],
        },
        "tasks": ["detect", "segment", "pose", "classify", "obb"],
        "export_formats": ["onnx","tensorrt","coreml","tflite","openvino","torchscript","tfjs","paddle","ncnn","mnn","rknn","qnn","hailo","axelera","ambarella","executorch","imx","saved_model","pb","edgetpu","litert"],
    })

# ══════════════════════════════════════════════════════
# ── 前端页面托管（解决摄像头 file:// 权限问题）────────────
@app.route("/")
def serve_index():
    """提供前端仪表盘页面（走 http:// 协议，摄像头权限正常）"""
    index_path = Path(__file__).parent / "Yolo26.html"
    if index_path.exists():
        from flask import Response
        return Response(index_path.read_text(encoding="utf-8"), mimetype="text/html",
                       headers={"Cache-Control":"no-cache, no-store, must-revalidate",
                               "Pragma":"no-cache","Expires":"0"})
    return "<h1>Yolo26.html 未找到</h1>", 404

@app.route("/<path:filename>")
def serve_static(filename):
    """提供静态文件"""
    file_path = Path(__file__).parent / filename
    if file_path.exists() and file_path.is_file():
        # 根据扩展名设置 MIME
        ext = file_path.suffix.lower()
        mime_map = {".html":"text/html",".js":"application/javascript",".css":"text/css",
                    ".json":"application/json",".png":"image/png",".jpg":"image/jpeg",
                    ".svg":"image/svg+xml",".ico":"image/x-icon"}
        from flask import Response
        return Response(file_path.read_bytes(), mimetype=mime_map.get(ext,"application/octet-stream"))
    return jsonify({"error":"Not found"}), 404

if __name__ == "__main__":
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument("--port", type=int, default=8050)
    p.add_argument("--host", default="0.0.0.0")
    args = p.parse_args()
    print(f"\n{'='*60}")
    print(f"  🚀 YOLO26 纯净态平台已启动")
    print(f"  📍 打开浏览器访问: http://localhost:{args.port}")
    print(f"  📍 API 地址:        http://localhost:{args.port}/api/health")
    print(f"{'='*60}\n")
    app.run(host=args.host, port=args.port, debug=False, threaded=True)
