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
    "n": "Nano", "s": "Small", "m": "Medium", "l": "Large", "x": "X-Large"
}
TASK_SUFFIX = {"segment": "-seg", "pose": "-pose", "classify": "-cls", "obb": "-obb", "track": ""}

def model_path(name):
    p = Path("weights") / name
    return str(p.resolve()) if p.exists() else name

def get_model(name, task=None):
    import ultralytics
    key = f"{name}_{task}" if task else name
    if key in MODEL_CACHE:
        return MODEL_CACHE[key]
    m = ultralytics.YOLO(model_path(name), task=task)
    MODEL_CACHE[key] = m
    return m

# ── 辅助函数 ──────────────────────────────────────

def pil_to_b64(pil_img, fmt="JPEG", quality=85):
    """PIL Image → base64"""
    buf = io.BytesIO()
    pil_img.save(buf, format=fmt, quality=quality)
    return base64.b64encode(buf.getvalue()).decode()

def results_to_json(results, model_key="", task="detect"):
    """将 YOLO Results 转为 JSON"""
    out = {"detections": [], "task": task, "model": model_key, "speed_ms": None}
    if results and len(results) > 0:
        r = results[0]
        sp = r.speed
        out["speed_ms"] = {
            "preprocess": round(sp.get("preprocess", 0), 1) if sp else 0,
            "inference": round(sp.get("inference", 0), 1) if sp else 0,
            "postprocess": round(sp.get("postprocess", 0), 1) if sp else 0,
        }
        names = r.names or {}
        # 检测/分割/姿态/OBB：使用 boxes
        if r.boxes is not None:
            boxes = r.boxes
            for i in range(len(boxes)):
                cls_id = int(boxes.cls[i]) if boxes.cls is not None else 0
                conf = float(boxes.conf[i]) if boxes.conf is not None else 0
                xyxy = boxes.xyxy[i].tolist() if boxes.xyxy is not None else [0,0,0,0]
                d = {
                    "bbox": [round(x, 1) for x in xyxy],
                    "class_id": cls_id,
                    "class_name": names.get(cls_id, f"cls_{cls_id}"),
                    "confidence": round(conf, 4),
                }
                # 分割多边形
                if task == "segment" and r.masks is not None:
                    try:
                        poly = r.masks.xy[i].tolist()
                        d["segmentation_polygon"] = [[round(p, 1) for p in pt] for pt in poly]
                    except:
                        pass
                # 关键点
                if task == "pose" and r.keypoints is not None:
                    try:
                        kpts = r.keypoints
                        d["keypoints"] = kpts.xy[i].tolist() if kpts.xy is not None else []
                    except:
                        pass
                out["detections"].append(d)
        elif r.probs is not None:
            # 分类
            for i, p in enumerate(r.probs.top5):
                out["detections"].append({
                    "class_id": r.probs.top5[i],
                    "class_name": names.get(r.probs.top5[i], f"cls_{r.probs.top5[i]}"),
                    "confidence": round(float(r.probs.top5conf[i]), 4),
                })
    return out

# ── Apple 风格标注 ──────────────────────────────
def draw_tech_boxes(pil_img, result):
    """圆角细线框 + 标签贴在框内顶部。先缩图再标注，坐标同步缩放。"""
    from PIL import Image, ImageDraw, ImageFont

    img = pil_img.copy().convert("RGB")
    ow, oh = img.size

    # 缩放到最大 1920px，计算缩放比
    MAX_SIZE = 1920
    scale = min(1.0, MAX_SIZE / max(ow, oh))
    if scale < 1.0:
        nw, nh = int(ow * scale), int(oh * scale)
        img = img.resize((nw, nh), Image.LANCZOS)
    else:
        nw, nh = ow, oh
        scale = 1.0

    APPLE = [
        (0, 122, 255), (52, 199, 89), (255, 149, 0), (255, 59, 48),
        (175, 82, 222), (90, 200, 250), (255, 204, 0), (255, 45, 85),
    ]

    # 按缩放后的尺寸计算线宽/字号
    s = max(nw, nh) / 640.0
    line_w = max(2, int(3.5 * s))
    font_s = max(14, int(18 * s))
    box_r = max(6, int(10 * s))
    try:
        font = ImageFont.truetype("/System/Library/Fonts/Helvetica.ttc", font_s)
    except Exception:
        try:
            font = ImageFont.truetype("/System/Library/Fonts/Supplemental/Arial.ttf", font_s)
        except Exception:
            font = ImageFont.load_default()

    boxes = result.boxes
    if boxes is None:
        return img

    draw = ImageDraw.Draw(img)

    for i in range(len(boxes)):
        box = boxes.xyxy[i].cpu().numpy()
        cls_id = int(boxes.cls[i]) if boxes.cls is not None else 0
        conf = float(boxes.conf[i]) if boxes.conf is not None else 0
        name = (result.names or {}).get(cls_id, f"cls_{cls_id}")
        color = APPLE[i % len(APPLE)]
        # 坐标同步缩放
        x1, y1 = int(box[0] * scale), int(box[1] * scale)
        x2, y2 = int(box[2] * scale), int(box[3] * scale)
        bw, bh = x2 - x1, y2 - y1

        # 圆角边框
        draw.rounded_rectangle([x1, y1, x2, y2], radius=box_r, outline=color, width=line_w)

        # 标签贴在框内顶部，小框自动缩短文字
        labels = [f" {name} {conf:.0%} ", f" {conf:.0%} ", f" {name} "]
        for label in labels:
            tb = draw.textbbox((0, 0), label, font=font)
            tw, th = tb[2] - tb[0], tb[3] - tb[1]
            pad = max(3, int(6 * s))
            lw_lb, lh_lb = tw + pad * 2, th + pad
            lx, ly = x1 + max(1, int(2 * s)), y1 + max(1, int(2 * s))
            if bw >= lw_lb + 4 and bh >= lh_lb + 4:
                draw.rounded_rectangle([lx, ly, lx + lw_lb, ly + lh_lb],
                                       radius=max(2, int(4 * s)), fill=(28, 28, 30))
                draw.text((lx + pad, ly + max(1, int(2 * s))), label, fill=(255, 255, 255), font=font)
                break

    return img


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
            # 如果是文件名/路径，尝试用 PIL 打开，否则传给 YOLO 自动解析
            import re
            if source and not re.match(r'^https?://', source):
                src_path = Path(source)
                if src_path.exists():
                    img = ImageOps.exif_transpose(Image.open(src_path))
                    results = model(img, conf=conf, iou=iou, imgsz=imgsz, max_det=max_det, device=device, verbose=False)
                else:
                    results = model(source, conf=conf, iou=iou, imgsz=imgsz, max_det=max_det, device=device, verbose=False)
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
    # 在原图上绘制检测框（保持原始宽高比，不变形）
    if results and len(results) > 0:
        try:
            # 优先用 YOLO 自带的 orig_img（BGR numpy array）
            r0 = results[0]
            if hasattr(r0, 'orig_img') and r0.orig_img is not None:
                import numpy as np
                arr = r0.orig_img
                if isinstance(arr, np.ndarray):
                    orig_img = Image.fromarray(arr[..., ::-1])  # BGR→RGB
                else:
                    orig_img = Image.fromarray(np.array(arr))
            elif source.startswith("data:image"):
                header, encoded = source.split(",", 1)
                img_bytes = base64.b64decode(encoded)
                orig_img = ImageOps.exif_transpose(Image.open(io.BytesIO(img_bytes))).convert("RGB")
            else:
                orig_img = ImageOps.exif_transpose(Image.open(source)).convert("RGB")
            annotated = draw_tech_boxes(orig_img, r0)
            out["annotated_b64"] = pil_to_b64(annotated, quality=92)
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
    job_id = uuid.uuid4().hex[:12]

    with job_lock:
        TRAIN_JOBS[job_id] = {"status": "running", "model": data.get("model", "yolo26n.pt")}

    def _train():
        try:
            model = get_model(data["model"])
            model.train(**{k: v for k, v in data.items() if k != "model"})
            with job_lock:
                TRAIN_JOBS[job_id]["status"] = "finished"
        except Exception as e:
            with job_lock:
                TRAIN_JOBS[job_id]["status"] = f"error: {e}"

    threading.Thread(target=_train, daemon=True).start()
    return jsonify({"job_id": job_id})

@app.route("/api/train/status/<job_id>")
def train_status(job_id):
    with job_lock:
        return jsonify(TRAIN_JOBS.get(job_id, {"status": "not found"}))

# ── 系统信息 ─────────────────────────────────────────
@app.route("/api/system/info")
def sys_info():
    import torch
    d = "mps" if torch.backends.mps.is_available() else ("cuda" if torch.cuda.is_available() else "cpu")
    return jsonify({"device": d, "torch_version": torch.__version__})

@app.route("/api/health")
def health():
    return jsonify({"status": "ok", "cached_models": list(MODEL_CACHE.keys()),
                    "timestamp": datetime.now(timezone.utc).isoformat()})

# ── 模型管理 ─────────────────────────────────────────
def _model_downloaded(name):
    """检查模型是否已下载（本地 weights/ 或 ultralytics 缓存）"""
    if (Path("weights") / name).exists():
        return True
    # ultralytics 默认缓存
    import ultralytics
    hub_dir = Path(ultralytics.__file__).parent / "hub" / "checkpoints"
    if (hub_dir / name).exists():
        return True
    # 也检查 ~/.cache
    home_cache = Path.home() / ".cache" / "torch" / "hub" / "ultralytics" / name
    if home_cache.exists():
        return True
    return name in MODEL_CACHE


@app.route("/api/models/available")
def available_models():
    variants = []
    for sz in ["n", "s", "m", "l", "x"]:
        for task, suffix in [("detect", ""), ("segment", "-seg"), ("pose", "-pose"),
                              ("classify", "-cls"), ("obb", "-obb")]:
            name = f"yolo26{sz}{suffix}.pt"
            variants.append({
                "name": name, "size": sz,
                "label": MODEL_VARIANTS[sz] if not suffix else f"{MODEL_VARIANTS[sz]} {task.title()}",
                "task": task,
                "downloaded": _model_downloaded(name)
            })
    return jsonify(variants)


@app.route("/api/models/download", methods=["POST"])
def download_model():
    data = request.get_json() or {}
    name = data.get("model", "yolo26n.pt")
    p = Path("weights") / name
    if p.exists():
        return jsonify({"success": True, "model": name, "status": "cached",
                       "size_mb": round(p.stat().st_size / 1024 / 1024, 1)})
    try:
        model = get_model(name)
        # ultralytics 默认下载到 cache，尝试拷贝到 weights/
        import ultralytics
        cache_dir = Path(ultralytics.__file__).parent / "hub" / "checkpoints"
        cache_path = cache_dir / name
        if cache_path.exists() and not p.exists():
            import shutil
            shutil.copy2(str(cache_path), str(p))
        return jsonify({"success": True, "model": name, "status": "downloaded",
                       "size_mb": round(p.stat().st_size / 1024 / 1024, 1) if p.exists() else 0})
    except Exception as e:
        return jsonify({"success": False, "error": str(e)})

@app.route("/api/models/info")
def models_info():
    """返回分组的模型列表，供前端下载网格使用"""
    variants = {}
    for sz in ["n", "s", "m", "l", "x"]:
        for task, suffix in [("detect", ""), ("segment", "-seg"), ("pose", "-pose"),
                              ("classify", "-cls"), ("obb", "-obb")]:
            name = f"yolo26{sz}{suffix}.pt"
            variants.setdefault(task, []).append({
                "name": name, "size": sz, "task": task,
                "downloaded": _model_downloaded(name)
            })
    return jsonify({"variants": variants})


@app.route("/api/models/cached")
def cached_models():
    return jsonify(list(MODEL_CACHE.keys()))

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
    eid = str(int(time.time() * 1000))
    # 解码并保存图片
    if b64img and b64img.startswith("data:image"):
        header, encoded = b64img.split(",", 1)
        img_path = HISTORY_DIR / f"{eid}.jpg"
        img_path.write_bytes(base64.b64decode(encoded))
    # 保存 JSON 元数据
    json_path = HISTORY_DIR / f"{eid}.json"
    json_path.write_text(json.dumps({**meta, "id": eid}, ensure_ascii=False, indent=2))
    return jsonify({"id": eid, "status": "saved"})

@app.route("/api/history/list")
def list_history():
    items = []
    for jf in sorted(HISTORY_DIR.glob("*.json"), reverse=True):
        try:
            data = json.loads(jf.read_text())
            data["has_image"] = (HISTORY_DIR / f"{data.get('id','')}.jpg").exists()
            items.append(data)
        except:
            pass
    return jsonify(items[:50])

@app.route("/api/history/image/<entry_id>")
def get_history_image(entry_id):
    img_path = HISTORY_DIR / f"{entry_id}.jpg"
    if img_path.exists():
        return send_file(str(img_path), mimetype="image/jpeg")
    return "not found", 404

@app.route("/api/history/delete/<entry_id>", methods=["DELETE"])
def delete_history(entry_id):
    for ext in [".jpg", ".json"]:
        p = HISTORY_DIR / f"{entry_id}{ext}"
        if p.exists():
            p.unlink()
    return jsonify({"status": "deleted"})

@app.route("/<path:filename>")
def serve_static(filename):
    """提供静态文件：图片、CSS 等"""
    from flask import abort
    path = Path(filename)
    if not path.exists():
        abort(404)
    return send_file(str(path.resolve()))

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
    app.run(host=args.host, port=args.port)
