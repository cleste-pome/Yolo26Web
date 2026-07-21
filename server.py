#!/usr/bin/env python3
"""
YOLO26 后端服务 — 基于 Flask 的视觉 AI 推理引擎
=================================================

本文件是 YOLO26 Web 工作站的服务器端，职责包括：
  - 加载并缓存 ultralytics YOLO 模型
  - 提供 RESTful API 进行目标检测、实例分割、姿态估计、分类、旋转框检测、目标跟踪等推理任务
  - 提供模型管理（浏览/下载/缓存）和训练/验证接口
  - 在检测结果图像上绘制 Apple 风格标注框
  - 管理检测历史记录的存储和查询

技术栈：Flask + flask-cors + ultralytics YOLO + Pillow

启动方式：python server.py --port 8050
"""

# ── 标准库与第三方依赖 ──────────────────────────────────
import base64, io, json, logging, os, sys, time, threading, uuid, re, traceback
from datetime import datetime, timezone
from pathlib import Path
from flask import Flask, jsonify, request, send_file
from flask_cors import CORS

# ── Flask 应用初始化 ────────────────────────────────────
app = Flask(__name__)
CORS(app)                          # 允许跨域请求，前端可跨端口访问

# ── 全局状态：模型缓存与训练任务追踪 ────────────────────
MODEL_CACHE = {}                   # 已加载的 YOLO 模型实例缓存 (key: 模型名)
TRAIN_JOBS = {}                    # 训练任务状态记录 (key: job_id)
job_lock = threading.Lock()        # 保护 TRAIN_JOBS 的线程锁

# 模型规格标签映射（用于前端展示）
MODEL_VARIANTS = {
    "n": "Nano", "s": "Small", "m": "Medium", "l": "Large", "x": "X-Large"
}
# 不同任务对应的模型文件名后缀
TASK_SUFFIX = {"segment": "-seg", "pose": "-pose", "classify": "-cls", "obb": "-obb", "track": ""}

def model_path(name):
    """
    解析模型文件路径。

    优先返回本地 weights/ 目录下的权重文件路径；
    若文件不存在则直接返回原始名称，由 ultralytics 自动从 Hub 下载。

    参数:
        name (str): 模型文件名，如 "yolo26n.pt"
    返回:
        str: 模型文件的绝对路径或原始名称
    """
    p = Path("weights") / name
    return str(p.resolve()) if p.exists() else name

def get_model(name, task=None):
    """
    获取（或加载并缓存）YOLO 模型实例。

    如果指定了 task 参数，则以不同键值缓存，避免同一模型文件因 task
    不同而重复加载。首次调用时由 ultralytics.YOLO() 自动下载权重。

    参数:
        name (str): 模型文件名
        task (str, optional): 任务类型（如 "detect"、"segment"），为 None 时自动推断
    返回:
        ultralytics.YOLO: 模型实例
    """
    import ultralytics
    key = f"{name}_{task}" if task else name
    if key in MODEL_CACHE:
        return MODEL_CACHE[key]
    m = ultralytics.YOLO(model_path(name), task=task)
    MODEL_CACHE[key] = m
    return m

# ── 辅助函数 ────────────────────────────────────────────

def pil_to_b64(pil_img, fmt="JPEG", quality=85):
    """
    将 PIL Image 图像编码为 base64 字符串。

    参数:
        pil_img (PIL.Image): 输入图像
        fmt (str): 输出图像格式，默认 "JPEG"
        quality (int): JPEG 压缩质量，默认 85
    返回:
        str: base64 编码的图像数据（不含 data:image 头）
    """
    buf = io.BytesIO()
    pil_img.save(buf, format=fmt, quality=quality)
    return base64.b64encode(buf.getvalue()).decode()

def results_to_json(results, model_key="", task="detect"):
    """
    将 ultralytics 预测结果 (Results 列表) 转为 JSON 字典。

    支持检测、分割、姿态估计、分类、目标追踪等任务的输出解析。
    对于分割任务会提取多边形坐标，姿态任务提取关键点坐标，
    追踪任务提取 track_id。

    参数:
        results (list): ultralytics Results 对象列表
        model_key (str): 使用的模型名称，供前端记录
        task (str): 任务类型标识 ("detect"/"segment"/"pose"/"classify"/"track")
    返回:
        dict: 包含 "detections"、"task"、"model"、"speed_ms" 字段的字典
    """
    out = {"detections": [], "task": task, "model": model_key, "speed_ms": None}
    if results and len(results) > 0:
        r = results[0]                                            # 取第一张图的结果
        sp = r.speed
        # 提取各阶段耗时（毫秒）
        out["speed_ms"] = {
            "preprocess": round(sp.get("preprocess", 0), 1) if sp else 0,
            "inference": round(sp.get("inference", 0), 1) if sp else 0,
            "postprocess": round(sp.get("postprocess", 0), 1) if sp else 0,
        }
        names = r.names or {}
        # 检测/分割/姿态/OBB：使用 boxes 属性
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
                # 追踪 ID（仅追踪任务有此字段）
                if task == "track" and boxes.id is not None:
                    try:
                        d["track_id"] = int(boxes.id[i])
                    except:
                        pass
                # 分割多边形（仅分割任务）
                if task == "segment" and r.masks is not None:
                    try:
                        poly = r.masks.xy[i].tolist()
                        d["segmentation_polygon"] = [[round(p, 1) for p in pt] for pt in poly]
                    except:
                        pass
                # 姿态关键点（仅姿态估计任务）
                if task == "pose" and r.keypoints is not None:
                    try:
                        kpts = r.keypoints
                        d["keypoints"] = kpts.xy[i].tolist() if kpts.xy is not None else []
                    except:
                        pass
                out["detections"].append(d)
        elif r.probs is not None:
            # 分类任务：使用 top5 概率
            for i, p in enumerate(r.probs.top5):
                out["detections"].append({
                    "class_id": r.probs.top5[i],
                    "class_name": names.get(r.probs.top5[i], f"cls_{r.probs.top5[i]}"),
                    "confidence": round(float(r.probs.top5conf[i]), 4),
                })
    return out

# ── Apple 风格标注绘制 ──────────────────────────────────

def draw_tech_boxes(pil_img, result):
    """
    在图像上绘制 Apple 风格的检测标注框。

    标注风格特点：圆角细线框、标签贴在框内左上角、颜色取自 Apple 色调。
    处理步骤：先将图像缩放到最大 1920px，然后同步缩放坐标进行绘制。
    支持检测框、分割多边形（半透明填充）和姿态关键点（骨架连线+圆点）。

    标签文字自适应排版逻辑：
      1. 优先尝试横排（"类别 概率 #ID"）
      2. 横排放不下时改为竖排（类别一行、概率一行）
      3. 空间极窄时仅显示类别名

    参数:
        pil_img (PIL.Image): 原始输入图像（RGB）
        result (ultralytics.Results): 单个图像的预测结果对象
    返回:
        PIL.Image: 绘制了标注框的图像（已缩放到最大 1920px）
    """
    from PIL import Image, ImageDraw, ImageFont

    img = pil_img.copy().convert("RGB")
    ow, oh = img.size

    # 缩放到最大 1920px，计算缩放比，避免前端展示超大图片
    MAX_SIZE = 1920
    scale = min(1.0, MAX_SIZE / max(ow, oh))
    if scale < 1.0:
        nw, nh = int(ow * scale), int(oh * scale)
        img = img.resize((nw, nh), Image.LANCZOS)
    else:
        nw, nh = ow, oh
        scale = 1.0

    # Apple 风格调色板（8 种颜色循环使用）
    APPLE = [
        (0, 122, 255), (52, 199, 89), (255, 149, 0), (255, 59, 48),
        (175, 82, 222), (90, 200, 250), (255, 204, 0), (255, 45, 85),
    ]

    # 按缩放后的图像尺寸等比计算线宽与字号（基准宽度 640px）
    s = max(nw, nh) / 640.0
    line_w = max(2, int(2.3 * s))
    font_s = max(11, int(10 * s))
    box_r = max(6, int(10 * s))                   # 框圆角半径
    # 加载系统字体，兜底使用默认字体
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
        # 坐标同步缩放（从原图坐标映射到缩略图坐标）
        x1, y1 = int(box[0] * scale), int(box[1] * scale)
        x2, y2 = int(box[2] * scale), int(box[3] * scale)
        bw, bh = x2 - x1, y2 - y1

        # 绘制圆角框
        draw.rounded_rectangle([x1, y1, x2, y2], radius=box_r, outline=color, width=line_w)

        # 标签字号：随框大小自适应，窄框用小字号
        box_fs = max(9, min(font_s, int(min(bw, bh) / 7.5)))
        try:
            box_font = ImageFont.truetype("/System/Library/Fonts/Helvetica.ttc", box_fs)
        except Exception:
            box_font = font
        pad = max(2, int(3 * box_fs / 14))
        lx, ly = x1 + max(1, int(2 * s)), y1 + max(1, int(2 * s))

        # 追踪 ID（可选字段，仅在追踪任务中存在）
        tid = int(boxes.id[i]) if boxes.id is not None and i < len(boxes.id) else None
        tid_str = f"#{tid}" if tid else ""
        # 尝试横排标签：" 类别 概率 #ID "
        label_h = f" {name} {conf:.0%} {tid_str}"
        tb_h = draw.textbbox((0, 0), label_h, font=box_font)
        tw_h, th_h = tb_h[2] - tb_h[0], tb_h[3] - tb_h[1]
        lw_h, lh_h = tw_h + pad * 2, th_h + pad * 2

        # 尝试竖排标签（两行：类别 + 概率）
        label_v_name = f" {name}{' '+tid_str if tid else ''} "
        label_v_conf = f" {conf:.0%} "
        tb_n = draw.textbbox((0, 0), label_v_name, font=box_font)
        tb_c = draw.textbbox((0, 0), label_v_conf, font=box_font)
        tw_v = max(tb_n[2]-tb_n[0], tb_c[2]-tb_c[0])
        th_v = (tb_n[3]-tb_n[0]) + (tb_c[3]-tb_c[0]) + pad

        if bw >= lw_h and bh >= lh_h:
            # 空间充足 → 横排标签
            draw.rounded_rectangle([lx, ly, lx + lw_h, ly + lh_h],
                                   radius=max(2, int(3 * box_fs / 14)), fill=(28, 28, 30))
            draw.text((lx + pad, ly + pad), label_h, fill=(255, 255, 255), font=box_font)
        elif bw >= tw_v + pad * 2 and bh >= th_v + pad * 2:
            # 横向空间不足但纵向够 → 竖排标签
            lw_v, lh_v = tw_v + pad * 2, th_v + pad * 2
            draw.rounded_rectangle([lx, ly, lx + lw_v, ly + lh_v],
                                   radius=max(2, int(3 * box_fs / 14)), fill=(28, 28, 30))
            draw.text((lx + pad, ly + pad), label_v_name, fill=(255, 255, 255), font=box_font)
            draw.text((lx + pad, ly + pad + (tb_n[3]-tb_n[0])), label_v_conf,
                      fill=(200, 200, 210), font=box_font)
        else:
            # 空间极窄 → 只显示类别名
            lw_v, lh_v = tw_v + pad * 2, tb_n[3] - tb_n[0] + pad * 2
            if bw >= lw_v and bh >= lh_v:
                draw.rounded_rectangle([lx, ly, lx + lw_v, ly + lh_v],
                                       radius=max(2, int(3 * box_fs / 14)), fill=(28, 28, 30))
                draw.text((lx + pad, ly + pad), label_v_name, fill=(255, 255, 255), font=box_font)

        # ── 分割掩码（半透明多边形叠加） ──
        if hasattr(result, 'masks') and result.masks is not None:
            try:
                polys = result.masks.xy
                if i < len(polys):
                    pts = [(int(p[0] * scale), int(p[1] * scale)) for p in polys[i]]
                    if len(pts) > 2:
                        draw.polygon(pts, fill=color + (25,), outline=color + (60,))
            except Exception:
                pass

        # ── 姿态关键点（骨架连线 + 关键点圆点） ──
        if hasattr(result, 'keypoints') and result.keypoints is not None:
            try:
                kpts = result.keypoints.xy
                if i < len(kpts):
                    pts = [(int(k[0] * scale), int(k[1] * scale)) for k in kpts[i]]
                    # COCO 骨架连接定义（17 个关键点的连接关系）
                    SKEL = [(0,1),(0,2),(1,3),(2,4),(3,5),(4,6),(5,6),(5,7),(6,8),(7,9),(8,10),
                            (5,11),(6,12),(11,12),(11,13),(12,14),(13,15),(14,16)]
                    sk_c = tuple(min(255, c + 60) for c in color)
                    for a, b in SKEL:
                        if a < len(pts) and b < len(pts) and pts[a][0] > 0 and pts[b][0] > 0:
                            draw.line([pts[a], pts[b]], fill=sk_c, width=max(1, line_w // 2))
                    # 关键点以实心圆表示
                    for px, py in pts:
                        if px > 0 and py > 0:
                            r_kp = max(2, line_w)
                            draw.ellipse([px - r_kp, py - r_kp, px + r_kp, py + r_kp], fill=color)
            except Exception:
                pass

    return img


# ════════════════════════════════════════════════════════
# API 路由：推理
# ════════════════════════════════════════════════════════

@app.route("/api/predict", methods=["POST"])
def predict():
    """
    图片推理接口。

    接收 base64 编码的图片、图片 URL 或本地文件路径，执行指定模型的推理，
    返回检测结果 JSON（含检测框坐标、类别、置信度等），
    同时返回终端风格的日志文本和带标注的图片（base64）。

    请求体 JSON 字段:
        - model (str): 模型文件名，默认 "yolo26n.pt"
        - task (str): 任务类型，默认 "detect"
        - conf (float): 置信度阈值，默认 0.25
        - iou (float): NMS IoU 阈值，默认 0.7
        - imgsz (int): 推理图像尺寸，默认 640
        - max_det (int): 最大检测数，默认 300
        - device (str): 计算设备，默认 "mps"（Mac MPS 加速）
        - source (str): 图片数据（base64 格式 data:image/...）或 URL 或文件路径

    返回:
        JSON: 包含检测列表(detections)、推理速度(speed_ms)、终端日志(terminal)、
              标注图(annotated_b64)
    """
    data = request.get_json() or {}
    model_name = data.get("model", "yolo26n.pt")
    task = data.get("task", "detect")
    conf = float(data.get("conf", 0.25))
    iou = float(data.get("iou", 0.7))
    imgsz = int(data.get("imgsz", 640))
    max_det = int(data.get("max_det", 300))
    device = data.get("device", "mps")  # MPS加速(Mac) / CUDA / CPU
    source = data.get("source")         # base64 image 或 URL 或文件路径

    try:
        model = get_model(model_name, task=task if task != "detect" else None)
    except Exception as e:
        return jsonify({"error": str(e)}), 500

    # 如果未提供 source，使用 ultralytics 自带的示例图片
    if not source:
        from ultralytics.utils import ASSETS
        source = str(Path(ASSETS) / "bus.jpg")

    # 根据 task 选择推理方法：追踪用 model.track()，其余用 model()
    from PIL import Image, ImageOps
    do_predict = lambda m, src, **kw: m.track(src, persist=True, **kw) if task == 'track' else m(src, **kw)
    try:
        if source.startswith("data:image"):
            # 处理 base64 编码的图片数据
            header, encoded = source.split(",", 1)
            img_bytes = base64.b64decode(encoded)
            img = Image.open(io.BytesIO(img_bytes))
            img = ImageOps.exif_transpose(img)  # 修正 EXIF 旋转（手机照片常见）
            results = do_predict(model, img, conf=conf, iou=iou, imgsz=imgsz, max_det=max_det, device=device, verbose=False)
        else:
            # 处理图片 URL 或本地文件路径
            import re
            if source and not re.match(r'^https?://', source):
                src_path = Path(source)
                if src_path.exists():
                    img = ImageOps.exif_transpose(Image.open(src_path))
                    results = do_predict(model, img, conf=conf, iou=iou, imgsz=imgsz, max_det=max_det, device=device, verbose=False)
                else:
                    # 路径不存在时直接传给 YOLO 自动解析（可能是数据集路径）
                    results = do_predict(model, source, conf=conf, iou=iou, imgsz=imgsz, max_det=max_det, device=device, verbose=False)
            else:
                results = do_predict(model, source, conf=conf, iou=iou, imgsz=imgsz, max_det=max_det, device=device, verbose=False)
    except Exception as e:
        return jsonify({"error": f"推理失败: {e}"}), 500

    # 构造终端风格的输出文本（模拟 YOLO CLI 的打印风格）
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
            # 优先使用 YOLO 自带的原始图像（BGR numpy array）
            r0 = results[0]
            if hasattr(r0, 'orig_img') and r0.orig_img is not None:
                import numpy as np
                arr = r0.orig_img
                if isinstance(arr, np.ndarray):
                    orig_img = Image.fromarray(arr[..., ::-1])  # BGR → RGB 转换
                else:
                    orig_img = Image.fromarray(np.array(arr))
            elif source.startswith("data:image"):
                # base64 来源：重新解码（保持 EXIF 修正）
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


# ════════════════════════════════════════════════════════
# API 路由：训练
# ════════════════════════════════════════════════════════

TRAIN_LOG_DIR = Path("runs/train_logs")
TRAIN_LOG_DIR.mkdir(parents=True, exist_ok=True)

@app.route("/api/train", methods=["POST"])
def start_training():
    """
    启动模型训练任务（异步）。

    接收训练参数，在后台线程中执行 model.train()，训练输出
    重定向到日志文件。训练开始后立即返回 job_id，前端可轮询
    /api/train/status/<job_id> 获取训练进度和日志。

    请求体 JSON 字段:
        - model (str): 模型文件名
        - 其他 ultralytics 支持的训练参数（data, epochs, batch, imgsz 等）
    返回:
        JSON: {"job_id": "xxxxxxxxxxxx"} — 12 位十六进制任务标识
    """
    data = request.get_json() or {}
    job_id = uuid.uuid4().hex[:12]
    log_path = TRAIN_LOG_DIR / f"{job_id}.log"

    with job_lock:
        TRAIN_JOBS[job_id] = {"status": "running", "model": data.get("model", "yolo26n.pt"), "log": str(log_path)}

    def _train():
        """后台训练线程：捕获 stdout 输出到日志文件"""
        try:
            import io, sys
            model = get_model(data["model"])
            params = {k: v for k, v in data.items() if k != "model"}
            # 将训练输出重定向到 StringIO 以便写入日志文件
            old_stdout = sys.stdout
            sys.stdout = tee = io.StringIO()
            try:
                model.train(**params)
                output = tee.getvalue()
            finally:
                sys.stdout = old_stdout
            log_path.write_text(output, encoding="utf-8")
            with job_lock:
                TRAIN_JOBS[job_id]["status"] = "finished"
        except Exception as e:
            import traceback
            log_path.write_text(traceback.format_exc(), encoding="utf-8")
            with job_lock:
                TRAIN_JOBS[job_id]["status"] = f"error: {e}"

    threading.Thread(target=_train, daemon=True).start()
    return jsonify({"job_id": job_id})


@app.route("/api/train/status/<job_id>")
def train_status(job_id):
    """
    查询训练任务状态和实时日志。

    返回最近的 8000 字符日志内容，供前端渐进式展示训练进度。
    任务状态包括: "running" / "finished" / "error: ..." / "not found"

    参数:
        job_id (str, URL路径参数): 训练任务标识
    返回:
        JSON: {"status": str, "log": str, "job_id": str}
    """
    with job_lock:
        job = TRAIN_JOBS.get(job_id, {"status": "not found"})
    log_path = job.get("log", "")
    log_text = ""
    if log_path and Path(log_path).exists():
        try:
            log_text = Path(log_path).read_text(encoding="utf-8")[-8000:]
        except Exception:
            pass
    return jsonify({"status": job.get("status", "not found"), "log": log_text, "job_id": job_id})


# ════════════════════════════════════════════════════════
# API 路由：验证
# ════════════════════════════════════════════════════════

@app.route("/api/val", methods=["POST"])
def run_validation():
    """
    运行模型验证（model.val()）。

    对指定模型在给定数据集上执行验证，返回验证指标和终端输出日志。
    验证是同步执行的，适合小数据集快速验证。

    请求体 JSON 字段:
        - model (str): 模型文件名
        - data (str): 数据集配置文件路径，默认 "coco8.yaml"
        - imgsz (int): 图像尺寸，默认 640
        - device (str): 计算设备，默认 "cpu"
        - split (str): 数据集划分，默认 "val"
    返回:
        JSON: {"status": "ok"/"error", "output": str, "metrics": dict}
    """
    data = request.get_json() or {}
    model_name = data.get("model", "yolo26n.pt")
    dataset = data.get("data", "coco8.yaml")
    imgsz = int(data.get("imgsz", 640))
    device = data.get("device", "cpu")
    split = data.get("split", "val")
    try:
        model = get_model(model_name)
        import io, sys
        old_stdout = sys.stdout
        sys.stdout = tee = io.StringIO()
        try:
            metrics = model.val(data=dataset, imgsz=imgsz, device=device, split=split, verbose=True)
        finally:
            sys.stdout = old_stdout
        output = tee.getvalue()
        result = {"status": "ok", "output": output[-3000:], "metrics": {}}
        if hasattr(metrics, 'results_dict'):
            result["metrics"] = {k: round(float(v), 4) if isinstance(v, (int, float)) else v
                                for k, v in metrics.results_dict.items()}
        return jsonify(result)
    except Exception as e:
        import traceback
        return jsonify({"status": "error", "error": str(e), "trace": traceback.format_exc()[-2000:]})


# ════════════════════════════════════════════════════════
# API 路由：系统信息
# ════════════════════════════════════════════════════════

@app.route("/api/system/info")
def sys_info():
    """
    返回系统硬件与软件环境信息。

    自动检测可用计算设备（MPS / CUDA / CPU）、PyTorch 版本
    和系统总内存（macOS 统一内存 / Linux / Windows）。

    返回:
        JSON: {"device": str, "torch_version": str, "memory_gb": int}
    """
    import torch
    d = "mps" if torch.backends.mps.is_available() else ("cuda" if torch.cuda.is_available() else "cpu")
    # 检测可用内存（macOS 统一内存 / Linux / Windows）
    mem_gb = 0
    try:
        import psutil
        mem_gb = round(psutil.virtual_memory().total / (1024**3))
    except Exception:
        try:
            import subprocess
            out = subprocess.check_output(["sysctl", "-n", "hw.memsize"], text=True).strip()
            mem_gb = round(int(out) / (1024**3))
        except Exception:
            mem_gb = 4  # fallback
    return jsonify({"device": d, "torch_version": torch.__version__, "memory_gb": mem_gb})


@app.route("/api/health")
def health():
    """
    健康检查端点。

    返回服务器运行状态、已缓存的模型列表和当前时间戳。
    前端心跳检测和服务器状态监控使用此接口。

    返回:
        JSON: {"status": "ok", "cached_models": [str, ...], "timestamp": str}
    """
    return jsonify({"status": "ok", "cached_models": list(MODEL_CACHE.keys()),
                    "timestamp": datetime.now(timezone.utc).isoformat()})


# ════════════════════════════════════════════════════════
# API 路由：模型管理
# ════════════════════════════════════════════════════════

def _model_downloaded(name):
    """
    检查模型权重文件是否已存在于本地。

    依次检查: weights/ 目录、ultralytics 缓存目录（hub/checkpoints）、
    ~/.cache/torch/hub/ultralytics/ 缓存、以及运行时内存缓存 MODEL_CACHE。

    参数:
        name (str): 模型文件名
    返回:
        bool: 模型是否已下载
    """
    if (Path("weights") / name).exists():
        return True
    # ultralytics 默认缓存路径
    import ultralytics
    hub_dir = Path(ultralytics.__file__).parent / "hub" / "checkpoints"
    if (hub_dir / name).exists():
        return True
    # torch hub 缓存路径
    home_cache = Path.home() / ".cache" / "torch" / "hub" / "ultralytics" / name
    if home_cache.exists():
        return True
    return name in MODEL_CACHE


@app.route("/api/models/available")
def available_models():
    """
    返回所有可用 YOLO26 变体模型的列表及下载状态。

    遍历五种规格（n/s/m/l/x）和五种任务（detect/segment/pose/classify/obb），
    生成模型清单供前端展示和下载选择。

    返回:
        JSON: 模型信息数组，每项包含 name/size/label/task/downloaded 字段
    """
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
    """
    下载 YOLO26 模型权重文件。

    若 weights/ 目录下已有该文件则直接返回缓存状态；
    否则通过 ultralytics.YOLO() 触发热加载（自动下载），
    然后尝试从缓存目录复制到 weights/ 以便持久化存储。

    请求体 JSON 字段:
        - model (str): 要下载的模型文件名
    返回:
        JSON: {"success": bool, "model": str, "status": str, "size_mb": float}
    """
    data = request.get_json() or {}
    name = data.get("model", "yolo26n.pt")
    p = Path("weights") / name
    if p.exists():
        return jsonify({"success": True, "model": name, "status": "cached",
                       "size_mb": round(p.stat().st_size / 1024 / 1024, 1)})
    try:
        model = get_model(name)
        # ultralytics 默认下载到 hub 缓存，尝试拷贝到 weights/ 目录下
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
    """
    返回按任务类型分组的模型列表。

    与 available_models 不同，此接口将模型按任务分组
    （detect/segment/pose/classify/obb），供前端下载网格布局使用。

    返回:
        JSON: {"variants": {task: [{name, size, task, downloaded}, ...]}}
    """
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
    """
    返回当前运行时内存中已加载的模型名称列表。

    用于前端判断是否需要重新加载模型，避免重复下载。

    返回:
        JSON: [str, ...] — 内存缓存的模型键名列表
    """
    return jsonify(list(MODEL_CACHE.keys()))


# ════════════════════════════════════════════════════════
# API 路由：文件上传
# ════════════════════════════════════════════════════════

UPLOAD_DIR = Path("uploads")
UPLOAD_DIR.mkdir(exist_ok=True)

@app.route("/api/upload", methods=["POST"])
def upload_file():
    """
    上传图片文件到服务器。

    将上传的文件保存到 uploads/ 目录，并生成唯一文件名避免冲突。
    返回文件路径供后续推理接口（/api/predict）使用。

    请求体: multipart/form-data，字段名 "file"
    返回:
        JSON: {"filename": str, "path": str, "size": int}
    """
    if "file" not in request.files:
        return jsonify({"error": "缺少 file"}), 400
    f = request.files["file"]
    fname = f"{uuid.uuid4().hex}_{f.filename}"
    fpath = UPLOAD_DIR / fname
    f.save(str(fpath))
    return jsonify({"filename": fname, "path": str(fpath.resolve()), "size": fpath.stat().st_size})


# ════════════════════════════════════════════════════════
# API 路由：检测历史管理
# ════════════════════════════════════════════════════════

HISTORY_DIR = Path("history")
HISTORY_DIR.mkdir(exist_ok=True)

@app.route("/api/history/save", methods=["POST"])
def save_history():
    """
    保存检测结果到磁盘。

    接收 base64 编码的标注图像和检测元数据 JSON，
    以时间戳（毫秒精度）为文件名，同时保存 .jpg 和 .json 文件。
    用于检测历史记录的回溯查看。

    请求体 JSON 字段:
        - image (str): base64 编码的图像数据 ("data:image/...")
        - meta (dict): 检测元数据（模型、任务、检测列表等）
    返回:
        JSON: {"id": str, "status": "saved"}
    """
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
    """
    列出最近的检测历史记录（最多 50 条）。

    扫描 history/ 目录下的 JSON 文件，按文件名（即时间戳）倒序排列。
    每条记录包含是否同时存在对应图片文件的标记。

    返回:
        JSON: [dict, ...] — 历史记录数组，每项含元数据及 has_image 标识
    """
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
    """
    获取指定历史记录的原始或标注图片。

    参数:
        entry_id (str, URL路径参数): 历史记录 ID（时间戳字符串）
    返回:
        image/jpeg 文件 或 404
    """
    img_path = HISTORY_DIR / f"{entry_id}.jpg"
    if img_path.exists():
        return send_file(str(img_path), mimetype="image/jpeg")
    return "not found", 404


@app.route("/api/history/delete/<entry_id>", methods=["DELETE"])
def delete_history(entry_id):
    """
    删除单条检测历史记录（图片 + JSON 文件）。

    参数:
        entry_id (str, URL路径参数): 历史记录 ID
    返回:
        JSON: {"status": "deleted"}
    """
    for ext in [".jpg", ".json"]:
        p = HISTORY_DIR / f"{entry_id}{ext}"
        if p.exists():
            p.unlink()
    return jsonify({"status": "deleted"})


@app.route("/api/history/clear-all", methods=["DELETE"])
def clear_all_history():
    """
    清空全部检测历史记录。

    删除 history/ 目录下的所有 .jpg 和 .json 文件，
    返回实际删除的文件数量。

    返回:
        JSON: {"status": "cleared", "count": int}
    """
    count = 0
    for p in list(HISTORY_DIR.glob("*.jpg")) + list(HISTORY_DIR.glob("*.json")):
        try:
            p.unlink()
            count += 1
        except Exception:
            pass
    return jsonify({"status": "cleared", "count": count})


# ════════════════════════════════════════════════════════
# API 路由：静态文件与前端页面
# ════════════════════════════════════════════════════════

@app.route("/<path:filename>")
def serve_static(filename):
    """
    提供静态文件服务。

    用于提供图片、CSS、JavaScript 等资源文件。
    如果文件不存在则返回 404。

    参数:
        filename (str, URL路径): 请求的文件路径
    返回:
        文件内容或 404
    """
    from flask import abort
    path = Path(filename)
    if not path.exists():
        abort(404)
    return send_file(str(path.resolve()))


@app.route("/")
def serve_index():
    """
    提供前端仪表盘主页（Yolo26.html）。

    读取同级目录下的 Yolo26.html 文件，以 text/html 响应返回。
    设置 Cache-Control 为 no-cache，确保浏览器每次都获取最新版本。
    HTTP 协议访问（非 file://），确保摄像头权限正常工作。

    返回:
        text/html: Yolo26.html 内容 或 404
    """
    index_path = Path(__file__).parent / "Yolo26.html"
    if index_path.exists():
        from flask import Response
        return Response(index_path.read_text(encoding="utf-8"), mimetype="text/html",
                       headers={"Cache-Control":"no-cache, no-store, must-revalidate",
                               "Pragma":"no-cache","Expires":"0"})
    return "<h1>Yolo26.html 未找到</h1>", 404


# ════════════════════════════════════════════════════════
# 程序入口
# ════════════════════════════════════════════════════════

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
