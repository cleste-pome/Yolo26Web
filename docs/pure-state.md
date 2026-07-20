# 纯净态（Pure State）：AI 模型状态管理的范式

> **「纯净态」是模型最原始、未经任何修改的基准状态——就像 Git 仓库的初始 commit，Docker 的 base image，音乐的母带。它是所有实验的起点，所有部署的溯源根。**

---

## 1. 什么是纯净态？

在 YOLO 生态系统中，当你从 GitHub Releases 下载 `yolo26n.pt` 的那一刻，它就是**纯净态**——一个具有确定 SHA256 指纹、未经任何训练/微调/量化的原始权重快照。

### 1.1 形式化定义

设：
- $M_0$ 为纯净态模型，参数 $\theta_0$ 由预训练确定
- 训练过程 $\mathcal{T}(\theta_0, \mathcal{D}, \mathcal{H}) \rightarrow \theta_t$，其中 $\mathcal{D}$ 为数据集，$\mathcal{H}$ 为超参数
- 导出过程 $\mathcal{E}(\theta_t, \mathcal{F}) \rightarrow M_{export}$，其中 $\mathcal{F}$ 为目标格式

**完整谱系链**：
```
M_0 ──[训练]──▶ M_t ──[导出]──▶ M_export
纯净态            训练态           导出态
```

每一步转换都必须是**可审计、可复现、可回滚**的。

### 1.2 核心属性

| 属性 | 说明 |
|------|------|
| **不可变性** | 纯净态一旦注册，其文件内容（SHA256）绝不被修改 |
| **完整性** | SHA256 校验和确保文件未被损坏或篡改 |
| **可追溯性** | 任何训练态和导出态都能回溯到其纯净态源头 |
| **可分支性** | 从任意状态 fork 新分支，独立实验互不干扰 |

---

## 2. 为什么需要纯净态管理？

### 2.1 当前痛点

| 问题 | 纯净态如何解决 |
|------|---------------|
| **模型腐败**：反复微调后，无法回到原始性能基线 | 纯净态始终作为不可变的参照点存在 |
| **实验不可复现**：不知道当前模型是基于哪个基准训练的 | 完整谱系记录每次训练的参数和数据集 |
| **检查点爆炸**：`best.pt`、`last.pt`、`epoch50.pt`…缺少元数据 | 每个训练态自动关联实验配置和指标 |
| **团队协作混乱**：多人训练不同版本，无法追踪改动 | Git 式的分支和 commit 模型 |
| **部署溯源**：线上模型出问题时，无法快速定位训练来源 | 从部署态逆向遍历谱系到纯净态 |

### 2.2 与 Git 的深度类比

| Git | 纯净态系统 |
|-----|-----------|
| `git init` | 注册首个纯净态 |
| `git commit` | 保存训练态快照（含指标和配置） |
| `git branch` | 从任意状态分叉并行实验 |
| `git diff` | 对比两个状态的指标差异 |
| `git log` | 查看完整谱系链 |
| `git checkout` | 回滚到任意历史状态 |
| `git tag` | 给关键状态打标签（`production`、`baseline`） |
| `.git/objects/` | Content-addressed SHA256 存储 |
| `git gc` | 清理文件已删除的孤立状态 |

---

## 3. YOLO 的三态体系

```
┌─────────────────────────────────────────────────────────────────┐
│                      YOLO 模型三态体系                            │
├─────────────────────────────────────────────────────────────────┤
│                                                                 │
│   ┌──────────┐       训练          ┌──────────┐       导出        ┌──────────┐
│   │ 纯净态    │ ───────────────▶ │ 训练态    │ ─────────────▶ │ 导出态    │
│   │ (Pure)   │                   │ (Trained) │                   │ (Export)  │
│   ├──────────┤                   ├──────────┤                   ├──────────┤
│   │ yolo26n  │                   │ best.pt  │                   │ .onnx     │
│   │ .pt      │                   │ last.pt  │                   │ .engine   │
│   │ SHA:abc  │                   │ SHA:def  │                   │ .mlpackage│
│   │          │                   │ mAP: 48.6 │                   │ .tflite   │
│   └──────────┘                   └──────────┘                   └──────────┘
│        │                              │                              │
│        │                              │                              │
│        ▼                              ▼                              ▼
│   不可变基准                     完整训练谱系                    部署就绪
│   SHA256 验证                   指标 + 配置                    跨平台格式
│                                                                 │
│   ┌──────────────────────────────────────────────────────────┐  │
│   │                    分支与并行实验                            │  │
│   │                                                          │  │
│   │  yolo26n.pt (纯净态)                                      │  │
│   │       │                                                   │  │
│   │       ├── branch: coco-finetune ──▶ best-coco.pt          │  │
│   │       │       └── export: onnx ──▶ model.onnx             │  │
│   │       │                                                   │  │
│   │       ├── branch: custom-data ──▶ best-custom.pt          │  │
│   │       │       ├── export: tensorrt ──▶ model.engine       │  │
│   │       │       └── export: coreml ──▶ model.mlpackage      │  │
│   │       │                                                   │  │
│   │       └── branch: ablation-lr ──▶ best-lr001.pt           │  │
│   │               └── compare ──▶ diff vs coco-finetune       │  │
│   └──────────────────────────────────────────────────────────┘  │
│                                                                 │
└─────────────────────────────────────────────────────────────────┘
```

### 3.1 纯净态（Pure State）

```python
from state_manager import StateManager

manager = StateManager("states.db")

# 注册 YOLO26 Nano 的纯净态
pure = manager.register_pure_state(
    name="YOLO26 Nano Base",
    model_variant="yolo26n",
    file_path="yolo26n.pt",
    metadata={"source": "GitHub Releases", "version": "8.4.0"}
)
print(f"纯净态已注册: SHA256={pure.sha256[:16]}...")
```

**何时创建纯净态**：
- 下载新的预训练模型时
- 从他人获取基准权重时
- 初始化项目时

### 3.2 训练态（Trained State）

```python
# 训练完成后自动注册（由 server.py 处理）
# 或手动注册：
trained = manager.register_trained_state(
    name="COCO微调-实验A",
    pure_state_id=pure.id,
    file_path="runs/detect/train/weights/best.pt",
    experiment_id=1,
    metrics={"mAP50-95": 0.486, "mAP50": 0.685}
)
```

**训练态自动记录**：
- 派生自哪个纯净态
- 训练使用的超参数
- 验证集指标
- 父训练态（分叉时）

### 3.3 导出态（Export State）

```python
export = manager.register_export_state(
    trained_state_id=trained.id,
    format_name="onnx",
    file_path="model.onnx",
    config={"opset": 17, "simplify": True}
)
```

---

## 4. 实操指南

### 4.1 启动后端服务

```bash
cd yolo26-purestate
pip install -r requirements.txt
python server.py --port 5000
```

### 4.2 前端仪表盘

在浏览器中打开 `index.html`，前端会自动连接到 `localhost:5000` 的后端 API。如果后端不可用，将显示模拟数据。

### 4.3 CLI 操作

```bash
# 注册纯净态
python state_manager.py register --name "YOLO26n" --variant yolo26n --file yolo26n.pt

# 查看所有纯净态
python state_manager.py list

# 查看谱系树
python state_manager.py tree --id 1

# 对比两个训练态
python state_manager.py compare --id1 1 --id2 2

# 验证完整性
python state_manager.py verify

# 系统统计
python state_manager.py stats

# 清理孤立状态
python state_manager.py gc
```

### 4.4 REST API 核心端点

| 方法 | 端点 | 功能 |
|------|------|------|
| POST | `/api/train` | 启动训练任务 |
| GET | `/api/train/status/<id>` | 训练进度（含实时日志） |
| GET | `/api/experiments` | 所有实验列表 |
| GET | `/api/experiments/<id>` | 实验详情 |
| GET | `/api/metrics/<id>` | 训练指标数据 |
| GET | `/api/pure-states` | 所有纯净态 |
| GET | `/api/trained-states` | 所有训练态 |
| GET | `/api/states/lineage/<id>` | 谱系追溯 |
| GET | `/api/states/lineage-tree/<id>` | 完整谱系树 |
| POST | `/api/states/compare` | 状态对比 |
| POST | `/api/export` | 导出模型 |
| GET | `/api/stats` | 系统统计 |

---

## 5. 与现有工具对比

| 维度 | 纯净态系统 | MLflow | DVC | W&B |
|------|-----------|--------|-----|-----|
| 模型谱系 | ✅ 完整三态追踪 | ❌ 仅跟踪实验 | ❌ 仅数据版本 | ❌ 仅实验日志 |
| SHA256 完整性 | ✅ 内置 | ❌ 需手动 | ✅ 对于数据 | ❌ |
| 分支管理 | ✅ Git 式 | ❌ | ❌ | ❌ |
| 离线可用 | ✅ 纯本地 SQLite | ⚠️ 需服务器 | ✅ | ❌ 需联网 |
| 与 YOLO 集成 | ✅ 原生 | ⚠️ 需回调 | ⚠️ | ⚠️ 需回调 |
| 学习成本 | 低（Git 用户直觉）| 中 | 中 | 低 |

**定位**：纯净态系统**不是** MLflow/W&B 的替代品，而是它们的**补充**——前者管理「实验过程」，纯净态管理「模型产物」。

---

## 6. 高级话题

### 6.1 Delta 压缩存储

受 [syckpt](https://pypi.org/project/syckpt/) 启发，相邻训练态的权重差 $\Delta W = W_t - W_{t-1}$ 远小于完整权重，可实现 10-50× 的存储压缩。纯净态系统预留了此扩展能力。

### 6.2 跨模型迁移学习

当一个模型（如 YOLO26n）的 backbone 被用于初始化另一个模型时，纯净态系统可以追踪这种跨模型的谱系关系。

### 6.3 联邦学习场景

在联邦学习中，中心服务器维护纯净态，各客户端提交训练态增量。SHA256 确保权重传输完整性。

---

## 7. 最佳实践清单

- [ ] **下载即注册**：每次获取新预训练模型后，立即注册纯净态
- [ ] **训练前记录**：启动训练前，确保 `pure_state_ref` 字段已填入
- [ ] **定期验证**：每周运行 `state_manager.py verify` 检查文件完整性
- [ ] **语义化命名**：使用描述性名称（如 `coco-finetune-lr001-bs16`）而非 `exp-001`
- [ ] **定期 GC**：每月运行 `state_manager.py gc` 清理已删除文件的状态记录
- [ ] **谱系文档化**：每次导出生产模型时，附带完整谱系 JSON
- [ ] **备份数据库**：`states.db` 和 `experiments.db` 应纳入版本控制或定期备份

---

## 8. 总结

纯净态管理的本质是将**软件工程的版本控制哲学**引入**AI 模型管理**。就像你不会在没有 Git 的情况下写代码一样，你也不应该在没有纯净态管理的情况下训练模型。每个 `best.pt` 都应该能回答三个问题：

1. **它从哪里来？**（哪个纯净态？什么配置？）
2. **它经历了什么？**（什么数据？多少 epoch？什么指标？）
3. **它能去哪里？**（导出到什么格式？部署到哪里？）

纯净态系统让这些问题的答案**自动记录、随时可查、永不丢失**。
