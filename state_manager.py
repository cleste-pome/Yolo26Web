#!/usr/bin/env python3
"""
纯净态（Pure State）模型状态管理器
=====================================

Git 启发式的 YOLO 模型版本控制系统。
提供不可变基准模型（PureState）、训练派生模型（TrainedState）、
导出部署模型（ExportState）以及实验分支（StateBranch）的完整生命周期管理。

核心概念:
    PureState   — 不可变的基准模型（如 yolo26n.pt 原始权重），SHA256 哈希标识
    TrainedState — 从 PureState 通过训练派生的状态，记录完整血统
    ExportState  — 从 TrainedState 导出为部署格式的状态
    StateBranch  — 命名分支，支持并行实验轨道
    StateManager — 核心编排器，统一管理所有状态

使用示例:
    >>> manager = StateManager("states.db")
    >>> pure = manager.register_pure_state("yolo26n-v1", "yolo26n", "weights/yolo26n.pt")
    >>> trained = manager.register_trained_state("exp001", pure.id, "runs/exp001/best.pt")
    >>> lineage = trained.get_lineage(manager)
"""

import hashlib
import json
import os
import sqlite3
from collections import deque
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

# ---------------------------------------------------------------------------
# 工具函数
# ---------------------------------------------------------------------------

def _compute_sha256(file_path: Union[str, Path]) -> str:
    """
    计算文件的 SHA256 哈希值。

    Args:
        file_path: 文件路径（字符串或 Path 对象）

    Returns:
        十六进制 SHA256 摘要字符串

    Raises:
        FileNotFoundError: 文件不存在时抛出
    """
    path = Path(file_path)
    if not path.exists():
        raise FileNotFoundError(f"文件不存在: {path}")
    sha256_hash = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(8192), b""):
            sha256_hash.update(chunk)
    return sha256_hash.hexdigest()


def _now_iso() -> str:
    """返回当前 UTC 时间的 ISO 8601 格式字符串。"""
    return datetime.now(timezone.utc).isoformat()


# ---------------------------------------------------------------------------
# PureState — 不可变的基准模型状态
# ---------------------------------------------------------------------------

class PureState:
    """
    不可变的基准模型状态。

    表示一个原始模型文件（如 YOLO 官方发布的权重），
    由 SHA256 哈希唯一标识，不可被修改。
    所有 TrainedState 都直接或间接派生自某个 PureState。

    Attributes:
        id: 数据库分配的唯一 ID（注册前为 None）
        name: 人类可读的名称
        model_variant: 模型变体标识（如 "yolo26n"）
        file_path: 权重文件的绝对路径
        sha256: 文件的 SHA256 校验和
        metadata: 附加元数据字典
        created_at: 数据库记录的创建时间戳
    """

    def __init__(
        self,
        name: str,
        model_variant: str,
        file_path: Union[str, Path],
        metadata: Optional[Dict[str, Any]] = None,
    ):
        """
        初始化一个 PureState。

        Args:
            name: 状态名称（如 "yolo26n-official"）
            model_variant: 模型变体（如 "yolo26n", "yolo26s"）
            file_path: 模型文件路径
            metadata: 可选的额外元数据字典
        """
        self.id: Optional[int] = None
        self.name: str = name
        self.model_variant: str = model_variant
        self.file_path: str = str(Path(file_path).resolve())
        self.sha256: str = _compute_sha256(self.file_path)
        self.metadata: Dict[str, Any] = metadata or {}
        self.created_at: Optional[str] = None

    def _compute_hash(self) -> str:
        """
        重新计算文件的 SHA256 哈希。

        用于验证文件自注册以来是否发生过变化。

        Returns:
            十六进制 SHA256 摘要字符串
        """
        return _compute_sha256(self.file_path)

    def to_dict(self) -> Dict[str, Any]:
        """
        将 PureState 序列化为字典。

        Returns:
            包含所有字段的字典，适用于 JSON 序列化或血统导出
        """
        return {
            "id": self.id,
            "name": self.name,
            "model_variant": self.model_variant,
            "file_path": self.file_path,
            "sha256": self.sha256,
            "metadata": self.metadata,
            "created_at": self.created_at,
        }

    @classmethod
    def from_db_row(cls, row: sqlite3.Row) -> "PureState":
        """
        从数据库行对象反序列化为 PureState。

        不重新计算哈希，直接使用数据库中存储的值。

        Args:
            row: sqlite3.Row 对象，包含 pure_states 表的所有列

        Returns:
            一个新的 PureState 实例
        """
        obj = cls.__new__(cls)
        obj.id = row["id"]
        obj.name = row["name"]
        obj.model_variant = row["model_variant"]
        obj.file_path = row["file_path"]
        obj.sha256 = row["sha256"]
        obj.metadata = json.loads(row["metadata_json"])
        obj.created_at = row["created_at"]
        return obj

    def __repr__(self) -> str:
        return (
            f"PureState(id={self.id}, name={self.name!r}, "
            f"variant={self.model_variant!r}, sha256={self.sha256[:12]}...)"
        )


# ---------------------------------------------------------------------------
# TrainedState — 训练派生状态
# ---------------------------------------------------------------------------

class TrainedState:
    """
    从 PureState 通过训练派生的状态。

    每次训练运行产生一个 TrainedState，包含训练指标、
    实验 ID、父状态引用（用于分支）以及文件哈希。
    支持沿 parent_state_id 链回溯到原始 PureState。

    Attributes:
        id: 数据库分配的唯一 ID
        name: 训练运行的名称
        pure_state_id: 关联的 PureState ID
        file_path: 训练后权重文件的绝对路径
        sha256: 文件的 SHA256 校验和
        experiment_id: 可选的实验标识
        parent_state_id: 父 TrainedState ID（用于分支链，
                         为 None 表示直接从 PureState 派生）
        metrics: 训练指标字典（如 {"mAP50": 0.85, "mAP50-95": 0.62}）
        created_at: 数据库记录的创建时间戳
    """

    def __init__(
        self,
        name: str,
        pure_state: Union["PureState", int],
        file_path: Union[str, Path],
        experiment_id: Optional[int] = None,
        parent_state: Optional[Union["TrainedState", int]] = None,
        metrics: Optional[Dict[str, Any]] = None,
    ):
        """
        初始化一个 TrainedState。

        Args:
            name: 训练运行的名称（如 "exp001-coco-baseline"）
            pure_state: PureState 实例或其数据库 ID
            file_path: 训练后的权重文件路径
            experiment_id: 可选的实验 ID，用于关联外部实验管理系统
            parent_state: 父 TrainedState 实例或其 ID（用于链式训练/分支）
            metrics: 可选的训练指标字典
        """
        self.id: Optional[int] = None
        self.name: str = name
        self.pure_state_id: int = (
            pure_state.id if isinstance(pure_state, PureState) else pure_state
        )
        self.file_path: str = str(Path(file_path).resolve())
        self.sha256: str = _compute_sha256(self.file_path)
        self.experiment_id: Optional[int] = experiment_id
        self.parent_state_id: Optional[int] = (
            parent_state.id
            if isinstance(parent_state, TrainedState)
            else parent_state
        )
        self.metrics: Dict[str, Any] = metrics or {}
        self.created_at: Optional[str] = None

    def _compute_hash(self) -> str:
        """
        重新计算文件的 SHA256 哈希。

        用于验证文件自注册以来是否发生过变化。

        Returns:
            十六进制 SHA256 摘要字符串
        """
        return _compute_sha256(self.file_path)

    def to_dict(self) -> Dict[str, Any]:
        """
        将 TrainedState 序列化为字典。

        Returns:
            包含所有字段的字典，适用于 JSON 序列化或血统导出
        """
        return {
            "id": self.id,
            "name": self.name,
            "pure_state_id": self.pure_state_id,
            "file_path": self.file_path,
            "sha256": self.sha256,
            "experiment_id": self.experiment_id,
            "parent_state_id": self.parent_state_id,
            "metrics": self.metrics,
            "created_at": self.created_at,
        }

    @classmethod
    def from_db_row(cls, row: sqlite3.Row) -> "TrainedState":
        """
        从数据库行对象反序列化为 TrainedState。

        不重新计算哈希，直接使用数据库中存储的值。

        Args:
            row: sqlite3.Row 对象，包含 trained_states 表的所有列

        Returns:
            一个新的 TrainedState 实例
        """
        obj = cls.__new__(cls)
        obj.id = row["id"]
        obj.name = row["name"]
        obj.pure_state_id = row["pure_state_id"]
        obj.file_path = row["file_path"]
        obj.sha256 = row["sha256"]
        obj.experiment_id = row["experiment_id"]
        obj.metrics = json.loads(row["metrics_json"])
        obj.parent_state_id = row["parent_state_id"]
        obj.created_at = row["created_at"]
        return obj

    def get_lineage(self, db: Union[sqlite3.Connection, "StateManager"]) -> List[Dict[str, Any]]:
        """
        回溯血统链：从当前 TrainedState 沿父状态链一直追溯到 PureState。

        沿着 parent_state_id 链逐级向上查询，直到遇到 parent_state_id
        为 None 的根 TrainedState，最后添加 PureState 作为血统的起点。

        Args:
            db: sqlite3.Connection 或 StateManager 实例。
                如果传入 StateManager，将使用其内部连接。

        Returns:
            按时间顺序排列的血统节点列表（最旧的 PureState 在最前，
            当前状态在最后），每个节点为包含 _type 和数据的字典。
            _type 取值为 "pure_state" 或 "trained_state"。

        Example:
            >>> ts = manager.get_trained_state(3)
            >>> lineage = ts.get_lineage(manager)
            >>> for node in lineage:
            ...     print(node["_type"], node["name"])
            pure_state yolo26n-official
            trained_state exp001-baseline
            trained_state exp003-finetune
        """
        conn = db.conn if isinstance(db, StateManager) else db
        lineage: List[Dict[str, Any]] = []
        visited: set = set()

        # 从当前 TrainedState 开始回溯
        current = self.to_dict()
        current["_type"] = "trained_state"
        lineage.append(current)
        visited.add(("trained", self.id))

        parent_id = self.parent_state_id
        while parent_id is not None and ("trained", parent_id) not in visited:
            cursor = conn.execute(
                "SELECT * FROM trained_states WHERE id = ?", (parent_id,)
            )
            row = cursor.fetchone()
            if row is None:
                break
            ts = TrainedState.from_db_row(row)
            node = ts.to_dict()
            node["_type"] = "trained_state"
            lineage.append(node)
            visited.add(("trained", ts.id))
            parent_id = ts.parent_state_id

        # 最后添加 PureState 作为血统根节点
        if self.pure_state_id is not None:
            cursor = conn.execute(
                "SELECT * FROM pure_states WHERE id = ?", (self.pure_state_id,)
            )
            row = cursor.fetchone()
            if row is not None:
                ps = PureState.from_db_row(row)
                node = ps.to_dict()
                node["_type"] = "pure_state"
                lineage.append(node)

        # 反转顺序：最老的（PureState）在前，最新的在后
        lineage.reverse()
        return lineage

    def __repr__(self) -> str:
        return (
            f"TrainedState(id={self.id}, name={self.name!r}, "
            f"pure_state_id={self.pure_state_id}, sha256={self.sha256[:12]}...)"
        )


# ---------------------------------------------------------------------------
# ExportState — 导出部署状态
# ---------------------------------------------------------------------------

class ExportState:
    """
    从 TrainedState 导出为部署格式的状态。

    支持 ONNX、TensorRT、CoreML、OpenVINO、TFLite 等多种导出格式。
    每种格式和配置组合可产生不同的 ExportState。

    Attributes:
        id: 数据库分配的唯一 ID
        trained_state_id: 关联的 TrainedState ID
        format: 导出格式名称（如 "onnx", "tensorrt"）
        file_path: 导出文件的绝对路径
        sha256: 文件的 SHA256 校验和
        config: 导出配置字典（如精度、输入尺寸、动态轴等）
        created_at: 数据库记录的创建时间戳
    """

    def __init__(
        self,
        trained_state: Union[TrainedState, int],
        format_name: str,
        file_path: Union[str, Path],
        config: Optional[Dict[str, Any]] = None,
    ):
        """
        初始化一个 ExportState。

        Args:
            trained_state: TrainedState 实例或其数据库 ID
            format_name: 导出格式（如 "onnx", "tensorrt", "coreml", "openvino"）
            file_path: 导出文件路径
            config: 可选的导出配置字典（如 {"opset": 17, "half": True}）
        """
        self.id: Optional[int] = None
        self.trained_state_id: int = (
            trained_state.id
            if isinstance(trained_state, TrainedState)
            else trained_state
        )
        self.format: str = format_name
        self.file_path: str = str(Path(file_path).resolve())
        self.sha256: str = _compute_sha256(self.file_path)
        self.config: Dict[str, Any] = config or {}
        self.created_at: Optional[str] = None

    def _compute_hash(self) -> str:
        """
        重新计算文件的 SHA256 哈希。

        Returns:
            十六进制 SHA256 摘要字符串
        """
        return _compute_sha256(self.file_path)

    def to_dict(self) -> Dict[str, Any]:
        """
        将 ExportState 序列化为字典。

        Returns:
            包含所有字段的字典，适用于 JSON 序列化或血统导出
        """
        return {
            "id": self.id,
            "trained_state_id": self.trained_state_id,
            "format": self.format,
            "file_path": self.file_path,
            "sha256": self.sha256,
            "config": self.config,
            "created_at": self.created_at,
        }

    @classmethod
    def from_db_row(cls, row: sqlite3.Row) -> "ExportState":
        """
        从数据库行对象反序列化为 ExportState。

        Args:
            row: sqlite3.Row 对象，包含 export_states 表的所有列

        Returns:
            一个新的 ExportState 实例
        """
        obj = cls.__new__(cls)
        obj.id = row["id"]
        obj.trained_state_id = row["trained_state_id"]
        obj.format = row["format"]
        obj.file_path = row["file_path"]
        obj.sha256 = row["sha256"]
        obj.config = json.loads(row["config_json"])
        obj.created_at = row["created_at"]
        return obj

    def __repr__(self) -> str:
        return (
            f"ExportState(id={self.id}, format={self.format!r}, "
            f"trained_state_id={self.trained_state_id}, sha256={self.sha256[:12]}...)"
        )


# ---------------------------------------------------------------------------
# StateBranch — 命名实验分支
# ---------------------------------------------------------------------------

class StateBranch:
    """
    命名实验分支。

    允许在同一 PureState 基础上创建多个并行实验轨道，
    类似于 Git 分支。每个分支有一个根 TrainedState，
    后续训练可通过 parent_state_id 在该分支上继续派生。

    Attributes:
        id: 数据库分配的唯一 ID
        name: 分支名称（全局唯一）
        root_state_id: 分支根节点的 TrainedState ID
        description: 可选的描述文本
        created_at: 数据库记录的创建时间戳
    """

    def __init__(
        self,
        name: str,
        root_state_id: int,
        description: Optional[str] = None,
    ):
        """
        初始化一个 StateBranch。

        Args:
            name: 分支名称（全局唯一，如 "baseline", "augmentation-experiment"）
            root_state_id: 分支根的 TrainedState ID，作为分支的起点
            description: 可选的描述文本，说明此分支的目的
        """
        self.id: Optional[int] = None
        self.name: str = name
        self.root_state_id: int = root_state_id
        self.description: Optional[str] = description
        self.created_at: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        """
        将 StateBranch 序列化为字典。

        Returns:
            包含所有字段的字典
        """
        return {
            "id": self.id,
            "name": self.name,
            "root_state_id": self.root_state_id,
            "description": self.description,
            "created_at": self.created_at,
        }

    @classmethod
    def from_db_row(cls, row: sqlite3.Row) -> "StateBranch":
        """
        从数据库行对象反序列化为 StateBranch。

        Args:
            row: sqlite3.Row 对象，包含 state_branches 表的所有列

        Returns:
            一个新的 StateBranch 实例
        """
        obj = cls.__new__(cls)
        obj.id = row["id"]
        obj.name = row["name"]
        obj.root_state_id = row["root_state_id"]
        obj.description = row["description"]
        obj.created_at = row["created_at"]
        return obj

    def __repr__(self) -> str:
        return (
            f"StateBranch(id={self.id}, name={self.name!r}, "
            f"root_state_id={self.root_state_id})"
        )


# ---------------------------------------------------------------------------
# StateManager — 核心编排器
# ---------------------------------------------------------------------------

class StateManager:
    """
    纯净态管理器 — 核心编排器。

    提供所有模型状态的注册、查询、血统追溯、分支管理、
    完整性验证和垃圾回收功能。
    底层使用 SQLite 数据库持久化所有元数据，支持内存和文件两种模式。

    使用方式:
        >>> manager = StateManager("states.db")        # 文件数据库，持久化存储
        >>> manager = StateManager()                   # 内存数据库（默认）
        >>> manager = StateManager(":memory:")          # 显式内存数据库

    上下文管理器支持:
        >>> with StateManager("states.db") as manager:
        ...     pure = manager.register_pure_state(...)

    Attributes:
        db_path: SQLite 数据库路径
        conn: 持久化的 sqlite3 连接对象（启用 WAL 模式和外键约束）
    """

    def __init__(self, db_path: str = ":memory:"):
        """
        初始化 StateManager 并创建数据库表。

        Args:
            db_path: SQLite 数据库文件路径。默认为 ":memory:"（内存数据库）。
                     使用文件路径（如 "states.db"）可实现持久化存储。
        """
        self.db_path: str = db_path
        self.conn: sqlite3.Connection = sqlite3.connect(self.db_path, check_same_thread=False)
        self.conn.row_factory = sqlite3.Row
        self.conn.execute("PRAGMA journal_mode=WAL")
        self.conn.execute("PRAGMA foreign_keys=ON")
        self._init_db()

    # ------------------------------------------------------------------
    # 数据库初始化
    # ------------------------------------------------------------------

    def _init_db(self) -> None:
        """
        初始化数据库表结构。

        创建四张核心表和六个加速索引:
            pure_states     — 不可变基准模型状态
            trained_states  — 训练派生模型状态
            export_states   — 导出部署模型状态
            state_branches  — 命名实验分支

        索引:
            idx_trained_pure   — 按 pure_state_id 查询 TrainedState
            idx_trained_parent — 按 parent_state_id 查询派生链
            idx_export_trained — 按 trained_state_id 查询导出状态
            idx_pure_sha256    — 按 SHA256 去重 PureState
            idx_trained_sha256 — 按 SHA256 去重 TrainedState
            idx_export_sha256  — 按 SHA256 去重 ExportState

        如果表已存在则跳过（CREATE TABLE IF NOT EXISTS），
        因此可以安全地多次调用。
        """
        schema = """
        CREATE TABLE IF NOT EXISTS pure_states (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            name TEXT NOT NULL,
            model_variant TEXT NOT NULL,
            file_path TEXT NOT NULL,
            sha256 TEXT NOT NULL,
            metadata_json TEXT DEFAULT '{}',
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        );

        CREATE TABLE IF NOT EXISTS trained_states (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            name TEXT NOT NULL,
            pure_state_id INTEGER NOT NULL REFERENCES pure_states(id),
            file_path TEXT NOT NULL,
            sha256 TEXT NOT NULL,
            experiment_id INTEGER,
            metrics_json TEXT DEFAULT '{}',
            parent_state_id INTEGER REFERENCES trained_states(id),
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        );

        CREATE TABLE IF NOT EXISTS export_states (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            trained_state_id INTEGER NOT NULL REFERENCES trained_states(id),
            format TEXT NOT NULL,
            file_path TEXT NOT NULL,
            sha256 TEXT NOT NULL,
            config_json TEXT DEFAULT '{}',
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        );

        CREATE TABLE IF NOT EXISTS state_branches (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            name TEXT NOT NULL UNIQUE,
            root_state_id INTEGER NOT NULL,
            description TEXT,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        );

        -- 索引：加速常见查询路径
        CREATE INDEX IF NOT EXISTS idx_trained_pure
            ON trained_states(pure_state_id);
        CREATE INDEX IF NOT EXISTS idx_trained_parent
            ON trained_states(parent_state_id);
        CREATE INDEX IF NOT EXISTS idx_export_trained
            ON export_states(trained_state_id);
        CREATE INDEX IF NOT EXISTS idx_pure_sha256
            ON pure_states(sha256);
        CREATE INDEX IF NOT EXISTS idx_trained_sha256
            ON trained_states(sha256);
        CREATE INDEX IF NOT EXISTS idx_export_sha256
            ON export_states(sha256);
        """
        self.conn.executescript(schema)
        self.conn.commit()

    # ------------------------------------------------------------------
    # 注册方法
    # ------------------------------------------------------------------

    def register_pure_state(
        self,
        name: str,
        model_variant: str,
        file_path: Union[str, Path],
        metadata: Optional[Dict[str, Any]] = None,
    ) -> PureState:
        """
        注册一个新的 PureState（基准模型）。

        计算文件的 SHA256 哈希，将元数据写入 pure_states 表。
        如果已存在相同 SHA256 的状态，直接返回已有记录（去重）。

        Args:
            name: 状态名称（如 "YOLO26n 官方权重"）
            model_variant: 模型变体（如 "yolo26n", "yolo26s"）
            file_path: 模型文件路径（.pt 文件）
            metadata: 可选的附加元数据字典

        Returns:
            已注册的 PureState 实例（含数据库分配的 ID）

        Raises:
            FileNotFoundError: 当指定的文件不存在时抛出
        """
        path = Path(file_path).resolve()
        if not path.exists():
            raise FileNotFoundError(f"文件不存在: {path}")

        sha256 = _compute_sha256(path)
        metadata_json = json.dumps(metadata or {}, ensure_ascii=False)

        # 去重：相同 SHA256 的文件视为同一个 PureState
        existing = self.conn.execute(
            "SELECT id FROM pure_states WHERE sha256 = ?", (sha256,)
        ).fetchone()
        if existing:
            return self.get_pure_state(existing["id"])

        with self.conn:
            cursor = self.conn.execute(
                """INSERT INTO pure_states (name, model_variant, file_path, sha256, metadata_json)
                   VALUES (?, ?, ?, ?, ?)""",
                (name, model_variant, str(path), sha256, metadata_json),
            )
            state_id = cursor.lastrowid

        state = PureState(name, model_variant, str(path), metadata)
        state.id = state_id
        state.sha256 = sha256
        # 读取数据库自动生成的时间戳
        row = self.conn.execute(
            "SELECT created_at FROM pure_states WHERE id = ?", (state_id,)
        ).fetchone()
        state.created_at = row["created_at"]
        return state

    def register_trained_state(
        self,
        name: str,
        pure_state_id: int,
        file_path: Union[str, Path],
        experiment_id: Optional[int] = None,
        parent_id: Optional[int] = None,
        metrics: Optional[Dict[str, Any]] = None,
    ) -> TrainedState:
        """
        注册一个新的 TrainedState（训练后模型）。

        Args:
            name: 训练运行的名称（如 "exp001-baseline"）
            pure_state_id: 关联的 PureState 数据库 ID
            file_path: 训练后的权重文件路径（best.pt 或 last.pt）
            experiment_id: 可选的实验 ID，用于关联外部实验管理系统
            parent_id: 可选的父 TrainedState ID，用于链式训练或分支
            metrics: 可选的训练指标字典（如 {"mAP50": 0.85, "mAP50-95": 0.62}）

        Returns:
            已注册的 TrainedState 实例（含数据库分配的 ID）

        Raises:
            FileNotFoundError: 当指定的文件不存在时抛出
            ValueError: 当引用的 PureState 不存在时抛出
        """
        # 验证 PureState 存在
        ps_row = self.conn.execute(
            "SELECT id FROM pure_states WHERE id = ?", (pure_state_id,)
        ).fetchone()
        if ps_row is None:
            raise ValueError(f"PureState id={pure_state_id} 不存在，请先注册")

        # 验证父状态存在（如果指定了）
        if parent_id is not None:
            parent_row = self.conn.execute(
                "SELECT id FROM trained_states WHERE id = ?", (parent_id,)
            ).fetchone()
            if parent_row is None:
                raise ValueError(f"父 TrainedState id={parent_id} 不存在")

        path = Path(file_path).resolve()
        if not path.exists():
            raise FileNotFoundError(f"文件不存在: {path}")

        sha256 = _compute_sha256(path)
        metrics_json = json.dumps(metrics or {}, ensure_ascii=False)

        with self.conn:
            cursor = self.conn.execute(
                """INSERT INTO trained_states
                   (name, pure_state_id, file_path, sha256, experiment_id,
                    metrics_json, parent_state_id)
                   VALUES (?, ?, ?, ?, ?, ?, ?)""",
                (name, pure_state_id, str(path), sha256,
                 experiment_id, metrics_json, parent_id),
            )
            state_id = cursor.lastrowid

        state = TrainedState(
            name, pure_state_id, str(path),
            experiment_id=experiment_id,
            parent_state=parent_id,
            metrics=metrics,
        )
        state.id = state_id
        state.sha256 = sha256
        row = self.conn.execute(
            "SELECT created_at FROM trained_states WHERE id = ?", (state_id,)
        ).fetchone()
        state.created_at = row["created_at"]
        return state

    def register_export_state(
        self,
        trained_state_id: int,
        format_name: str,
        file_path: Union[str, Path],
        config: Optional[Dict[str, Any]] = None,
    ) -> ExportState:
        """
        注册一个新的 ExportState（导出部署模型）。

        Args:
            trained_state_id: 关联的 TrainedState 数据库 ID
            format_name: 导出格式名称，如 "onnx", "tensorrt", "coreml",
                         "openvino", "tflite", "torchscript"
            file_path: 导出文件路径（.onnx, .engine 等）
            config: 可选的导出配置字典
                    （如 {"opset": 17, "dynamic": True, "half": False}）

        Returns:
            已注册的 ExportState 实例（含数据库分配的 ID）

        Raises:
            FileNotFoundError: 当指定的文件不存在时抛出
            ValueError: 当引用的 TrainedState 不存在时抛出
        """
        # 验证 TrainedState 存在
        ts_row = self.conn.execute(
            "SELECT id FROM trained_states WHERE id = ?", (trained_state_id,)
        ).fetchone()
        if ts_row is None:
            raise ValueError(f"TrainedState id={trained_state_id} 不存在，请先注册")

        path = Path(file_path).resolve()
        if not path.exists():
            raise FileNotFoundError(f"文件不存在: {path}")

        sha256 = _compute_sha256(path)
        config_json = json.dumps(config or {}, ensure_ascii=False)

        with self.conn:
            cursor = self.conn.execute(
                """INSERT INTO export_states
                   (trained_state_id, format, file_path, sha256, config_json)
                   VALUES (?, ?, ?, ?, ?)""",
                (trained_state_id, format_name, str(path), sha256, config_json),
            )
            state_id = cursor.lastrowid

        state = ExportState(trained_state_id, format_name, str(path), config)
        state.id = state_id
        state.sha256 = sha256
        row = self.conn.execute(
            "SELECT created_at FROM export_states WHERE id = ?", (state_id,)
        ).fetchone()
        state.created_at = row["created_at"]
        return state

    # ------------------------------------------------------------------
    # 查询方法
    # ------------------------------------------------------------------

    def get_pure_state(self, state_id: int) -> PureState:
        """
        根据数据库 ID 获取 PureState。

        Args:
            state_id: PureState 的数据库 ID

        Returns:
            PureState 实例

        Raises:
            KeyError: 当指定 ID 的 PureState 不存在时抛出
        """
        row = self.conn.execute(
            "SELECT * FROM pure_states WHERE id = ?", (state_id,)
        ).fetchone()
        if row is None:
            raise KeyError(f"PureState id={state_id} 不存在")
        return PureState.from_db_row(row)

    def get_trained_state(self, state_id: int) -> TrainedState:
        """
        根据数据库 ID 获取 TrainedState。

        Args:
            state_id: TrainedState 的数据库 ID

        Returns:
            TrainedState 实例

        Raises:
            KeyError: 当指定 ID 的 TrainedState 不存在时抛出
        """
        row = self.conn.execute(
            "SELECT * FROM trained_states WHERE id = ?", (state_id,)
        ).fetchone()
        if row is None:
            raise KeyError(f"TrainedState id={state_id} 不存在")
        return TrainedState.from_db_row(row)

    def get_export_state(self, state_id: int) -> ExportState:
        """
        根据数据库 ID 获取 ExportState。

        Args:
            state_id: ExportState 的数据库 ID

        Returns:
            ExportState 实例

        Raises:
            KeyError: 当指定 ID 的 ExportState 不存在时抛出
        """
        row = self.conn.execute(
            "SELECT * FROM export_states WHERE id = ?", (state_id,)
        ).fetchone()
        if row is None:
            raise KeyError(f"ExportState id={state_id} 不存在")
        return ExportState.from_db_row(row)

    def list_pure_states(self) -> List[PureState]:
        """
        列出所有已注册的 PureState。

        Returns:
            按创建时间降序排列的 PureState 列表（最新的在前）
        """
        rows = self.conn.execute(
            "SELECT * FROM pure_states ORDER BY created_at DESC"
        ).fetchall()
        return [PureState.from_db_row(r) for r in rows]

    def list_trained_states(
        self, pure_state_id: Optional[int] = None
    ) -> List[TrainedState]:
        """
        列出所有已注册的 TrainedState，可按 PureState 过滤。

        Args:
            pure_state_id: 可选，若提供则仅返回关联此 PureState 的 TrainedState

        Returns:
            按创建时间降序排列的 TrainedState 列表
        """
        if pure_state_id is not None:
            rows = self.conn.execute(
                "SELECT * FROM trained_states WHERE pure_state_id = ? "
                "ORDER BY created_at DESC",
                (pure_state_id,),
            ).fetchall()
        else:
            rows = self.conn.execute(
                "SELECT * FROM trained_states ORDER BY created_at DESC"
            ).fetchall()
        return [TrainedState.from_db_row(r) for r in rows]

    def list_export_states(
        self, trained_state_id: Optional[int] = None
    ) -> List[ExportState]:
        """
        列出所有已注册的 ExportState，可按 TrainedState 过滤。

        Args:
            trained_state_id: 可选，若提供则仅返回关联此 TrainedState 的 ExportState

        Returns:
            按创建时间降序排列的 ExportState 列表
        """
        if trained_state_id is not None:
            rows = self.conn.execute(
                "SELECT * FROM export_states WHERE trained_state_id = ? "
                "ORDER BY created_at DESC",
                (trained_state_id,),
            ).fetchall()
        else:
            rows = self.conn.execute(
                "SELECT * FROM export_states ORDER BY created_at DESC"
            ).fetchall()
        return [ExportState.from_db_row(r) for r in rows]

    # ------------------------------------------------------------------
    # 血统追溯
    # ------------------------------------------------------------------

    def get_lineage(self, state_id: int) -> List[Dict[str, Any]]:
        """
        获取指定 TrainedState 的完整血统链。

        从 PureState 开始，沿 parent_state_id 链向上追溯所有训练步骤，
        返回按时间顺序排列的完整祖先链。这是血统追溯的核心方法。

        Args:
            state_id: TrainedState 的数据库 ID

        Returns:
            按时间顺序排列的血统节点列表（PureState 在最前，
            当前 TrainedState 在最后）。每个节点是包含 _type 和
            所有字段的字典。

        Raises:
            KeyError: 当指定 ID 的 TrainedState 不存在时抛出
        """
        try:
            ts = self.get_trained_state(state_id)
        except KeyError:
            raise KeyError(f"TrainedState id={state_id} 不存在，无法追溯血统")
        return ts.get_lineage(self)

    def get_lineage_tree(self, state_id: int) -> Dict[str, Any]:
        """
        构建以指定 TrainedState 为中心的血统树结构。

        包含 PureState 根节点、从 PureState 到目标节点的完整祖先链、
        以及沿途每个 TrainedState 的子节点（兄弟分支），
        便于可视化整个实验演进过程。

        Args:
            state_id: 目标 TrainedState 的数据库 ID

        Returns:
            嵌套的字典树结构:
            {
                "root": { ... pure_state 的 to_dict() ... },
                "target": { ... 目标 trained_state 的 to_dict() ... },
                "ancestor_chain": [
                    { ... 从 root 到 target 的每个训练态节点 ... }
                ],
                "branches": {
                    <node_id>: [
                        { ... 该节点的兄弟分支（不含自身） ... }
                    ]
                }
            }
        """
        lineage = self.get_lineage(state_id)

        # 分离 PureState 根节点和 TrainedState 链
        root = None
        ancestors: List[Dict[str, Any]] = []
        for node in lineage:
            if node["_type"] == "pure_state":
                root = node
            else:
                ancestors.append(node)

        target = ancestors[-1] if ancestors else None

        # 查找沿途每个 TrainedState 的所有子节点（分支）
        branches: Dict[int, List[Dict[str, Any]]] = {}
        for ancestor in ancestors:
            anc_id = ancestor["id"]
            children = self.conn.execute(
                "SELECT * FROM trained_states WHERE parent_state_id = ? AND id != ?",
                (anc_id, state_id),
            ).fetchall()
            if children:
                branches[anc_id] = [
                    TrainedState.from_db_row(c).to_dict() for c in children
                ]

        return {
            "root": root,
            "target": target,
            "ancestor_chain": ancestors,
            "branches": branches,
        }

    # ------------------------------------------------------------------
    # 状态比较
    # ------------------------------------------------------------------

    def compare_states(self, id1: int, id2: int) -> Dict[str, Any]:
        """
        比较两个 TrainedState 之间的差异。

        对比内容包括: 文件哈希、指标差异、PureState 根是否相同、
        血统距离（最近公共祖先的步数）。

        Args:
            id1: 第一个 TrainedState 的数据库 ID
            id2: 第二个 TrainedState 的数据库 ID

        Returns:
            差异报告字典:
            {
                "state1": TrainedState.to_dict(),
                "state2": TrainedState.to_dict(),
                "hash_match": bool,              # 文件哈希是否相同
                "same_pure_root": bool,           # 是否派生自同一 PureState
                "metrics_diff": {                 # 共同指标的差值
                    指标名: (值1, 值2, 差值)
                },
                "metrics_only_in_1": { ... },     # 仅在状态1中的指标
                "metrics_only_in_2": { ... },     # 仅在状态2中的指标
                "shared_metrics": { ... },        # 共享指标的值对
                "lineage_distance": int | None,   # 血统树上的距离
            }
        """
        state1 = self.get_trained_state(id1)
        state2 = self.get_trained_state(id2)

        # 文件哈希比较
        hash_match = state1.sha256 == state2.sha256

        # 同一 PureState 根？
        same_pure_root = state1.pure_state_id == state2.pure_state_id

        # 指标比较
        all_keys = set(state1.metrics.keys()) | set(state2.metrics.keys())
        shared_keys = set(state1.metrics.keys()) & set(state2.metrics.keys())
        only_in_1 = set(state1.metrics.keys()) - set(state2.metrics.keys())
        only_in_2 = set(state2.metrics.keys()) - set(state1.metrics.keys())

        metrics_diff: Dict[str, Tuple[Any, Any, Optional[float]]] = {}
        for key in sorted(shared_keys):
            v1 = state1.metrics[key]
            v2 = state2.metrics[key]
            try:
                delta = v2 - v1
            except TypeError:
                delta = None  # 非数值指标无法计算差值
            metrics_diff[key] = (v1, v2, delta)

        # 血统距离
        lineage_distance = self._compute_lineage_distance(id1, id2)

        return {
            "state1": state1.to_dict(),
            "state2": state2.to_dict(),
            "hash_match": hash_match,
            "same_pure_root": same_pure_root,
            "metrics_diff": metrics_diff,
            "metrics_only_in_1": {
                k: state1.metrics[k] for k in sorted(only_in_1)
            },
            "metrics_only_in_2": {
                k: state2.metrics[k] for k in sorted(only_in_2)
            },
            "shared_metrics": {
                k: (state1.metrics[k], state2.metrics[k])
                for k in sorted(shared_keys)
            },
            "lineage_distance": lineage_distance,
        }

    def _compute_lineage_distance(self, id1: int, id2: int) -> Optional[int]:
        """
        计算两个 TrainedState 在血统树中的最近公共祖先距离。

        通过收集 id1 的所有祖先，然后遍历 id2 的祖先找到第一个
        交集（即最近公共祖先），返回从该祖先到两个节点的步数之和。
        如果两个节点不在同一棵血统树上（没有共同祖先），返回 None。

        Args:
            id1: 第一个 TrainedState ID
            id2: 第二个 TrainedState ID

        Returns:
            血统距离（步数），如果不在同棵树则返回 None
        """
        # 收集 id1 的所有祖先（包括自身），记录深度
        ancestors1: Dict[int, int] = {}
        current: Optional[int] = id1
        depth = 0
        while current is not None:
            ancestors1[current] = depth
            row = self.conn.execute(
                "SELECT parent_state_id FROM trained_states WHERE id = ?",
                (current,),
            ).fetchone()
            current = row["parent_state_id"] if row else None
            depth += 1

        # 遍历 id2 的祖先，找到第一个在 ancestors1 中的
        current = id2
        depth = 0
        while current is not None:
            if current in ancestors1:
                return ancestors1[current] + depth
            row = self.conn.execute(
                "SELECT parent_state_id FROM trained_states WHERE id = ?",
                (current,),
            ).fetchone()
            current = row["parent_state_id"] if row else None
            depth += 1

        return None  # 不在同一棵树上

    # ------------------------------------------------------------------
    # 分支管理
    # ------------------------------------------------------------------

    def create_branch(
        self,
        name: str,
        from_state_id: int,
        description: Optional[str] = None,
    ) -> int:
        """
        从指定 TrainedState 创建一个新的命名分支。

        类似于 Git 的 `git branch <name> <commit>`，
        创建一个指向当前状态的命名引用，后续训练可通过
        parent_state_id 在此分支上继续派生。

        Args:
            name: 分支名称（全局唯一，如 "baseline", "aug-v2"）
            from_state_id: 作为分支根节点的 TrainedState ID
            description: 可选的描述文本，说明此分支的目的

        Returns:
            新创建分支的数据库 ID

        Raises:
            ValueError: 分支名已存在，或引用的 TrainedState 不存在时抛出
        """
        # 检查分支名唯一性
        existing = self.conn.execute(
            "SELECT id FROM state_branches WHERE name = ?", (name,)
        ).fetchone()
        if existing:
            raise ValueError(f"分支 '{name}' 已存在，请使用不同的名称")

        # 验证状态存在
        try:
            self.get_trained_state(from_state_id)
        except KeyError:
            raise ValueError(f"TrainedState id={from_state_id} 不存在，无法创建分支")

        with self.conn:
            cursor = self.conn.execute(
                """INSERT INTO state_branches (name, root_state_id, description)
                   VALUES (?, ?, ?)""",
                (name, from_state_id, description),
            )
            branch_id = cursor.lastrowid
        return branch_id

    def list_branches(self) -> List[Dict[str, Any]]:
        """
        列出所有已注册的分支。

        Returns:
            按创建时间降序排列的分支字典列表，
            每个字典包含 id, name, root_state_id, description, created_at
        """
        rows = self.conn.execute(
            "SELECT * FROM state_branches ORDER BY created_at DESC"
        ).fetchall()
        return [StateBranch.from_db_row(r).to_dict() for r in rows]

    def get_branch(self, name: str) -> Optional[StateBranch]:
        """
        根据名称查找分支。

        Args:
            name: 分支名称

        Returns:
            StateBranch 实例，如果不存在则返回 None
        """
        row = self.conn.execute(
            "SELECT * FROM state_branches WHERE name = ?", (name,)
        ).fetchone()
        if row is None:
            return None
        return StateBranch.from_db_row(row)

    def delete_branch(self, name: str) -> bool:
        """
        删除指定名称的分支。

        注意: 仅删除分支记录（命名引用），不会删除关联的模型状态。
        类似于 Git 的 `git branch -d <name>`。

        Args:
            name: 要删除的分支名称

        Returns:
            成功删除返回 True，分支不存在返回 False
        """
        with self.conn:
            cursor = self.conn.execute(
                "DELETE FROM state_branches WHERE name = ?", (name,)
            )
            return cursor.rowcount > 0

    # ------------------------------------------------------------------
    # 完整性验证
    # ------------------------------------------------------------------

    def verify_integrity(self, state_id: int) -> Dict[str, Any]:
        """
        验证指定 TrainedState 的完整性。

        检查以下项目:
            1. 权重文件是否仍然存在于注册时的路径
            2. 当前文件的 SHA256 哈希是否与注册时一致
            3. 引用的 PureState 记录是否存在
            4. 引用的父状态记录是否有效（如果有父状态）
            5. PureState 的权重文件是否完整

        Args:
            state_id: 要验证的 TrainedState 数据库 ID

        Returns:
            验证报告字典:
            {
                "state_id": int,
                "name": str,
                "valid": bool,                    # 整体是否通过验证
                "checks": {
                    "file_exists": bool,          # 权重文件存在
                    "hash_valid": bool,           # SHA256 匹配
                    "pure_state_valid": bool,     # PureState 引用有效
                    "parent_valid": bool | None,  # 父状态引用有效
                    "pure_state_file_valid": bool # PureState 文件存在
                },
                "errors": [str, ...],             # 错误信息列表
                "warnings": [str, ...],           # 警告信息列表
            }
        """
        errors: List[str] = []
        warnings: List[str] = []
        checks: Dict[str, Optional[bool]] = {}

        # 获取 TrainedState
        try:
            ts = self.get_trained_state(state_id)
        except KeyError:
            return {
                "state_id": state_id,
                "name": None,
                "valid": False,
                "checks": {},
                "errors": [f"TrainedState id={state_id} 在数据库中不存在"],
                "warnings": [],
            }

        # 检查 1: 文件是否存在
        ts_path = Path(ts.file_path)
        checks["file_exists"] = ts_path.exists()
        if not checks["file_exists"]:
            errors.append(f"训练权重文件不存在于注册路径: {ts.file_path}")

        # 检查 2: SHA256 哈希是否匹配
        if checks["file_exists"]:
            actual_hash = _compute_sha256(ts_path)
            checks["hash_valid"] = (actual_hash == ts.sha256)
            if not checks["hash_valid"]:
                errors.append(
                    f"SHA256 哈希不匹配: 注册时={ts.sha256[:16]}..., "
                    f"当前={actual_hash[:16]}..."
                )
        else:
            checks["hash_valid"] = False

        # 检查 3: PureState 引用
        try:
            ps = self.get_pure_state(ts.pure_state_id)
            checks["pure_state_valid"] = True

            # 检查 PureState 文件完整性
            ps_path = Path(ps.file_path)
            checks["pure_state_file_valid"] = ps_path.exists()
            if not checks["pure_state_file_valid"]:
                warnings.append(
                    f"关联的 PureState 文件丢失: {ps.file_path}"
                )
            else:
                actual_ps_hash = _compute_sha256(ps_path)
                if actual_ps_hash != ps.sha256:
                    warnings.append(
                        f"PureState 文件哈希不匹配: {ps.file_path} "
                        f"(可能已被修改)"
                    )
        except KeyError:
            checks["pure_state_valid"] = False
            checks["pure_state_file_valid"] = False
            errors.append(
                f"引用的 PureState id={ts.pure_state_id} 在数据库中不存在"
            )

        # 检查 4: 父状态引用
        if ts.parent_state_id is not None:
            try:
                self.get_trained_state(ts.parent_state_id)
                checks["parent_valid"] = True
            except KeyError:
                checks["parent_valid"] = False
                errors.append(
                    f"父 TrainedState id={ts.parent_state_id} 不存在（血统链断裂）"
                )
        else:
            checks["parent_valid"] = None  # 无父状态，不需要验证

        valid = len(errors) == 0
        return {
            "state_id": state_id,
            "name": ts.name,
            "valid": valid,
            "checks": checks,
            "errors": errors,
            "warnings": warnings,
        }

    def verify_all_integrity(self) -> List[Dict[str, Any]]:
        """
        验证所有已注册 TrainedState 的完整性。

        遍历所有 TrainedState 记录，对每个执行 verify_integrity 检查。

        Returns:
            每个 TrainedState 的验证报告列表，按 state_id 排序
        """
        reports: List[Dict[str, Any]] = []
        rows = self.conn.execute(
            "SELECT id FROM trained_states ORDER BY id"
        ).fetchall()
        for row in rows:
            reports.append(self.verify_integrity(row["id"]))
        return reports

    # ------------------------------------------------------------------
    # 垃圾回收
    # ------------------------------------------------------------------

    def gc_orphans(self) -> Dict[str, Any]:
        """
        清理数据库中的孤立记录。

        孤立定义:
            - ExportState 引用的 TrainedState 已被删除
            - StateBranch 的 root_state_id 指向已删除的 TrainedState
            - TrainedState 引用的 PureState 已被删除
            - PureState 未被任何 TrainedState 引用
            - TrainedState 的 parent_state_id 指向已删除的记录

        注意: 此方法仅删除数据库中的元数据记录，
        不会删除磁盘上的模型文件。被清理的记录无法恢复。

        Returns:
            垃圾回收报告:
            {
                "orphan_export_states": int,    # 清理的导出状态数
                "orphan_trained_states": int,   # 清理的训练状态数
                "orphan_pure_states": int,      # 清理的纯净态数
                "orphan_branches": int,          # 清理的分支数
                "total_removed": int,            # 总计删除数
            }
        """
        report = {
            "orphan_export_states": 0,
            "orphan_trained_states": 0,
            "orphan_pure_states": 0,
            "orphan_branches": 0,
            "total_removed": 0,
        }

        with self.conn:
            # 第1步: 清理孤立 ExportState（trained_state_id 无效）
            cursor = self.conn.execute(
                """DELETE FROM export_states
                   WHERE trained_state_id NOT IN (
                       SELECT id FROM trained_states
                   )"""
            )
            report["orphan_export_states"] = cursor.rowcount

            # 第2步: 清理 root_state_id 无效的分支
            cursor = self.conn.execute(
                """DELETE FROM state_branches
                   WHERE root_state_id NOT IN (
                       SELECT id FROM trained_states
                   )"""
            )
            report["orphan_branches"] = cursor.rowcount

            # 第3步: 清理孤立 TrainedState（pure_state_id 无效）
            cursor = self.conn.execute(
                """DELETE FROM trained_states
                   WHERE pure_state_id NOT IN (
                       SELECT id FROM pure_states
                   )"""
            )
            report["orphan_trained_states"] += cursor.rowcount

            # 第4步: 清理未被任何 TrainedState 引用的 PureState
            cursor = self.conn.execute(
                """DELETE FROM pure_states
                   WHERE id NOT IN (
                       SELECT DISTINCT pure_state_id FROM trained_states
                   )"""
            )
            report["orphan_pure_states"] = cursor.rowcount

            # 第5步: 清理 parent_state_id 指向已删除记录的 TrainedState
            cursor = self.conn.execute(
                """DELETE FROM trained_states
                   WHERE parent_state_id IS NOT NULL
                   AND parent_state_id NOT IN (
                       SELECT id FROM trained_states
                   )"""
            )
            report["orphan_trained_states"] += cursor.rowcount

        report["total_removed"] = sum(report.values())
        return report

    # ------------------------------------------------------------------
    # 血统导出
    # ------------------------------------------------------------------

    def export_lineage_json(self, state_id: int) -> Dict[str, Any]:
        """
        导出指定 TrainedState 的完整血统为可序列化的 JSON 结构。

        包含从 PureState 到目标状态的所有节点信息、
        沿途每个 TrainedState 的导出状态以及所有分支信息。
        适用于导出到外部系统或生成可视化报告。

        Args:
            state_id: TrainedState 的数据库 ID

        Returns:
            完整的血统 JSON 结构:
            {
                "export_time": str,              # ISO 8601 导出时间
                "state_id": int,                 # 目标状态 ID
                "lineage": [                     # 按时间排序的血统链
                    { "_type": "pure_state", ... },
                    { "_type": "trained_state", ... },
                    ...
                ],
                "export_states": {               # 每个训练态的导出状态
                    <trained_id>: [
                        { ... ExportState.to_dict() ... }
                    ]
                },
                "branches": [                    # 所有分支信息
                    { ... StateBranch.to_dict() ... }
                ]
            }
        """
        lineage = self.get_lineage(state_id)

        # 收集沿途每个 TrainedState 的导出状态
        export_states_map: Dict[int, List[Dict[str, Any]]] = {}
        for node in lineage:
            if node["_type"] == "trained_state":
                tid = node["id"]
                exports = self.list_export_states(trained_state_id=tid)
                if exports:
                    export_states_map[tid] = [e.to_dict() for e in exports]

        # 获取所有已注册的分支
        branch_rows = self.conn.execute(
            "SELECT * FROM state_branches ORDER BY created_at DESC"
        ).fetchall()
        all_branches = [StateBranch.from_db_row(r).to_dict() for r in branch_rows]

        return {
            "export_time": _now_iso(),
            "state_id": state_id,
            "lineage": lineage,
            "export_states": export_states_map,
            "branches": all_branches,
        }

    # ------------------------------------------------------------------
    # 统计与搜索
    # ------------------------------------------------------------------

    def get_stats(self) -> Dict[str, int]:
        """
        获取数据库的统计摘要。

        Returns:
            各类记录数量的字典:
            {
                "pure_states": int,
                "trained_states": int,
                "export_states": int,
                "branches": int,
            }
        """
        return {
            "pure_states": self.conn.execute(
                "SELECT COUNT(*) FROM pure_states"
            ).fetchone()[0],
            "trained_states": self.conn.execute(
                "SELECT COUNT(*) FROM trained_states"
            ).fetchone()[0],
            "export_states": self.conn.execute(
                "SELECT COUNT(*) FROM export_states"
            ).fetchone()[0],
            "branches": self.conn.execute(
                "SELECT COUNT(*) FROM state_branches"
            ).fetchone()[0],
        }

    def search_states(
        self,
        name_pattern: Optional[str] = None,
        model_variant: Optional[str] = None,
        sha256: Optional[str] = None,
    ) -> List[Dict[str, Any]]:
        """
        按条件搜索所有类型的状态。

        支持按名称模糊匹配（SQL LIKE 语法）、模型变体精确匹配
        和 SHA256 精确匹配。跨 PureState、TrainedState、ExportState
        三种类型进行搜索。

        Args:
            name_pattern: 名称搜索模式（SQL LIKE 语法，如 "%baseline%"）
            model_variant: 精确匹配的模型变体（如 "yolo26n"），仅适用于 PureState
            sha256: 精确匹配的 SHA256 哈希（支持前缀匹配，如 "abc123%"）

        Returns:
            匹配的状态列表，每项为包含 _type 和所有字段的字典。
            _type 取值为 "pure_state", "trained_state", "export_state"。
        """
        results: List[Dict[str, Any]] = []

        # 搜索 PureState
        query = "SELECT * FROM pure_states WHERE 1=1"
        params: List[Any] = []
        if name_pattern:
            query += " AND name LIKE ?"
            params.append(name_pattern)
        if model_variant:
            query += " AND model_variant = ?"
            params.append(model_variant)
        if sha256:
            query += " AND sha256 LIKE ?"
            params.append(sha256)

        for row in self.conn.execute(
            query + " ORDER BY created_at DESC", params
        ):
            node = PureState.from_db_row(row).to_dict()
            node["_type"] = "pure_state"
            results.append(node)

        # 搜索 TrainedState（model_variant 不适用于此表）
        query = "SELECT * FROM trained_states WHERE 1=1"
        params = []
        if name_pattern:
            query += " AND name LIKE ?"
            params.append(name_pattern)
        if sha256:
            query += " AND sha256 LIKE ?"
            params.append(sha256)

        for row in self.conn.execute(
            query + " ORDER BY created_at DESC", params
        ):
            node = TrainedState.from_db_row(row).to_dict()
            node["_type"] = "trained_state"
            results.append(node)

        # 搜索 ExportState
        query = "SELECT * FROM export_states WHERE 1=1"
        params = []
        if sha256:
            query += " AND sha256 LIKE ?"
            params.append(sha256)

        for row in self.conn.execute(
            query + " ORDER BY created_at DESC", params
        ):
            node = ExportState.from_db_row(row).to_dict()
            node["_type"] = "export_state"
            results.append(node)

        return results

    # ------------------------------------------------------------------
    # 数据库生命周期
    # ------------------------------------------------------------------

    def close(self) -> None:
        """
        关闭数据库连接。

        释放 SQLite 连接资源。关闭后不应再使用此 StateManager 实例。
        推荐使用上下文管理器（with 语句）自动管理连接生命周期。
        """
        if self.conn:
            self.conn.close()

    def __enter__(self) -> "StateManager":
        """上下文管理器入口，支持 with 语句。"""
        return self

    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        """上下文管理器出口，自动关闭数据库连接。"""
        self.close()

    def __del__(self) -> None:
        """析构时尝试关闭连接，防止资源泄漏。"""
        try:
            self.close()
        except Exception:
            pass  # 忽略析构时的任何异常


# ---------------------------------------------------------------------------
# 模块自检（直接运行此文件时执行）
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import shutil
    import tempfile

    print("=" * 64)
    print("  Pure State Manager — 模块自检")
    print("=" * 64)

    # 创建临时目录和模拟权重文件
    tmpdir = Path(tempfile.mkdtemp(prefix="purestate_test_"))

    def _make_dummy(path: Path, content: str) -> None:
        path.write_text(content)

    pure_file = tmpdir / "yolo26n.pt"
    trained_file1 = tmpdir / "exp001_best.pt"
    trained_file2 = tmpdir / "exp002_best.pt"
    trained_file3 = tmpdir / "exp003_best.pt"  # 基于 exp001 的微调
    export_onnx = tmpdir / "exp001.onnx"
    export_trt = tmpdir / "exp002.engine"

    _make_dummy(pure_file, "yolo26n_official_weights_v1.0")
    _make_dummy(trained_file1, "exp001_trained_on_coco_120epochs")
    _make_dummy(trained_file2, "exp002_trained_with_mosaic_aug")
    _make_dummy(trained_file3, "exp003_finetuned_from_exp001_30epochs")
    _make_dummy(export_onnx, "exp001_exported_onnx_opset17")
    _make_dummy(export_trt, "exp002_exported_tensorrt_fp16")

    # 使用内存数据库进行测试
    manager = StateManager(":memory:")

    try:
        # ── 1. 注册 PureState ──
        print("\n[1] 注册 PureState ...")
        pure = manager.register_pure_state(
            "YOLO26n 官方权重",
            "yolo26n",
            pure_file,
            metadata={"source": "ultralytics", "version": "26.0.0"},
        )
        print(f"    -> {pure}")

        # ── 2. 注册 TrainedState ──
        print("\n[2] 注册 TrainedState ...")
        ts1 = manager.register_trained_state(
            "实验001-COCO基线",
            pure.id,
            trained_file1,
            experiment_id=1001,
            metrics={
                "mAP50": 0.852, "mAP50-95": 0.621,
                "precision": 0.88, "recall": 0.79,
            },
        )
        print(f"    -> {ts1}")

        ts2 = manager.register_trained_state(
            "实验002-数据增强",
            pure.id,
            trained_file2,
            experiment_id=1002,
            metrics={"mAP50": 0.861, "mAP50-95": 0.633, "precision": 0.89, "recall": 0.80},
        )
        print(f"    -> {ts2}")

        # ── 3. 链式训练（基于 ts1）──
        print("\n[3] 注册派生 TrainedState（基于 ts1）...")
        ts3 = manager.register_trained_state(
            "实验003-微调",
            pure.id,
            trained_file3,
            experiment_id=1003,
            parent_id=ts1.id,
            metrics={"mAP50": 0.870, "mAP50-95": 0.645, "precision": 0.90, "recall": 0.82},
        )
        print(f"    -> {ts3}")

        # ── 4. 注册 ExportState ──
        print("\n[4] 注册 ExportState ...")
        export1 = manager.register_export_state(
            ts1.id, "onnx", export_onnx,
            config={"opset": 17, "dynamic": True, "half": False},
        )
        print(f"    -> {export1}")

        export2 = manager.register_export_state(
            ts2.id, "tensorrt", export_trt,
            config={"precision": "fp16", "workspace": 4096},
        )
        print(f"    -> {export2}")

        # ── 5. 分支管理 ──
        print("\n[5] 创建分支 ...")
        branch_id = manager.create_branch(
            "finetune-v1",
            ts3.id,
            description="从实验003分叉的微调实验分支",
        )
        print(f"    -> 分支 ID: {branch_id}")
        print(f"    -> 分支列表: {len(manager.list_branches())} 个分支")

        fetched_branch = manager.get_branch("finetune-v1")
        print(f"    -> 获取分支: {fetched_branch.name if fetched_branch else 'None'}")

        # ── 6. 列表查询 ──
        print("\n[6] 列表查询 ...")
        print(f"    PureStates:                {len(manager.list_pure_states())}")
        print(f"    TrainedStates (全部):       {len(manager.list_trained_states())}")
        print(f"    TrainedStates (pure={pure.id}): {len(manager.list_trained_states(pure.id))}")
        print(f"    ExportStates (全部):        {len(manager.list_export_states())}")
        print(f"    ExportStates (ts1={ts1.id}):  {len(manager.list_export_states(ts1.id))}")

        # ── 7. 血统追溯 ──
        print("\n[7] 血统追溯 (ts3, 链式训练) ...")
        lineage = manager.get_lineage(ts3.id)
        for i, node in enumerate(lineage):
            indent = "  " * i
            print(f"    {indent}[{i}] {node['_type']}: {node.get('name', '?')}"
                  f"  (sha256={str(node.get('sha256', ''))[:12]}...)")

        # ── 8. 血统树 ──
        print("\n[8] 血统树 (ts3) ...")
        tree = manager.get_lineage_tree(ts3.id)
        print(f"    root:   {tree['root']['name'] if tree['root'] else None}")
        print(f"    target: {tree['target']['name'] if tree['target'] else None}")
        print(f"    chain:  {len(tree['ancestor_chain'])} 个训练态节点")
        print(f"    分支节点: {list(tree['branches'].keys())}")

        # ── 9. 状态比较 ──
        print("\n[9] 比较 ts1 vs ts2 ...")
        diff = manager.compare_states(ts1.id, ts2.id)
        print(f"    hash_match:        {diff['hash_match']}")
        print(f"    same_pure_root:    {diff['same_pure_root']}")
        print(f"    lineage_distance:  {diff['lineage_distance']}")
        print(f"    metrics_diff:      {diff['metrics_diff']}")

        # ── 10. 完整性验证 ──
        print("\n[10] 完整性验证 (ts1) ...")
        report = manager.verify_integrity(ts1.id)
        print(f"    valid:   {report['valid']}")
        print(f"    checks:  {report['checks']}")
        print(f"    errors:  {report['errors']}")
        print(f"    warnings: {report['warnings']}")

        # 验证所有
        all_reports = manager.verify_all_integrity()
        passed = sum(1 for r in all_reports if r["valid"])
        print(f"    verify_all: {passed}/{len(all_reports)} 通过")

        # ── 11. 垃圾回收 ──
        print("\n[11] 垃圾回收 ...")
        gc_report = manager.gc_orphans()
        print(f"    {gc_report}")

        # ── 12. 血统 JSON 导出 ──
        print("\n[12] 导出血统 JSON (ts3) ...")
        export_json = manager.export_lineage_json(ts3.id)
        print(f"    export_time:     {export_json['export_time']}")
        print(f"    lineage nodes:   {len(export_json['lineage'])}")
        print(f"    export_states:   {list(export_json['export_states'].keys())}")
        print(f"    branches count:  {len(export_json['branches'])}")

        # ── 13. 搜索 ──
        print("\n[13] 搜索状态 ...")
        results = manager.search_states(name_pattern="%实验%")
        print(f"    匹配 '{'%实验%'}': {len(results)} 条")
        for r in results:
            print(f"      [{r['_type']}] {r.get('name', '?')}")

        # ── 14. 统计 ──
        print("\n[14] 数据库统计 ...")
        stats = manager.get_stats()
        print(f"    {stats}")

        print("\n" + "=" * 64)
        print("  所有自检通过!")
        print("=" * 64)

    finally:
        manager.close()
        shutil.rmtree(tmpdir, ignore_errors=True)
