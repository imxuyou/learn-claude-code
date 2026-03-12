#!/usr/bin/env python3
"""
s12_worktree_task_isolation.py - Worktree + Task Isolation (工作树 + 任务隔离)

=== 本课要点 ===

引入"工作树隔离"概念：每个任务在独立的目录中执行，互不干扰。

核心洞察：
    "按目录隔离，按任务 ID 协调。"

=== 为什么需要工作树隔离？ ===

问题场景：同时开发多个功能

    没有隔离：
    ┌─────────────────────────────────────────────────────────┐
    │  Alice: 修改 auth.py（添加 OAuth）                       │
    │  Bob:   修改 auth.py（修复 bug）                         │
    │                                                         │
    │  冲突！两人在同一个文件上工作                             │
    │  更糟：半成品代码互相影响                                 │
    └─────────────────────────────────────────────────────────┘

    有隔离（使用 Git Worktree）：
    ┌─────────────────────────────────────────────────────────┐
    │  主目录:           /project/                            │
    │  Alice 的工作树:   /project/.worktrees/oauth-feature/   │
    │  Bob 的工作树:     /project/.worktrees/auth-bugfix/     │
    │                                                         │
    │  各自在独立目录工作，互不干扰！                           │
    │  完成后再合并到主分支                                    │
    └─────────────────────────────────────────────────────────┘

=== 什么是 Git Worktree？ ===

Git Worktree 是 Git 的一个功能：
    - 允许同一个仓库有多个工作目录
    - 每个工作目录对应不同的分支
    - 共享同一个 .git 目录

    git worktree add -b feature/oauth .worktrees/oauth-feature HEAD
    → 创建新分支 feature/oauth
    → 在 .worktrees/oauth-feature/ 目录检出这个分支

=== 控制平面 vs 执行平面 ===

    控制平面（Control Plane）：任务管理
    ┌──────────────────────────────────────┐
    │  .tasks/task_12.json                 │
    │  {                                   │
    │    "id": 12,                         │
    │    "subject": "Implement OAuth",     │
    │    "status": "in_progress",          │
    │    "worktree": "oauth-feature"  ←────┼── 绑定到工作树
    │  }                                   │
    └──────────────────────────────────────┘
                    │
                    │ 协调
                    ▼
    执行平面（Execution Plane）：实际工作
    ┌──────────────────────────────────────┐
    │  .worktrees/index.json               │
    │  {                                   │
    │    "worktrees": [{                   │
    │      "name": "oauth-feature",        │
    │      "path": ".worktrees/oauth-...", │
    │      "branch": "wt/oauth-feature",   │
    │      "task_id": 12,             ←────┼── 绑定到任务
    │      "status": "active"              │
    │    }]                                │
    │  }                                   │
    └──────────────────────────────────────┘

=== 生命周期事件 ===

    所有操作都会记录到事件日志：

    .worktrees/events.jsonl
    {"event":"worktree.create.before","ts":...,"task":{"id":12},"worktree":{"name":"oauth"}}
    {"event":"worktree.create.after","ts":...,"task":{"id":12},"worktree":{"name":"oauth","status":"active"}}
    {"event":"worktree.remove.before","ts":...}
    {"event":"task.completed","ts":...,"task":{"id":12,"status":"completed"}}

=== 典型工作流程 ===

    1. 创建任务
       task_create("Implement OAuth")  → task #12

    2. 为任务创建工作树
       worktree_create("oauth-feature", task_id=12)
       → 创建 .worktrees/oauth-feature/ 目录
       → 自动绑定到 task #12

    3. 在工作树中执行命令
       worktree_run("oauth-feature", "npm install")
       worktree_run("oauth-feature", "npm test")

    4. 完成后选择：保留或删除
       worktree_keep("oauth-feature")   # 保留工作树
       或
       worktree_remove("oauth-feature", complete_task=True)  # 删除并完成任务
"""

# ============================================================================
# 第一部分：导入依赖
# ============================================================================

import json
import os
import re           # 正则表达式，用于验证工作树名称
import subprocess
import time
from pathlib import Path

from anthropic import Anthropic
from dotenv import load_dotenv

# ============================================================================
# 第二部分：初始化配置
# ============================================================================

load_dotenv(override=True)

if os.getenv("ANTHROPIC_BASE_URL"):
    os.environ.pop("ANTHROPIC_AUTH_TOKEN", None)

WORKDIR = Path.cwd()
client = Anthropic(base_url=os.getenv("ANTHROPIC_BASE_URL"))
MODEL = os.environ["MODEL_ID"]


def detect_repo_root(cwd: Path) -> Path | None:
    """
    检测 Git 仓库根目录。

    Git Worktree 需要在 Git 仓库中使用，
    这个函数找到仓库的根目录。

    Returns:
        仓库根目录，如果不在仓库中则返回 None
    """
    try:
        r = subprocess.run(
            ["git", "rev-parse", "--show-toplevel"],
            cwd=cwd,
            capture_output=True,
            text=True,
            timeout=10,
        )
        if r.returncode != 0:
            return None
        root = Path(r.stdout.strip())
        return root if root.exists() else None
    except Exception:
        return None


# 仓库根目录（如果不在仓库中，使用当前目录）
REPO_ROOT = detect_repo_root(WORKDIR) or WORKDIR

# 系统提示词
SYSTEM = (
    f"You are a coding agent at {WORKDIR}. "
    "Use task + worktree tools for multi-task work. "
    "For parallel or risky changes: create tasks, allocate worktree lanes, "
    "run commands in those lanes, then choose keep/remove for closeout. "
    "Use worktree_events when you need lifecycle visibility."
)


# ============================================================================
# 第三部分：EventBus 类（事件总线）
# ============================================================================

class EventBus:
    """
    事件总线：记录所有生命周期事件。

    为什么需要事件日志？
    1. 可观测性：知道发生了什么
    2. 调试：出问题时可以回溯
    3. 审计：记录所有操作

    事件格式：
    {
        "event": "worktree.create.after",
        "ts": 1234567890.123,
        "task": {"id": 12},
        "worktree": {"name": "oauth", "status": "active"}
    }
    """

    def __init__(self, event_log_path: Path):
        self.path = event_log_path
        self.path.parent.mkdir(parents=True, exist_ok=True)
        if not self.path.exists():
            self.path.write_text("")

    def emit(
        self,
        event: str,
        task: dict | None = None,
        worktree: dict | None = None,
        error: str | None = None,
    ):
        """
        发送一个事件。

        Args:
            event: 事件名称（如 "worktree.create.before"）
            task: 相关任务信息
            worktree: 相关工作树信息
            error: 错误信息（如果有）
        """
        payload = {
            "event": event,
            "ts": time.time(),
            "task": task or {},
            "worktree": worktree or {},
        }
        if error:
            payload["error"] = error

        # 追加到日志文件
        with self.path.open("a", encoding="utf-8") as f:
            f.write(json.dumps(payload) + "\n")

    def list_recent(self, limit: int = 20) -> str:
        """
        获取最近的事件。

        Args:
            limit: 返回的事件数量（最多 200）

        Returns:
            JSON 格式的事件列表
        """
        n = max(1, min(int(limit or 20), 200))
        lines = self.path.read_text(encoding="utf-8").splitlines()
        recent = lines[-n:]

        items = []
        for line in recent:
            try:
                items.append(json.loads(line))
            except Exception:
                items.append({"event": "parse_error", "raw": line})

        return json.dumps(items, indent=2)


# ============================================================================
# 第四部分：TaskManager 类（支持工作树绑定）
# ============================================================================

class TaskManager:
    """
    任务管理器：支持工作树绑定。

    与 s07 的区别：
    1. 新增 worktree 字段
    2. 新增 bind_worktree / unbind_worktree 方法
    3. 绑定时自动更新任务状态为 in_progress
    """

    def __init__(self, tasks_dir: Path):
        self.dir = tasks_dir
        self.dir.mkdir(parents=True, exist_ok=True)
        self._next_id = self._max_id() + 1

    def _max_id(self) -> int:
        ids = []
        for f in self.dir.glob("task_*.json"):
            try:
                ids.append(int(f.stem.split("_")[1]))
            except Exception:
                pass
        return max(ids) if ids else 0

    def _path(self, task_id: int) -> Path:
        return self.dir / f"task_{task_id}.json"

    def _load(self, task_id: int) -> dict:
        path = self._path(task_id)
        if not path.exists():
            raise ValueError(f"Task {task_id} not found")
        return json.loads(path.read_text())

    def _save(self, task: dict):
        self._path(task["id"]).write_text(json.dumps(task, indent=2))

    def create(self, subject: str, description: str = "") -> str:
        """创建新任务"""
        task = {
            "id": self._next_id,
            "subject": subject,
            "description": description,
            "status": "pending",
            "owner": "",
            "worktree": "",        # 新增：工作树绑定
            "blockedBy": [],
            "created_at": time.time(),
            "updated_at": time.time(),
        }
        self._save(task)
        self._next_id += 1
        return json.dumps(task, indent=2)

    def get(self, task_id: int) -> str:
        return json.dumps(self._load(task_id), indent=2)

    def exists(self, task_id: int) -> bool:
        return self._path(task_id).exists()

    def update(self, task_id: int, status: str = None, owner: str = None) -> str:
        """更新任务状态或负责人"""
        task = self._load(task_id)
        if status:
            if status not in ("pending", "in_progress", "completed"):
                raise ValueError(f"Invalid status: {status}")
            task["status"] = status
        if owner is not None:
            task["owner"] = owner
        task["updated_at"] = time.time()
        self._save(task)
        return json.dumps(task, indent=2)

    def bind_worktree(self, task_id: int, worktree: str, owner: str = "") -> str:
        """
        将任务绑定到工作树。

        绑定时：
        1. 设置 worktree 字段
        2. 如果是 pending 状态，自动变为 in_progress

        Args:
            task_id: 任务 ID
            worktree: 工作树名称
            owner: 负责人（可选）
        """
        task = self._load(task_id)
        task["worktree"] = worktree
        if owner:
            task["owner"] = owner
        # 自动更新状态
        if task["status"] == "pending":
            task["status"] = "in_progress"
        task["updated_at"] = time.time()
        self._save(task)
        return json.dumps(task, indent=2)

    def unbind_worktree(self, task_id: int) -> str:
        """解除任务与工作树的绑定"""
        task = self._load(task_id)
        task["worktree"] = ""
        task["updated_at"] = time.time()
        self._save(task)
        return json.dumps(task, indent=2)

    def list_all(self) -> str:
        """列出所有任务（包含工作树信息）"""
        tasks = []
        for f in sorted(self.dir.glob("task_*.json")):
            tasks.append(json.loads(f.read_text()))

        if not tasks:
            return "No tasks."

        lines = []
        for t in tasks:
            marker = {
                "pending": "[ ]",
                "in_progress": "[>]",
                "completed": "[x]",
            }.get(t["status"], "[?]")
            owner = f" owner={t['owner']}" if t.get("owner") else ""
            wt = f" wt={t['worktree']}" if t.get("worktree") else ""
            lines.append(f"{marker} #{t['id']}: {t['subject']}{owner}{wt}")

        return "\n".join(lines)


# ============================================================================
# 第五部分：创建全局实例
# ============================================================================

TASKS = TaskManager(REPO_ROOT / ".tasks")
EVENTS = EventBus(REPO_ROOT / ".worktrees" / "events.jsonl")


# ============================================================================
# 第六部分：WorktreeManager 类（本课核心）
# ============================================================================

class WorktreeManager:
    """
    工作树管理器：创建、列出、执行、删除 Git 工作树。

    工作树目录结构：
    .worktrees/
    ├── index.json              # 工作树索引
    ├── events.jsonl            # 生命周期事件
    ├── oauth-feature/          # 工作树目录 1
    │   ├── src/
    │   └── ...
    └── auth-bugfix/            # 工作树目录 2
        ├── src/
        └── ...

    索引文件格式（index.json）：
    {
        "worktrees": [
            {
                "name": "oauth-feature",
                "path": "/project/.worktrees/oauth-feature",
                "branch": "wt/oauth-feature",
                "task_id": 12,
                "status": "active",
                "created_at": 1234567890.123
            }
        ]
    }
    """

    def __init__(self, repo_root: Path, tasks: TaskManager, events: EventBus):
        self.repo_root = repo_root
        self.tasks = tasks
        self.events = events

        # 工作树目录
        self.dir = repo_root / ".worktrees"
        self.dir.mkdir(parents=True, exist_ok=True)

        # 索引文件
        self.index_path = self.dir / "index.json"
        if not self.index_path.exists():
            self.index_path.write_text(json.dumps({"worktrees": []}, indent=2))

        # 检查是否在 Git 仓库中
        self.git_available = self._is_git_repo()

    def _is_git_repo(self) -> bool:
        """检查是否在 Git 仓库中"""
        try:
            r = subprocess.run(
                ["git", "rev-parse", "--is-inside-work-tree"],
                cwd=self.repo_root,
                capture_output=True,
                text=True,
                timeout=10,
            )
            return r.returncode == 0
        except Exception:
            return False

    def _run_git(self, args: list[str]) -> str:
        """执行 Git 命令"""
        if not self.git_available:
            raise RuntimeError("Not in a git repository. worktree tools require git.")
        r = subprocess.run(
            ["git", *args],
            cwd=self.repo_root,
            capture_output=True,
            text=True,
            timeout=120,
        )
        if r.returncode != 0:
            msg = (r.stdout + r.stderr).strip()
            raise RuntimeError(msg or f"git {' '.join(args)} failed")
        return (r.stdout + r.stderr).strip() or "(no output)"

    def _load_index(self) -> dict:
        return json.loads(self.index_path.read_text())

    def _save_index(self, data: dict):
        self.index_path.write_text(json.dumps(data, indent=2))

    def _find(self, name: str) -> dict | None:
        """根据名称查找工作树"""
        idx = self._load_index()
        for wt in idx.get("worktrees", []):
            if wt.get("name") == name:
                return wt
        return None

    def _validate_name(self, name: str):
        """验证工作树名称（只允许字母、数字、点、下划线、横线）"""
        if not re.fullmatch(r"[A-Za-z0-9._-]{1,40}", name or ""):
            raise ValueError(
                "Invalid worktree name. Use 1-40 chars: letters, numbers, ., _, -"
            )

    # -------------------------------------------------------------------------
    # 核心操作：创建工作树
    # -------------------------------------------------------------------------

    def create(self, name: str, task_id: int = None, base_ref: str = "HEAD") -> str:
        """
        创建新的 Git 工作树。

        工作流程：
        1. 验证名称
        2. 检查是否已存在
        3. 执行 git worktree add
        4. 更新索引
        5. 绑定到任务（如果指定）
        6. 记录事件

        Args:
            name: 工作树名称（如 "oauth-feature"）
            task_id: 要绑定的任务 ID（可选）
            base_ref: 基于哪个提交创建（默认 HEAD）

        Returns:
            创建的工作树信息 JSON
        """
        self._validate_name(name)

        if self._find(name):
            raise ValueError(f"Worktree '{name}' already exists in index")

        if task_id is not None and not self.tasks.exists(task_id):
            raise ValueError(f"Task {task_id} not found")

        path = self.dir / name
        branch = f"wt/{name}"

        # 记录创建前事件
        self.events.emit(
            "worktree.create.before",
            task={"id": task_id} if task_id is not None else {},
            worktree={"name": name, "base_ref": base_ref},
        )

        try:
            # 执行 git worktree add -b <branch> <path> <base_ref>
            self._run_git(["worktree", "add", "-b", branch, str(path), base_ref])

            # 创建索引条目
            entry = {
                "name": name,
                "path": str(path),
                "branch": branch,
                "task_id": task_id,
                "status": "active",
                "created_at": time.time(),
            }

            # 更新索引
            idx = self._load_index()
            idx["worktrees"].append(entry)
            self._save_index(idx)

            # 绑定到任务
            if task_id is not None:
                self.tasks.bind_worktree(task_id, name)

            # 记录创建后事件
            self.events.emit(
                "worktree.create.after",
                task={"id": task_id} if task_id is not None else {},
                worktree={
                    "name": name,
                    "path": str(path),
                    "branch": branch,
                    "status": "active",
                },
            )

            return json.dumps(entry, indent=2)

        except Exception as e:
            # 记录失败事件
            self.events.emit(
                "worktree.create.failed",
                task={"id": task_id} if task_id is not None else {},
                worktree={"name": name, "base_ref": base_ref},
                error=str(e),
            )
            raise

    def list_all(self) -> str:
        """列出所有工作树"""
        idx = self._load_index()
        wts = idx.get("worktrees", [])
        if not wts:
            return "No worktrees in index."

        lines = []
        for wt in wts:
            suffix = f" task={wt['task_id']}" if wt.get("task_id") else ""
            lines.append(
                f"[{wt.get('status', 'unknown')}] {wt['name']} -> "
                f"{wt['path']} ({wt.get('branch', '-')}){suffix}"
            )
        return "\n".join(lines)

    def status(self, name: str) -> str:
        """获取工作树的 Git 状态"""
        wt = self._find(name)
        if not wt:
            return f"Error: Unknown worktree '{name}'"

        path = Path(wt["path"])
        if not path.exists():
            return f"Error: Worktree path missing: {path}"

        r = subprocess.run(
            ["git", "status", "--short", "--branch"],
            cwd=path,
            capture_output=True,
            text=True,
            timeout=60,
        )
        text = (r.stdout + r.stderr).strip()
        return text or "Clean worktree"

    def run(self, name: str, command: str) -> str:
        """
        在指定工作树中执行命令。

        这是工作树的核心用法：在隔离的目录中执行命令。

        Args:
            name: 工作树名称
            command: 要执行的命令

        Returns:
            命令输出
        """
        dangerous = ["rm -rf /", "sudo", "shutdown", "reboot", "> /dev/"]
        if any(d in command for d in dangerous):
            return "Error: Dangerous command blocked"

        wt = self._find(name)
        if not wt:
            return f"Error: Unknown worktree '{name}'"

        path = Path(wt["path"])
        if not path.exists():
            return f"Error: Worktree path missing: {path}"

        try:
            r = subprocess.run(
                command,
                shell=True,
                cwd=path,  # 关键：在工作树目录中执行
                capture_output=True,
                text=True,
                timeout=300,
            )
            out = (r.stdout + r.stderr).strip()
            return out[:50000] if out else "(no output)"
        except subprocess.TimeoutExpired:
            return "Error: Timeout (300s)"

    # -------------------------------------------------------------------------
    # 关闭操作：删除或保留
    # -------------------------------------------------------------------------

    def remove(self, name: str, force: bool = False, complete_task: bool = False) -> str:
        """
        删除工作树。

        Args:
            name: 工作树名称
            force: 是否强制删除（忽略未提交的更改）
            complete_task: 是否同时完成关联的任务

        Returns:
            删除结果消息
        """
        wt = self._find(name)
        if not wt:
            return f"Error: Unknown worktree '{name}'"

        # 记录删除前事件
        self.events.emit(
            "worktree.remove.before",
            task={"id": wt.get("task_id")} if wt.get("task_id") is not None else {},
            worktree={"name": name, "path": wt.get("path")},
        )

        try:
            # 执行 git worktree remove
            args = ["worktree", "remove"]
            if force:
                args.append("--force")
            args.append(wt["path"])
            self._run_git(args)

            # 如果需要，完成关联的任务
            if complete_task and wt.get("task_id") is not None:
                task_id = wt["task_id"]
                before = json.loads(self.tasks.get(task_id))
                self.tasks.update(task_id, status="completed")
                self.tasks.unbind_worktree(task_id)

                # 记录任务完成事件
                self.events.emit(
                    "task.completed",
                    task={
                        "id": task_id,
                        "subject": before.get("subject", ""),
                        "status": "completed",
                    },
                    worktree={"name": name},
                )

            # 更新索引状态
            idx = self._load_index()
            for item in idx.get("worktrees", []):
                if item.get("name") == name:
                    item["status"] = "removed"
                    item["removed_at"] = time.time()
            self._save_index(idx)

            # 记录删除后事件
            self.events.emit(
                "worktree.remove.after",
                task={"id": wt.get("task_id")} if wt.get("task_id") is not None else {},
                worktree={"name": name, "path": wt.get("path"), "status": "removed"},
            )

            return f"Removed worktree '{name}'"

        except Exception as e:
            # 记录失败事件
            self.events.emit(
                "worktree.remove.failed",
                task={"id": wt.get("task_id")} if wt.get("task_id") is not None else {},
                worktree={"name": name, "path": wt.get("path")},
                error=str(e),
            )
            raise

    def keep(self, name: str) -> str:
        """
        标记工作树为"保留"状态。

        与 remove 的区别：
        - keep: 不删除目录，只更新状态
        - remove: 删除目录

        用途：工作完成但想保留代码以便后续参考

        Args:
            name: 工作树名称

        Returns:
            更新后的工作树信息
        """
        wt = self._find(name)
        if not wt:
            return f"Error: Unknown worktree '{name}'"

        idx = self._load_index()
        kept = None
        for item in idx.get("worktrees", []):
            if item.get("name") == name:
                item["status"] = "kept"
                item["kept_at"] = time.time()
                kept = item
        self._save_index(idx)

        # 记录保留事件
        self.events.emit(
            "worktree.keep",
            task={"id": wt.get("task_id")} if wt.get("task_id") is not None else {},
            worktree={
                "name": name,
                "path": wt.get("path"),
                "status": "kept",
            },
        )

        return json.dumps(kept, indent=2) if kept else f"Error: Unknown worktree '{name}'"


# ============================================================================
# 第七部分：创建全局工作树管理器
# ============================================================================

WORKTREES = WorktreeManager(REPO_ROOT, TASKS, EVENTS)


# ============================================================================
# 第八部分：基础工具实现函数
# ============================================================================

def safe_path(p: str) -> Path:
    path = (WORKDIR / p).resolve()
    if not path.is_relative_to(WORKDIR):
        raise ValueError(f"Path escapes workspace: {p}")
    return path


def run_bash(command: str) -> str:
    dangerous = ["rm -rf /", "sudo", "shutdown", "reboot", "> /dev/"]
    if any(d in command for d in dangerous):
        return "Error: Dangerous command blocked"
    try:
        r = subprocess.run(
            command,
            shell=True,
            cwd=WORKDIR,
            capture_output=True,
            text=True,
            timeout=120,
        )
        out = (r.stdout + r.stderr).strip()
        return out[:50000] if out else "(no output)"
    except subprocess.TimeoutExpired:
        return "Error: Timeout (120s)"


def run_read(path: str, limit: int = None) -> str:
    try:
        lines = safe_path(path).read_text().splitlines()
        if limit and limit < len(lines):
            lines = lines[:limit] + [f"... ({len(lines) - limit} more)"]
        return "\n".join(lines)[:50000]
    except Exception as e:
        return f"Error: {e}"


def run_write(path: str, content: str) -> str:
    try:
        fp = safe_path(path)
        fp.parent.mkdir(parents=True, exist_ok=True)
        fp.write_text(content)
        return f"Wrote {len(content)} bytes"
    except Exception as e:
        return f"Error: {e}"


def run_edit(path: str, old_text: str, new_text: str) -> str:
    try:
        fp = safe_path(path)
        c = fp.read_text()
        if old_text not in c:
            return f"Error: Text not found in {path}"
        fp.write_text(c.replace(old_text, new_text, 1))
        return f"Edited {path}"
    except Exception as e:
        return f"Error: {e}"


# ============================================================================
# 第九部分：工具分发映射
# ============================================================================

TOOL_HANDLERS = {
    # 基础工具
    "bash": lambda **kw: run_bash(kw["command"]),
    "read_file": lambda **kw: run_read(kw["path"], kw.get("limit")),
    "write_file": lambda **kw: run_write(kw["path"], kw["content"]),
    "edit_file": lambda **kw: run_edit(kw["path"], kw["old_text"], kw["new_text"]),

    # 任务工具
    "task_create": lambda **kw: TASKS.create(kw["subject"], kw.get("description", "")),
    "task_list": lambda **kw: TASKS.list_all(),
    "task_get": lambda **kw: TASKS.get(kw["task_id"]),
    "task_update": lambda **kw: TASKS.update(kw["task_id"], kw.get("status"), kw.get("owner")),
    "task_bind_worktree": lambda **kw: TASKS.bind_worktree(kw["task_id"], kw["worktree"], kw.get("owner", "")),

    # 工作树工具（新增）
    "worktree_create": lambda **kw: WORKTREES.create(kw["name"], kw.get("task_id"), kw.get("base_ref", "HEAD")),
    "worktree_list": lambda **kw: WORKTREES.list_all(),
    "worktree_status": lambda **kw: WORKTREES.status(kw["name"]),
    "worktree_run": lambda **kw: WORKTREES.run(kw["name"], kw["command"]),
    "worktree_keep": lambda **kw: WORKTREES.keep(kw["name"]),
    "worktree_remove": lambda **kw: WORKTREES.remove(kw["name"], kw.get("force", False), kw.get("complete_task", False)),
    "worktree_events": lambda **kw: EVENTS.list_recent(kw.get("limit", 20)),
}


# ============================================================================
# 第十部分：工具定义列表
# ============================================================================

TOOLS = [
    # 基础工具
    {
        "name": "bash",
        "description": "Run a shell command in the current workspace (blocking).",
        "input_schema": {
            "type": "object",
            "properties": {"command": {"type": "string"}},
            "required": ["command"],
        },
    },
    {
        "name": "read_file",
        "description": "Read file contents.",
        "input_schema": {
            "type": "object",
            "properties": {
                "path": {"type": "string"},
                "limit": {"type": "integer"},
            },
            "required": ["path"],
        },
    },
    {
        "name": "write_file",
        "description": "Write content to file.",
        "input_schema": {
            "type": "object",
            "properties": {
                "path": {"type": "string"},
                "content": {"type": "string"},
            },
            "required": ["path", "content"],
        },
    },
    {
        "name": "edit_file",
        "description": "Replace exact text in file.",
        "input_schema": {
            "type": "object",
            "properties": {
                "path": {"type": "string"},
                "old_text": {"type": "string"},
                "new_text": {"type": "string"},
            },
            "required": ["path", "old_text", "new_text"],
        },
    },

    # 任务工具
    {
        "name": "task_create",
        "description": "Create a new task on the shared task board.",
        "input_schema": {
            "type": "object",
            "properties": {
                "subject": {"type": "string"},
                "description": {"type": "string"},
            },
            "required": ["subject"],
        },
    },
    {
        "name": "task_list",
        "description": "List all tasks with status, owner, and worktree binding.",
        "input_schema": {"type": "object", "properties": {}},
    },
    {
        "name": "task_get",
        "description": "Get task details by ID.",
        "input_schema": {
            "type": "object",
            "properties": {"task_id": {"type": "integer"}},
            "required": ["task_id"],
        },
    },
    {
        "name": "task_update",
        "description": "Update task status or owner.",
        "input_schema": {
            "type": "object",
            "properties": {
                "task_id": {"type": "integer"},
                "status": {
                    "type": "string",
                    "enum": ["pending", "in_progress", "completed"],
                },
                "owner": {"type": "string"},
            },
            "required": ["task_id"],
        },
    },
    {
        "name": "task_bind_worktree",
        "description": "Bind a task to a worktree name.",
        "input_schema": {
            "type": "object",
            "properties": {
                "task_id": {"type": "integer"},
                "worktree": {"type": "string"},
                "owner": {"type": "string"},
            },
            "required": ["task_id", "worktree"],
        },
    },

    # 工作树工具（新增）
    {
        "name": "worktree_create",
        "description": "Create a git worktree and optionally bind it to a task.",
        "input_schema": {
            "type": "object",
            "properties": {
                "name": {"type": "string"},
                "task_id": {"type": "integer"},
                "base_ref": {"type": "string"},
            },
            "required": ["name"],
        },
    },
    {
        "name": "worktree_list",
        "description": "List worktrees tracked in .worktrees/index.json.",
        "input_schema": {"type": "object", "properties": {}},
    },
    {
        "name": "worktree_status",
        "description": "Show git status for one worktree.",
        "input_schema": {
            "type": "object",
            "properties": {"name": {"type": "string"}},
            "required": ["name"],
        },
    },
    {
        "name": "worktree_run",
        "description": "Run a shell command in a named worktree directory.",
        "input_schema": {
            "type": "object",
            "properties": {
                "name": {"type": "string"},
                "command": {"type": "string"},
            },
            "required": ["name", "command"],
        },
    },
    {
        "name": "worktree_remove",
        "description": "Remove a worktree and optionally mark its bound task completed.",
        "input_schema": {
            "type": "object",
            "properties": {
                "name": {"type": "string"},
                "force": {"type": "boolean"},
                "complete_task": {"type": "boolean"},
            },
            "required": ["name"],
        },
    },
    {
        "name": "worktree_keep",
        "description": "Mark a worktree as kept in lifecycle state without removing it.",
        "input_schema": {
            "type": "object",
            "properties": {"name": {"type": "string"}},
            "required": ["name"],
        },
    },
    {
        "name": "worktree_events",
        "description": "List recent worktree/task lifecycle events from .worktrees/events.jsonl.",
        "input_schema": {
            "type": "object",
            "properties": {"limit": {"type": "integer"}},
        },
    },
]


# ============================================================================
# 第十一部分：序列化响应内容
# ============================================================================

def serialize_content(content):
    """将 API 响应转换为可序列化的字典格式"""
    result = []
    for block in content:
        if block.type == "text":
            result.append({"type": "text", "text": block.text})
        elif block.type == "tool_use":
            result.append({
                "type": "tool_use",
                "id": block.id,
                "name": block.name,
                "input": block.input
            })
    return result


# ============================================================================
# 第十二部分：Agent 循环
# ============================================================================

def agent_loop(messages: list):
    """Agent 核心循环"""
    while True:
        response = client.messages.create(
            model=MODEL,
            system=SYSTEM,
            messages=messages,
            tools=TOOLS,
            max_tokens=8000,
        )

        messages.append({"role": "assistant", "content": serialize_content(response.content)})

        if response.stop_reason != "tool_use":
            return

        results = []
        for block in response.content:
            if block.type == "tool_use":
                handler = TOOL_HANDLERS.get(block.name)
                try:
                    output = handler(**block.input) if handler else f"Unknown tool: {block.name}"
                except Exception as e:
                    output = f"Error: {e}"

                print(f"> {block.name}: {str(output)[:200]}")
                results.append({
                    "type": "tool_result",
                    "tool_use_id": block.id,
                    "content": str(output),
                })

        messages.append({"role": "user", "content": results})


# ============================================================================
# 第十三部分：主程序入口
# ============================================================================

if __name__ == "__main__":
    """
    交互式命令行界面。

    使用示例：
        s12 >> 创建两个任务，分别在独立的工作树中开发

    典型工作流程：
        1. task_create("实现 OAuth 登录")
        2. worktree_create("oauth-feature", task_id=1)
        3. worktree_run("oauth-feature", "npm install oauth-lib")
        4. worktree_run("oauth-feature", "npm test")
        5. worktree_status("oauth-feature")
        6. worktree_remove("oauth-feature", complete_task=True)

    注意：需要在 Git 仓库中运行才能使用 worktree_* 工具

    快捷命令：
        /tasks - 查看任务列表
        /wts   - 查看工作树列表
        /events - 查看最近事件
    """
    print("=" * 60)
    print("s12_worktree_task_isolation.py - 工作树 + 任务隔离示例")
    print(f"仓库根目录: {REPO_ROOT}")

    if not WORKTREES.git_available:
        print("警告: 不在 Git 仓库中。worktree_* 工具将返回错误。")
    else:
        print("Git 仓库已检测到。worktree_* 工具可用。")

    print("新增工具: worktree_create, worktree_run, worktree_remove, ...")
    print("输入 'q' 或 'exit' 退出")
    print("=" * 60)

    history = []
    while True:
        try:
            query = input("\033[36ms12 >> \033[0m")
        except (EOFError, KeyboardInterrupt):
            break

        if query.strip().lower() in ("q", "exit", ""):
            break

        history.append({"role": "user", "content": query})
        agent_loop(history)

        # 打印模型的文字回复
        response_content = history[-1]["content"]
        if isinstance(response_content, list):
            for block in response_content:
                if isinstance(block, dict) and block.get("type") == "text":
                    print(block["text"])
                elif hasattr(block, "text"):
                    print(block.text)

        print()
