#!/usr/bin/env python3
"""
s07_task_system.py - Tasks (持久化任务系统)

=== 本课要点 ===

引入"持久化任务系统"：任务作为 JSON 文件存储，支持依赖图。

核心洞察：
    "对话外的状态可以在压缩中存活——因为它不在对话里。"

=== 为什么需要持久化任务？ ===

回顾 s03 的 TodoManager：
    - 任务存储在内存中（Python 变量）
    - 压缩对话时，任务列表会丢失
    - 模型可能忘记自己创建的任务

s07 的解决方案：
    - 任务存储在磁盘上（.tasks/*.json）
    - 压缩对话不影响任务文件
    - 模型随时可以读取任务状态

=== 任务系统架构 ===

    .tasks/
      task_1.json  {"id":1, "subject":"...", "status":"completed", ...}
      task_2.json  {"id":2, "blockedBy":[1], "status":"pending", ...}
      task_3.json  {"id":3, "blockedBy":[2], "blocks":[], ...}

=== 依赖图（Dependency Graph）===

    核心概念：任务之间可以有依赖关系

    blockedBy: 我被谁阻塞（必须等它们完成）
    blocks:    我阻塞谁（我完成后它们才能开始）

    示例：
    +----------+     +----------+     +----------+
    | task 1   | --> | task 2   | --> | task 3   |
    | complete |     | blocked  |     | blocked  |
    +----------+     +----------+     +----------+
         |                ^
         +--- 完成 task 1 后，自动从 task 2 的 blockedBy 中移除

    实际场景：
        task 1: 设计数据库 schema
        task 2: 实现数据访问层 (blockedBy: [1])
        task 3: 实现 API 接口 (blockedBy: [2])

=== 与 s03 的对比 ===

    s03 (TodoManager):
        - 内存存储
        - 无依赖关系
        - 压缩后丢失

    s07 (TaskManager):
        - 磁盘存储（JSON 文件）
        - 支持依赖图
        - 永久保存
"""

# ============================================================================
# 第一部分：导入依赖
# ============================================================================

import json        # JSON 序列化，用于任务持久化
import os
import subprocess
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

# 任务存储目录
TASKS_DIR = WORKDIR / ".tasks"

# 系统提示词：强调使用任务工具来规划和跟踪工作
SYSTEM = f"You are a coding agent at {WORKDIR}. Use task tools to plan and track work."


# ============================================================================
# 第三部分：TaskManager 类（本课核心）
# ============================================================================

class TaskManager:
    """
    持久化任务管理器：支持 CRUD 操作和依赖图。

    数据结构（每个任务的 JSON 文件）：
    {
        "id": 1,                    # 唯一标识
        "subject": "实现登录功能",   # 任务标题
        "description": "...",       # 详细描述
        "status": "pending",        # 状态: pending/in_progress/completed
        "blockedBy": [2, 3],        # 被哪些任务阻塞
        "blocks": [4, 5],           # 阻塞哪些任务
        "owner": ""                 # 负责人（可选）
    }

    存储结构：
        .tasks/
        ├── task_1.json
        ├── task_2.json
        └── task_3.json

    为什么用 JSON 文件而不是数据库？
    1. 简单：无需额外依赖
    2. 可读：人可以直接查看和编辑
    3. 版本控制：可以用 git 跟踪变更
    4. 教学目的：便于理解原理
    """

    def __init__(self, tasks_dir: Path):
        """
        初始化任务管理器。

        Args:
            tasks_dir: 任务文件存储目录
        """
        self.dir = tasks_dir
        # 确保目录存在
        self.dir.mkdir(exist_ok=True)
        # 计算下一个可用的任务 ID
        self._next_id = self._max_id() + 1

    def _max_id(self) -> int:
        """
        获取当前最大的任务 ID。

        通过扫描目录中的文件名来确定：
            task_1.json → 1
            task_5.json → 5
            max → 5

        Returns:
            最大的任务 ID，如果没有任务则返回 0
        """
        # 从文件名提取 ID: task_1.json → "task_1" → ["task", "1"] → 1
        ids = [int(f.stem.split("_")[1]) for f in self.dir.glob("task_*.json")]
        return max(ids) if ids else 0

    def _load(self, task_id: int) -> dict:
        """
        加载指定任务的数据。

        Args:
            task_id: 任务 ID

        Returns:
            任务数据字典

        Raises:
            ValueError: 任务不存在时抛出
        """
        path = self.dir / f"task_{task_id}.json"
        if not path.exists():
            raise ValueError(f"Task {task_id} not found")
        return json.loads(path.read_text())

    def _save(self, task: dict):
        """
        保存任务数据到文件。

        Args:
            task: 任务数据字典（必须包含 'id' 字段）
        """
        path = self.dir / f"task_{task['id']}.json"
        # indent=2 使 JSON 易于阅读
        path.write_text(json.dumps(task, indent=2))

    # -------------------------------------------------------------------------
    # CRUD 操作
    # -------------------------------------------------------------------------

    def create(self, subject: str, description: str = "") -> str:
        """
        创建新任务。

        Args:
            subject: 任务标题
            description: 任务描述（可选）

        Returns:
            新创建的任务 JSON 字符串
        """
        task = {
            "id": self._next_id,
            "subject": subject,
            "description": description,
            "status": "pending",      # 新任务默认为待处理
            "blockedBy": [],          # 初始无依赖
            "blocks": [],             # 初始不阻塞任何任务
            "owner": "",              # 初始无负责人
        }
        self._save(task)
        self._next_id += 1
        return json.dumps(task, indent=2)

    def get(self, task_id: int) -> str:
        """
        获取指定任务的详细信息。

        Args:
            task_id: 任务 ID

        Returns:
            任务 JSON 字符串
        """
        return json.dumps(self._load(task_id), indent=2)

    def update(self, task_id: int, status: str = None,
               add_blocked_by: list = None, add_blocks: list = None) -> str:
        """
        更新任务状态或依赖关系。

        这是最复杂的方法，因为需要维护依赖图的一致性。

        Args:
            task_id: 任务 ID
            status: 新状态（可选）
            add_blocked_by: 要添加的 blockedBy 列表（可选）
            add_blocks: 要添加的 blocks 列表（可选）

        Returns:
            更新后的任务 JSON 字符串

        依赖图维护逻辑：
            1. 当任务完成时，从其他任务的 blockedBy 中移除它
            2. 当添加 blocks 时，同时更新被阻塞任务的 blockedBy
        """
        task = self._load(task_id)

        # === 更新状态 ===
        if status:
            if status not in ("pending", "in_progress", "completed"):
                raise ValueError(f"Invalid status: {status}")
            task["status"] = status

            # === 关键：完成任务时清理依赖 ===
            # 当一个任务完成，它不应该再阻塞其他任务
            if status == "completed":
                self._clear_dependency(task_id)

        # === 添加 blockedBy 依赖 ===
        if add_blocked_by:
            # 使用 set 去重
            task["blockedBy"] = list(set(task["blockedBy"] + add_blocked_by))

        # === 添加 blocks 依赖（双向维护）===
        if add_blocks:
            task["blocks"] = list(set(task["blocks"] + add_blocks))

            # 双向维护：同时更新被阻塞任务的 blockedBy
            # 如果 A blocks B，那么 B 的 blockedBy 应该包含 A
            for blocked_id in add_blocks:
                try:
                    blocked = self._load(blocked_id)
                    if task_id not in blocked["blockedBy"]:
                        blocked["blockedBy"].append(task_id)
                        self._save(blocked)
                except ValueError:
                    # 被阻塞的任务不存在，忽略
                    pass

        self._save(task)
        return json.dumps(task, indent=2)

    def _clear_dependency(self, completed_id: int):
        """
        从所有任务的 blockedBy 列表中移除已完成的任务。

        这是依赖图的核心维护逻辑：
        当任务 X 完成时，所有"被 X 阻塞"的任务现在可以开始了。

        示例：
            before:
                task_2: {"blockedBy": [1, 3]}
            after (task 1 completed):
                task_2: {"blockedBy": [3]}

        Args:
            completed_id: 已完成的任务 ID
        """
        for f in self.dir.glob("task_*.json"):
            task = json.loads(f.read_text())
            if completed_id in task.get("blockedBy", []):
                task["blockedBy"].remove(completed_id)
                self._save(task)

    def list_all(self) -> str:
        """
        列出所有任务的摘要信息。

        返回格式：
            [ ] #1: 设计数据库 schema
            [>] #2: 实现数据访问层 (blocked by: [1])
            [x] #3: 编写单元测试

        标记含义：
            [ ] pending     - 待处理
            [>] in_progress - 进行中
            [x] completed   - 已完成

        Returns:
            格式化的任务列表字符串
        """
        tasks = []
        # sorted 确保按文件名排序（即按 ID 排序）
        for f in sorted(self.dir.glob("task_*.json")):
            tasks.append(json.loads(f.read_text()))

        if not tasks:
            return "No tasks."

        lines = []
        for t in tasks:
            # 状态标记
            marker = {
                "pending": "[ ]",
                "in_progress": "[>]",
                "completed": "[x]"
            }.get(t["status"], "[?]")

            # 依赖信息
            blocked = f" (blocked by: {t['blockedBy']})" if t.get("blockedBy") else ""

            lines.append(f"{marker} #{t['id']}: {t['subject']}{blocked}")

        return "\n".join(lines)


# ============================================================================
# 第四部分：创建全局任务管理器实例
# ============================================================================

TASKS = TaskManager(TASKS_DIR)


# ============================================================================
# 第五部分：工具实现函数
# ============================================================================

def safe_path(p: str) -> Path:
    """路径安全检查（防止目录逃逸）"""
    path = (WORKDIR / p).resolve()
    if not path.is_relative_to(WORKDIR):
        raise ValueError(f"Path escapes workspace: {p}")
    return path


def run_bash(command: str) -> str:
    """执行 bash 命令"""
    dangerous = ["rm -rf /", "sudo", "shutdown", "reboot", "> /dev/"]
    if any(d in command for d in dangerous):
        return "Error: Dangerous command blocked"
    try:
        r = subprocess.run(command, shell=True, cwd=WORKDIR,
                           capture_output=True, text=True, timeout=120)
        out = (r.stdout + r.stderr).strip()
        return out[:50000] if out else "(no output)"
    except subprocess.TimeoutExpired:
        return "Error: Timeout (120s)"


def run_read(path: str, limit: int = None) -> str:
    """读取文件内容"""
    try:
        lines = safe_path(path).read_text().splitlines()
        if limit and limit < len(lines):
            lines = lines[:limit] + [f"... ({len(lines) - limit} more)"]
        return "\n".join(lines)[:50000]
    except Exception as e:
        return f"Error: {e}"


def run_write(path: str, content: str) -> str:
    """写入文件"""
    try:
        fp = safe_path(path)
        fp.parent.mkdir(parents=True, exist_ok=True)
        fp.write_text(content)
        return f"Wrote {len(content)} bytes"
    except Exception as e:
        return f"Error: {e}"


def run_edit(path: str, old_text: str, new_text: str) -> str:
    """编辑文件（查找替换）"""
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
# 第六部分：工具分发映射
# ============================================================================

TOOL_HANDLERS = {
    # 基础工具
    "bash":        lambda **kw: run_bash(kw["command"]),
    "read_file":   lambda **kw: run_read(kw["path"], kw.get("limit")),
    "write_file":  lambda **kw: run_write(kw["path"], kw["content"]),
    "edit_file":   lambda **kw: run_edit(kw["path"], kw["old_text"], kw["new_text"]),

    # 任务工具（新增）
    "task_create": lambda **kw: TASKS.create(kw["subject"], kw.get("description", "")),
    "task_update": lambda **kw: TASKS.update(
        kw["task_id"],
        kw.get("status"),
        kw.get("addBlockedBy"),
        kw.get("addBlocks")
    ),
    "task_list":   lambda **kw: TASKS.list_all(),
    "task_get":    lambda **kw: TASKS.get(kw["task_id"]),
}


# ============================================================================
# 第七部分：工具定义列表
# ============================================================================

TOOLS = [
    # 基础工具
    {"name": "bash", "description": "Run a shell command.",
     "input_schema": {"type": "object", "properties": {"command": {"type": "string"}}, "required": ["command"]}},
    {"name": "read_file", "description": "Read file contents.",
     "input_schema": {"type": "object", "properties": {"path": {"type": "string"}, "limit": {"type": "integer"}}, "required": ["path"]}},
    {"name": "write_file", "description": "Write content to file.",
     "input_schema": {"type": "object", "properties": {"path": {"type": "string"}, "content": {"type": "string"}}, "required": ["path", "content"]}},
    {"name": "edit_file", "description": "Replace exact text in file.",
     "input_schema": {"type": "object", "properties": {"path": {"type": "string"}, "old_text": {"type": "string"}, "new_text": {"type": "string"}}, "required": ["path", "old_text", "new_text"]}},

    # 任务工具（新增）
    {
        "name": "task_create",
        "description": "Create a new task.",
        "input_schema": {
            "type": "object",
            "properties": {
                "subject": {"type": "string"},
                "description": {"type": "string"}
            },
            "required": ["subject"]
        }
    },
    {
        "name": "task_update",
        "description": "Update a task's status or dependencies.",
        "input_schema": {
            "type": "object",
            "properties": {
                "task_id": {"type": "integer"},
                "status": {
                    "type": "string",
                    "enum": ["pending", "in_progress", "completed"]
                },
                "addBlockedBy": {
                    "type": "array",
                    "items": {"type": "integer"}
                },
                "addBlocks": {
                    "type": "array",
                    "items": {"type": "integer"}
                }
            },
            "required": ["task_id"]
        }
    },
    {
        "name": "task_list",
        "description": "List all tasks with status summary.",
        "input_schema": {"type": "object", "properties": {}}
    },
    {
        "name": "task_get",
        "description": "Get full details of a task by ID.",
        "input_schema": {
            "type": "object",
            "properties": {
                "task_id": {"type": "integer"}
            },
            "required": ["task_id"]
        }
    },
]


# ============================================================================
# 第八部分：序列化响应内容
# ============================================================================

def serialize_content(content):
    """
    将 API 响应的 content 转换为可序列化的字典格式。

    重要：不能使用 model_dump()！必须手动提取字段。
    详见 s01-s03 的注释说明。
    """
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
# 第九部分：Agent 循环
# ============================================================================

def agent_loop(messages: list):
    """
    Agent 核心循环。

    与之前版本的区别：
    - 增加了 task_* 系列工具
    - 任务状态持久化存储，不受对话压缩影响

    典型工作流程：
    1. 用户: "帮我重构这个项目"
    2. 模型: task_create("分析现有代码结构")
    3. 模型: task_create("设计新架构")
    4. 模型: task_update(2, addBlockedBy=[1])  # 先分析，再设计
    5. 模型: task_update(1, status="in_progress")
    6. 模型: [开始分析...]
    7. 模型: task_update(1, status="completed")
    8. 模型: task_update(2, status="in_progress")  # 依赖解除
    9. ...
    """
    while True:
        response = client.messages.create(
            model=MODEL, system=SYSTEM, messages=messages,
            tools=TOOLS, max_tokens=8000,
        )

        # 保存响应（序列化以避免兼容性问题）
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
                    "content": str(output)
                })

        messages.append({"role": "user", "content": results})


# ============================================================================
# 第十部分：主程序入口
# ============================================================================

if __name__ == "__main__":
    """
    交互式命令行界面。

    使用示例：
        s07 >> 帮我规划一个用户认证系统的开发任务

    观察：
        1. 模型会创建多个任务
        2. 任务之间可能有依赖关系
        3. 查看 .tasks/ 目录可以看到持久化的任务文件
        4. 即使重启程序，任务仍然存在

    测试持久化：
        1. 创建几个任务
        2. 退出程序 (q)
        3. 重新运行程序
        4. 让模型调用 task_list，任务仍然在
    """
    history = []

    print("=" * 60)
    print("s07_task_system.py - 持久化任务系统示例")
    print("新增工具: task_create, task_update, task_list, task_get")
    print(f"任务存储目录: {TASKS_DIR}")
    print("输入 'q' 或 'exit' 退出")
    print("=" * 60)

    while True:
        try:
            query = input("\033[36ms07 >> \033[0m")
        except (EOFError, KeyboardInterrupt):
            break

        if query.strip().lower() in ("q", "exit", ""):
            break

        history.append({"role": "user", "content": query})
        agent_loop(history)

        # 打印模型的文字回复（处理序列化后的字典格式）
        response_content = history[-1]["content"]
        if isinstance(response_content, list):
            for block in response_content:
                # 处理序列化后的字典格式
                if isinstance(block, dict) and block.get("type") == "text":
                    print(block["text"])
                # 兼容 Pydantic 对象格式
                elif hasattr(block, "text"):
                    print(block.text)

        print()
