#!/usr/bin/env python3
"""
s08_background_tasks.py - Background Tasks (后台任务)

=== 本课要点 ===

引入"后台任务"概念：命令在后台线程中执行，Agent 不需要阻塞等待。

核心洞察：
    "发射后不管（Fire and Forget）—— Agent 在命令运行时不会阻塞。"

=== 为什么需要后台任务？ ===

场景：用户说"运行测试套件"（需要 5 分钟）

没有后台任务：
    Agent: bash("pytest")
    ... 阻塞 5 分钟 ...
    Agent: 测试完成，结果是...
    （Agent 无法做其他事情）

有后台任务：
    Agent: background_run("pytest")
    Agent: 测试已在后台启动，我继续做其他事情...
    Agent: （收到通知）测试完成，结果是...
    （Agent 可以并行处理多个任务）

=== 架构图 ===

    主线程                          后台线程
    +-----------------+            +-----------------+
    | agent loop      |            | task executes   |
    | ...             |            | ...             |
    | [LLM call] <---+---------- | enqueue(result) |
    |  ^drain queue   |            +-----------------+
    +-----------------+

    时间线：
    Agent ----[spawn A]----[spawn B]----[其他工作]----
                 |              |
                 v              v
              [A 运行]      [B 运行]        (并行执行)
                 |              |
                 +-- 通知队列 --> [结果注入到对话]

=== 通知机制 ===

    问题：后台任务完成时，如何通知 LLM？

    解决：通知队列（Notification Queue）
    1. 后台线程完成后，将结果推入队列
    2. 每次 LLM 调用前，清空队列
    3. 将完成的结果注入到对话中

    +------------------+
    | Notification     |
    | Queue            |
    | +--------------+ |
    | | task A done  | | ← 后台线程推入
    | | task B done  | |
    | +--------------+ |
    +--------+---------+
             |
             | drain (清空)
             v
    +------------------+
    | messages.append( |
    |   <background-   |
    |    results>...   | → 注入到对话
    | )                |
    +------------------+
"""

# ============================================================================
# 第一部分：导入依赖
# ============================================================================

import os
import subprocess
import threading   # 多线程支持
import uuid        # 生成唯一任务 ID
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

# 系统提示词：强调使用 background_run 执行长时间命令
SYSTEM = f"You are a coding agent at {WORKDIR}. Use background_run for long-running commands."


# ============================================================================
# 第三部分：BackgroundManager 类（本课核心）
# ============================================================================

class BackgroundManager:
    """
    后台任务管理器：使用线程执行命令，通过队列通知结果。

    核心组件：
    1. tasks: 任务状态字典 {task_id: {status, result, command}}
    2. _notification_queue: 完成通知队列
    3. _lock: 线程锁（保护队列的并发访问）

    生命周期：
        1. run() 被调用 → 创建线程，立即返回 task_id
        2. 线程执行命令 → 完成后推入通知队列
        3. agent_loop 中 drain_notifications() → 取出通知
        4. 通知注入到对话 → LLM 看到结果
    """

    def __init__(self):
        # 任务状态存储: {task_id: {status, result, command}}
        self.tasks = {}
        # 完成通知队列（后台线程写入，主线程读取）
        self._notification_queue = []
        # 线程锁：保护队列的并发访问
        # 因为多个后台线程可能同时完成任务
        self._lock = threading.Lock()

    def run(self, command: str) -> str:
        """
        在后台线程中启动命令，立即返回任务 ID。

        这是"发射后不管"模式的关键：
        - 主线程不等待命令完成
        - 立即返回，让 Agent 继续工作
        - 命令在独立线程中执行

        Args:
            command: 要执行的 shell 命令

        Returns:
            启动确认消息，包含 task_id
        """
        # 生成短 UUID 作为任务 ID（取前 8 位）
        task_id = str(uuid.uuid4())[:8]

        # 初始化任务状态
        self.tasks[task_id] = {
            "status": "running",
            "result": None,
            "command": command
        }

        # 创建并启动后台线程
        thread = threading.Thread(
            target=self._execute,       # 线程执行的函数
            args=(task_id, command),    # 传递给函数的参数
            daemon=True                 # 守护线程：主线程退出时自动终止
        )
        thread.start()

        # 立即返回！不等待命令完成
        return f"Background task {task_id} started: {command[:80]}"

    def _execute(self, task_id: str, command: str):
        """
        线程执行函数：运行命令，捕获输出，推入通知队列。

        这个方法在后台线程中执行，流程：
        1. 执行 subprocess 命令
        2. 捕获输出（stdout + stderr）
        3. 更新任务状态
        4. 将结果推入通知队列

        Args:
            task_id: 任务 ID
            command: 要执行的命令
        """
        try:
            # 执行命令（最长等待 300 秒）
            r = subprocess.run(
                command, shell=True, cwd=WORKDIR,
                capture_output=True, text=True, timeout=300
            )
            output = (r.stdout + r.stderr).strip()[:50000]
            status = "completed"

        except subprocess.TimeoutExpired:
            output = "Error: Timeout (300s)"
            status = "timeout"

        except Exception as e:
            output = f"Error: {e}"
            status = "error"

        # 更新任务状态
        self.tasks[task_id]["status"] = status
        self.tasks[task_id]["result"] = output or "(no output)"

        # === 关键：将结果推入通知队列 ===
        # 使用锁保护，因为可能有多个线程同时完成
        with self._lock:
            self._notification_queue.append({
                "task_id": task_id,
                "status": status,
                "command": command[:80],
                # 通知中只包含前 500 字符，完整结果可以通过 check 获取
                "result": (output or "(no output)")[:500],
            })

    def check(self, task_id: str = None) -> str:
        """
        检查后台任务状态。

        两种用法：
        1. check(task_id) - 查看特定任务的详细状态
        2. check() - 列出所有后台任务

        Args:
            task_id: 可选，要查看的任务 ID

        Returns:
            任务状态信息
        """
        if task_id:
            # 查看特定任务
            t = self.tasks.get(task_id)
            if not t:
                return f"Error: Unknown task {task_id}"
            return f"[{t['status']}] {t['command'][:60]}\n{t.get('result') or '(running)'}"

        # 列出所有任务
        lines = []
        for tid, t in self.tasks.items():
            lines.append(f"{tid}: [{t['status']}] {t['command'][:60]}")
        return "\n".join(lines) if lines else "No background tasks."

    def drain_notifications(self) -> list:
        """
        清空并返回所有待处理的完成通知。

        这是"拉取"模式：
        - 不是后台线程主动打断 Agent
        - 而是 Agent 在合适的时机（LLM 调用前）主动拉取

        为什么用"drain"（排空）而不是"peek"（查看）？
        - 每个通知只应该被处理一次
        - drain 后队列清空，下次调用返回新通知

        Returns:
            通知列表，每个通知是一个字典:
            {task_id, status, command, result}
        """
        with self._lock:
            # 复制列表（不是引用）
            notifs = list(self._notification_queue)
            # 清空队列
            self._notification_queue.clear()
        return notifs


# ============================================================================
# 第四部分：创建全局后台管理器实例
# ============================================================================

BG = BackgroundManager()


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
    """执行 bash 命令（阻塞式，与 background_run 对比）"""
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
    "bash":             lambda **kw: run_bash(kw["command"]),
    "read_file":        lambda **kw: run_read(kw["path"], kw.get("limit")),
    "write_file":       lambda **kw: run_write(kw["path"], kw["content"]),
    "edit_file":        lambda **kw: run_edit(kw["path"], kw["old_text"], kw["new_text"]),

    # 后台任务工具（新增）
    "background_run":   lambda **kw: BG.run(kw["command"]),
    "check_background": lambda **kw: BG.check(kw.get("task_id")),
}


# ============================================================================
# 第七部分：工具定义列表
# ============================================================================

TOOLS = [
    # 基础工具（阻塞式）
    {
        "name": "bash",
        "description": "Run a shell command (blocking).",  # 注意：强调是阻塞的
        "input_schema": {
            "type": "object",
            "properties": {"command": {"type": "string"}},
            "required": ["command"]
        }
    },
    {"name": "read_file", "description": "Read file contents.",
     "input_schema": {"type": "object", "properties": {"path": {"type": "string"}, "limit": {"type": "integer"}}, "required": ["path"]}},
    {"name": "write_file", "description": "Write content to file.",
     "input_schema": {"type": "object", "properties": {"path": {"type": "string"}, "content": {"type": "string"}}, "required": ["path", "content"]}},
    {"name": "edit_file", "description": "Replace exact text in file.",
     "input_schema": {"type": "object", "properties": {"path": {"type": "string"}, "old_text": {"type": "string"}, "new_text": {"type": "string"}}, "required": ["path", "old_text", "new_text"]}},

    # 后台任务工具（新增）
    {
        "name": "background_run",
        "description": "Run command in background thread. Returns task_id immediately.",
        "input_schema": {
            "type": "object",
            "properties": {"command": {"type": "string"}},
            "required": ["command"]
        }
    },
    {
        "name": "check_background",
        "description": "Check background task status. Omit task_id to list all.",
        "input_schema": {
            "type": "object",
            "properties": {
                "task_id": {"type": "string"}
            }
            # task_id 是可选的，不在 required 中
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
# 第九部分：Agent 循环（包含通知注入）
# ============================================================================

def agent_loop(messages: list):
    """
    Agent 核心循环，包含后台任务通知机制。

    与之前版本的关键区别：
    - 每次 LLM 调用前，检查并注入后台任务完成通知

    通知注入流程：
    1. drain_notifications() 获取所有完成通知
    2. 格式化为 <background-results> 消息
    3. 注入到对话中
    4. 添加一个 assistant 确认消息（保持对话格式）
    5. 然后才调用 LLM

    为什么需要 assistant 确认消息？
    - API 要求 user/assistant 消息交替
    - 注入的通知是 user 消息
    - 需要一个 assistant 回复来保持格式
    """
    while True:
        # === 关键：每次 LLM 调用前，注入后台任务通知 ===
        notifs = BG.drain_notifications()
        if notifs and messages:
            # 格式化通知消息
            notif_text = "\n".join(
                f"[bg:{n['task_id']}] {n['status']}: {n['result']}" for n in notifs
            )

            # 注入通知（作为 user 消息）
            messages.append({
                "role": "user",
                "content": f"<background-results>\n{notif_text}\n</background-results>"
            })

            # 添加 assistant 确认（保持对话格式）
            messages.append({
                "role": "assistant",
                "content": "Noted background results."
            })

        # 调用 LLM
        response = client.messages.create(
            model=MODEL, system=SYSTEM, messages=messages,
            tools=TOOLS, max_tokens=8000,
        )

        # 保存响应（序列化以避免兼容性问题）
        messages.append({"role": "assistant", "content": serialize_content(response.content)})

        if response.stop_reason != "tool_use":
            return

        # 执行工具调用
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
        s08 >> 在后台运行 "sleep 10 && echo done"，同时读取 README.md

    观察：
        1. background_run 立即返回任务 ID
        2. Agent 继续执行其他操作（读取文件）
        3. 10 秒后，后台任务完成
        4. 下次 LLM 调用时，Agent 收到完成通知

    测试并发：
        s08 >> 同时启动三个后台任务：sleep 5, sleep 3, sleep 1

    观察不同任务按完成顺序通知（1秒 → 3秒 → 5秒）
    """
    history = []

    print("=" * 60)
    print("s08_background_tasks.py - 后台任务示例")
    print("新增工具: background_run, check_background")
    print("特性: 非阻塞执行、通知队列、并行任务")
    print("输入 'q' 或 'exit' 退出")
    print("=" * 60)

    while True:
        try:
            query = input("\033[36ms08 >> \033[0m")
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
