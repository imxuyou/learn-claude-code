#!/usr/bin/env python3
"""
s03_todo_write.py - TodoWrite (任务追踪与自我管理)

=== 本课要点 ===

让 LLM 自己管理任务进度！通过 TodoManager，模型可以：
1. 创建任务列表
2. 标记当前正在做的任务
3. 完成后打勾

核心洞察：
    "Agent 可以追踪自己的进度 -- 而且我能看到它。"

这解决了一个常见问题：复杂任务时，你不知道 AI 做到哪一步了。

=== 为什么需要 TodoManager？ ===

场景：用户说"帮我重构这个项目的认证模块"

没有 TodoManager：
    - LLM 闷头干活，你不知道它在做什么
    - 中途出错时，不知道完成了哪些步骤
    - 无法判断整体进度

有 TodoManager：
    [ ] #1: 分析现有认证代码
    [>] #2: 提取公共接口      ← 正在做这个
    [ ] #3: 实现新的认证服务
    [ ] #4: 更新调用方
    [ ] #5: 添加测试

    (1/5 completed)

=== 架构图 ===

    +----------+      +-------+      +---------+
    |   User   | ---> |  LLM  | ---> | Tools   |
    |  prompt  |      |       |      | + todo  |
    +----------+      +---+---+      +----+----+
                          ^               |
                          |   tool_result |
                          +---------------+
                                |
                    +-----------+-----------+
                    | TodoManager state     |
                    | [ ] task A            |
                    | [>] task B <- doing   |
                    | [x] task C            |
                    +-----------------------+
                                |
                    if rounds_since_todo >= 3:
                      inject <reminder>

=== Nag Reminder 机制 ===

LLM 有时会"忘记"更新任务状态。为了解决这个问题，
如果连续 3 轮没有调用 todo 工具，系统会注入一个提醒：

    <reminder>Update your todos.</reminder>

这是一种"软约束"：不强制，但温柔地提醒。

重要：reminder 必须放在 tool_result 之后！
    否则 API 会报错 "tool_use ids were found without tool_result"
"""

# ============================================================================
# 第一部分：导入依赖
# ============================================================================

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

# 系统提示词：明确告诉模型使用 todo 工具来规划多步骤任务
# "Mark in_progress before starting, completed when done" 是关键指令
SYSTEM = f"""You are a coding agent at {WORKDIR}.
Use the todo tool to plan multi-step tasks. Mark in_progress before starting, completed when done.
Prefer tools over prose."""


# ============================================================================
# 第三部分：TodoManager 类（本课核心）
# ============================================================================

class TodoManager:
    """
    任务管理器：维护一个结构化的任务列表。

    这不是普通的内存变量 -- 它是 LLM 可以读写的"外部状态"。

    任务状态流转：
        pending (待办) → in_progress (进行中) → completed (已完成)

    约束规则：
        1. 最多 20 个任务（防止滥用）
        2. 同一时间只能有 1 个 in_progress 任务（强制专注）
        3. 每个任务必须有 id、text、status

    为什么是类而不是简单的列表？
        - 封装验证逻辑
        - 提供格式化输出
        - 便于未来扩展（如持久化、依赖关系等）
    """

    def __init__(self):
        # 任务列表，每个元素是 {"id": "1", "text": "...", "status": "pending"}
        self.items = []

    def update(self, items: list) -> str:
        """
        更新整个任务列表。

        注意：这是"全量更新"而非"增量更新"。
        每次调用都会替换整个列表，这样设计更简单：
        - LLM 不需要记住复杂的增删改命令
        - 直接输出当前的完整状态即可

        Args:
            items: 新的任务列表，格式如：
                [
                    {"id": "1", "text": "分析代码", "status": "completed"},
                    {"id": "2", "text": "重构函数", "status": "in_progress"},
                    {"id": "3", "text": "写测试", "status": "pending"},
                ]

        Returns:
            格式化的任务列表字符串（会返回给 LLM 作为确认）

        Raises:
            ValueError: 如果数据不符合约束规则
        """
        # 约束 1：最多 20 个任务
        if len(items) > 20:
            raise ValueError("Max 20 todos allowed")

        validated = []
        in_progress_count = 0

        for i, item in enumerate(items):
            # 提取并清理字段
            text = str(item.get("text", "")).strip()
            status = str(item.get("status", "pending")).lower()
            item_id = str(item.get("id", str(i + 1)))

            # 约束：text 不能为空
            if not text:
                raise ValueError(f"Item {item_id}: text required")

            # 约束：status 必须是有效值
            if status not in ("pending", "in_progress", "completed"):
                raise ValueError(f"Item {item_id}: invalid status '{status}'")

            # 统计 in_progress 数量
            if status == "in_progress":
                in_progress_count += 1

            validated.append({"id": item_id, "text": text, "status": status})

        # 约束 2：同时只能有一个 in_progress
        # 这个规则强制 LLM 专注于一个任务，避免并行思维混乱
        if in_progress_count > 1:
            raise ValueError("Only one task can be in_progress at a time")

        self.items = validated
        return self.render()

    def render(self) -> str:
        """
        将任务列表渲染为人类可读的格式。

        输出示例：
            [ ] #1: 分析现有代码
            [>] #2: 重构核心函数
            [x] #3: 更新文档

            (1/3 completed)

        这个格式设计考虑：
        - [ ] [>] [x] 符号直观，类似 markdown checkbox
        - #id 方便引用特定任务
        - 底部统计让进度一目了然
        """
        if not self.items:
            return "No todos."

        lines = []
        for item in self.items:
            # 状态到符号的映射
            marker = {
                "pending": "[ ]",      # 待办
                "in_progress": "[>]",  # 进行中（箭头表示当前）
                "completed": "[x]"     # 已完成（打勾）
            }[item["status"]]

            lines.append(f"{marker} #{item['id']}: {item['text']}")

        # 添加完成统计
        done = sum(1 for t in self.items if t["status"] == "completed")
        lines.append(f"\n({done}/{len(self.items)} completed)")

        return "\n".join(lines)


# 全局 TodoManager 实例
# 在整个会话中保持状态
TODO = TodoManager()


# ============================================================================
# 第四部分：工具实现函数（与 s02 相同）
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
        content = fp.read_text()
        if old_text not in content:
            return f"Error: Text not found in {path}"
        fp.write_text(content.replace(old_text, new_text, 1))
        return f"Edited {path}"
    except Exception as e:
        return f"Error: {e}"


# ============================================================================
# 第五部分：工具分发映射
# ============================================================================

# 注意：新增了 "todo" 工具
TOOL_HANDLERS = {
    "bash":       lambda **kw: run_bash(kw["command"]),
    "read_file":  lambda **kw: run_read(kw["path"], kw.get("limit")),
    "write_file": lambda **kw: run_write(kw["path"], kw["content"]),
    "edit_file":  lambda **kw: run_edit(kw["path"], kw["old_text"], kw["new_text"]),
    "todo":       lambda **kw: TODO.update(kw["items"]),  # ← 新增！
}

# ============================================================================
# 第六部分：工具定义列表
# ============================================================================

TOOLS = [
    # 工具 1-4：与 s02 相同
    {"name": "bash", "description": "Run a shell command.",
     "input_schema": {"type": "object", "properties": {"command": {"type": "string"}}, "required": ["command"]}},
    {"name": "read_file", "description": "Read file contents.",
     "input_schema": {"type": "object", "properties": {"path": {"type": "string"}, "limit": {"type": "integer"}}, "required": ["path"]}},
    {"name": "write_file", "description": "Write content to file.",
     "input_schema": {"type": "object", "properties": {"path": {"type": "string"}, "content": {"type": "string"}}, "required": ["path", "content"]}},
    {"name": "edit_file", "description": "Replace exact text in file.",
     "input_schema": {"type": "object", "properties": {"path": {"type": "string"}, "old_text": {"type": "string"}, "new_text": {"type": "string"}}, "required": ["path", "old_text", "new_text"]}},

    # 工具 5：todo（新增！）
    # 这是一个结构化数据工具，接受数组类型的参数
    {"name": "todo", "description": "Update task list. Track progress on multi-step tasks.",
     "input_schema": {"type": "object", "properties": {"items": {"type": "array", "items": {"type": "object", "properties": {"id": {"type": "string"}, "text": {"type": "string"}, "status": {"type": "string", "enum": ["pending", "in_progress", "completed"]}}, "required": ["id", "text", "status"]}}}, "required": ["items"]}},
]


# ============================================================================
# 第七部分：序列化响应内容（关键！）
# ============================================================================

def serialize_content(content):
    """
    将 API 响应的 content 转换为可序列化的字典格式。

    重要警告：不能使用 model_dump()！
    =================================

    我们在调试中发现，model_dump() 输出的格式与 API 期望的不同：

        model_dump() 输出:
        {"id": "toolu_xxx", "caller": {"type": "direct"}, "input": {...}}

        API 期望的格式:
        {"type": "tool_use", "id": "toolu_xxx", "name": "bash", "input": {...}}

    注意 model_dump() 的输出缺少 "type" 和 "name" 字段！
    这会导致 API 返回 400 错误：
        "tool_use ids were found without tool_result blocks"

    必须手动提取必要字段，确保格式正确。
    """
    result = []
    for block in content:
        if block.type == "text":
            # TextBlock: 只需要 type 和 text
            result.append({"type": "text", "text": block.text})
        elif block.type == "tool_use":
            # ToolUseBlock: 需要 type, id, name, input
            result.append({
                "type": "tool_use",
                "id": block.id,
                "name": block.name,
                "input": block.input
            })
        # 忽略其他类型的块（如 thinking 等）
    return result


# ============================================================================
# 第八部分：Agent 循环（带 Nag Reminder）
# ============================================================================

def agent_loop(messages: list):
    """
    Agent 核心循环，带有 Nag Reminder 机制。

    与 s02 的区别：
    - 追踪 rounds_since_todo：记录上次使用 todo 工具后过了几轮
    - 如果超过 3 轮没用 todo，注入提醒消息

    为什么需要 Nag Reminder？
    - LLM 可能会"忘记"更新任务状态
    - 特别是在复杂任务中，容易直接执行而不汇报进度
    - 轻量提醒比强制约束更符合 LLM 的工作方式

    重要：reminder 必须放在 tool_result 之后！
    =========================================
    我们在调试中发现，如果把 reminder 放在 tool_result 之前：

        错误的顺序（会导致 API 报错）:
        [
          {"type": "text", "text": "<reminder>..."},      ← API 先看到这个
          {"type": "tool_result", "tool_use_id": "..."}   ← 找不到匹配
        ]

        正确的顺序:
        [
          {"type": "tool_result", "tool_use_id": "..."},  ← API 先找到 tool_result
          {"type": "text", "text": "<reminder>..."}       ← 提醒放在后面
        ]

    API 要求 tool_result 必须紧跟在包含 tool_use 的 assistant 消息之后。
    """
    # 追踪自上次使用 todo 工具以来的轮数
    rounds_since_todo = 0

    while True:
        # 调用 API
        response = client.messages.create(
            model=MODEL, system=SYSTEM, messages=messages,
            tools=TOOLS, max_tokens=8000,
        )

        # 保存响应（必须序列化！）
        messages.append({"role": "assistant", "content": serialize_content(response.content)})

        # 检查是否停止
        if response.stop_reason != "tool_use":
            return

        # 执行工具调用
        results = []
        used_todo = False  # 标记本轮是否使用了 todo 工具

        for block in response.content:
            if block.type == "tool_use":
                handler = TOOL_HANDLERS.get(block.name)

                try:
                    output = handler(**block.input) if handler else f"Unknown tool: {block.name}"
                except Exception as e:
                    # 捕获工具执行错误（如 TodoManager 的验证错误）
                    output = f"Error: {e}"

                print(f"> {block.name}: {str(output)[:200]}")
                results.append({
                    "type": "tool_result",
                    "tool_use_id": block.id,
                    "content": str(output)
                })

                # 检查是否使用了 todo 工具
                if block.name == "todo":
                    used_todo = True

        # === Nag Reminder 逻辑 ===
        # 如果使用了 todo，重置计数器；否则计数器 +1
        rounds_since_todo = 0 if used_todo else rounds_since_todo + 1

        # 如果连续 3 轮没有使用 todo，注入提醒
        # 重要：使用 append 而非 insert，把提醒放在 tool_result 之后！
        if rounds_since_todo >= 3:
            results.append({
                "type": "text",
                "text": "<reminder>Update your todos.</reminder>"
            })

        messages.append({"role": "user", "content": results})


# ============================================================================
# 第九部分：主程序入口
# ============================================================================

if __name__ == "__main__":
    """
    交互式命令行界面。

    使用示例：
        s03 >> 帮我创建一个简单的 Flask Web 应用，包含首页和关于页面

    观察 LLM 如何：
        1. 先用 todo 工具规划任务
        2. 逐个标记 in_progress 并执行
        3. 完成后标记 completed
    """
    history = []

    print("=" * 60)
    print("s03_todo_write.py - 任务追踪示例")
    print("新增工具: todo（任务管理）")
    print("特性: Nag Reminder（提醒更新进度）")
    print("输入 'q' 或 'exit' 退出")
    print("=" * 60)

    while True:
        try:
            query = input("\033[36ms03 >> \033[0m")
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
                    print(block.get("text", ""))

        # 显示当前任务列表状态
        if TODO.items:
            print("\n--- 当前任务列表 ---")
            print(TODO.render())
            print("-------------------")

        print()
