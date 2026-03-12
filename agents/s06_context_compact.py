#!/usr/bin/env python3
"""
s06_context_compact.py - Compact (上下文压缩)

=== 本课要点 ===

引入"上下文压缩"概念：让 Agent 可以永远工作下去。

核心洞察：
    "Agent 可以策略性地遗忘，然后继续无限工作。"

=== 为什么需要上下文压缩？ ===

问题：LLM 的上下文窗口是有限的（如 200K tokens）
    - 长时间工作会累积大量工具调用结果
    - 特别是读取文件、执行命令等操作产生大量输出
    - 最终会超出 token 限制，导致 Agent 停止工作

解决：三层压缩管道
    - Layer 1: micro_compact - 静默压缩旧的工具结果
    - Layer 2: auto_compact - 自动保存并总结
    - Layer 3: compact tool - 手动触发压缩

=== 三层压缩架构 ===

    Every turn（每轮循环）:
    +------------------+
    | Tool call result |
    +------------------+
            |
            v
    [Layer 1: micro_compact]        (静默执行，每轮都做)
      将 3 轮之前的 tool_result 内容
      替换为 "[Previous: used {tool_name}]"
            |
            v
    [Check: tokens > 50000?]        (检查 token 阈值)
       |               |
       no              yes
       |               |
       v               v
    continue    [Layer 2: auto_compact]
                  保存完整对话到 .transcripts/
                  调用 LLM 总结对话
                  用 [summary] 替换所有消息
                        |
                        v
                [Layer 3: compact tool]
                  模型调用 compact → 立即总结
                  与 auto 相同，手动触发

=== 为什么是三层？ ===

    Layer 1 - 细粒度压缩（每轮）:
        - 代价低：只是字符串替换
        - 效果：减少旧工具结果的 token 消耗
        - 保留最近 3 个结果，足以让模型理解当前状态

    Layer 2 - 粗粒度压缩（自动触发）:
        - 当 token 超过阈值时触发
        - 保存完整历史到文件（可以回溯）
        - 用 LLM 生成摘要替换整个对话

    Layer 3 - 手动压缩（模型触发）:
        - 模型觉得上下文太长时可以主动调用
        - 与 Layer 2 相同的压缩流程
        - 给模型更多控制权
"""

# ============================================================================
# 第一部分：导入依赖
# ============================================================================

import json        # JSON 序列化，用于保存对话历史
import os          # 操作系统功能
import subprocess  # 执行系统命令
import time        # 时间戳，用于命名转录文件
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

SYSTEM = f"You are a coding agent at {WORKDIR}. Use tools to solve tasks."

# === 压缩相关配置 ===
THRESHOLD = 50000      # token 阈值，超过此值触发自动压缩
TRANSCRIPT_DIR = WORKDIR / ".transcripts"  # 对话历史保存目录
KEEP_RECENT = 3        # Layer 1 保留最近几个工具结果


# ============================================================================
# 第三部分：Token 估算函数
# ============================================================================

def estimate_tokens(messages: list) -> int:
    """
    估算消息列表的 token 数量。

    为什么用估算而不是精确计算？
    1. 精确计算需要调用 tokenizer，有性能开销
    2. 我们只需要知道"大概多少"来决定是否压缩
    3. 4 字符 ≈ 1 token 是一个合理的估算

    Args:
        messages: 对话历史列表

    Returns:
        估算的 token 数量
    """
    # 将整个 messages 转为字符串，然后除以 4
    # 这是一个粗略但实用的估算方法
    return len(str(messages)) // 4


# ============================================================================
# 第四部分：Layer 1 - micro_compact（微观压缩）
# ============================================================================

def micro_compact(messages: list) -> list:
    """
    Layer 1: 微观压缩 - 替换旧的工具结果。

    这是最轻量级的压缩方式：
    - 每轮都执行
    - 只替换字符串内容，不改变消息结构
    - 保留最近 KEEP_RECENT 个工具结果完整

    为什么保留最近几个？
    - 模型需要看到最近的操作结果来理解当前状态
    - 太旧的结果通常已经不重要了
    - 3 个是一个平衡点：足够的上下文，又不会太长

    压缩前:
        tool_result: "文件内容有 1000 行..."  (很长)

    压缩后:
        tool_result: "[Previous: used read_file]"  (很短)

    Args:
        messages: 对话历史列表

    Returns:
        压缩后的消息列表（原地修改）
    """
    # === 步骤 1: 收集所有 tool_result 的位置和内容 ===
    # 格式: (消息索引, 内容块索引, tool_result字典)
    tool_results = []

    for msg_idx, msg in enumerate(messages):
        # tool_result 总是在 user 角色的消息中（API 规定）
        if msg["role"] == "user" and isinstance(msg.get("content"), list):
            for part_idx, part in enumerate(msg["content"]):
                if isinstance(part, dict) and part.get("type") == "tool_result":
                    tool_results.append((msg_idx, part_idx, part))

    # 如果工具结果数量不超过 KEEP_RECENT，不需要压缩
    if len(tool_results) <= KEEP_RECENT:
        return messages

    # === 步骤 2: 构建 tool_use_id → tool_name 的映射 ===
    # 我们需要知道每个 tool_result 对应的是什么工具
    # 这样压缩后的占位符更有意义：[Previous: used bash] vs [Previous: used unknown]
    tool_name_map = {}

    for msg in messages:
        if msg["role"] == "assistant":
            content = msg.get("content", [])
            if isinstance(content, list):
                for block in content:
                    # 处理 Pydantic 对象（原始响应）
                    if hasattr(block, "type") and block.type == "tool_use":
                        tool_name_map[block.id] = block.name
                    # 处理序列化后的字典（s01-s05 的修复）
                    elif isinstance(block, dict) and block.get("type") == "tool_use":
                        tool_name_map[block.get("id")] = block.get("name")

    # === 步骤 3: 压缩旧的工具结果 ===
    # 保留最后 KEEP_RECENT 个，压缩其余的
    to_clear = tool_results[:-KEEP_RECENT]

    for _, _, result in to_clear:
        content = result.get("content")
        # 只压缩超过 100 字符的内容（太短的不值得压缩）
        if isinstance(content, str) and len(content) > 100:
            tool_id = result.get("tool_use_id", "")
            tool_name = tool_name_map.get(tool_id, "unknown")
            # 替换为简短的占位符
            result["content"] = f"[Previous: used {tool_name}]"

    return messages


# ============================================================================
# 第五部分：Layer 2 - auto_compact（自动压缩）
# ============================================================================

def auto_compact(messages: list) -> list:
    """
    Layer 2: 自动压缩 - 保存历史并用摘要替换。

    当 token 超过阈值时触发，执行以下步骤：
    1. 保存完整对话历史到文件（可追溯）
    2. 调用 LLM 生成对话摘要
    3. 用摘要替换所有消息

    为什么要保存历史？
    - 摘要可能丢失细节
    - 需要回溯时可以查看原始对话
    - 调试和审计用途

    压缩前:
        messages = [很多消息，可能几十条]

    压缩后:
        messages = [
            {user: "[Conversation compressed...]\n\n摘要内容"},
            {assistant: "Understood. I have the context..."}
        ]

    Args:
        messages: 原始对话历史

    Returns:
        压缩后的新消息列表（只有 2 条）
    """
    # === 步骤 1: 保存完整对话历史到磁盘 ===
    TRANSCRIPT_DIR.mkdir(exist_ok=True)

    # 使用时间戳命名，确保唯一性
    transcript_path = TRANSCRIPT_DIR / f"transcript_{int(time.time())}.jsonl"

    with open(transcript_path, "w") as f:
        for msg in messages:
            # default=str 处理不可序列化的对象（如 Pydantic 对象）
            f.write(json.dumps(msg, default=str) + "\n")

    print(f"[transcript saved: {transcript_path}]")

    # === 步骤 2: 调用 LLM 生成摘要 ===
    # 将对话历史转为文本，限制长度防止超出 token
    conversation_text = json.dumps(messages, default=str)[:80000]

    # 用一个独立的 LLM 调用来生成摘要
    # 注意：这不是 Agent 循环的一部分，只是一个单独的请求
    response = client.messages.create(
        model=MODEL,
        messages=[{
            "role": "user",
            "content": (
                "Summarize this conversation for continuity. Include: "
                "1) What was accomplished, 2) Current state, 3) Key decisions made. "
                "Be concise but preserve critical details.\n\n" + conversation_text
            )
        }],
        max_tokens=2000,
    )

    summary = response.content[0].text

    # === 步骤 3: 用摘要替换所有消息 ===
    # 新的对话只有 2 条消息：
    # 1. user: 压缩说明 + 摘要
    # 2. assistant: 确认理解
    return [
        {
            "role": "user",
            "content": f"[Conversation compressed. Transcript: {transcript_path}]\n\n{summary}"
        },
        {
            "role": "assistant",
            "content": "Understood. I have the context from the summary. Continuing."
        },
    ]


# ============================================================================
# 第六部分：工具实现函数
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
# 第七部分：工具分发映射
# ============================================================================

TOOL_HANDLERS = {
    "bash":       lambda **kw: run_bash(kw["command"]),
    "read_file":  lambda **kw: run_read(kw["path"], kw.get("limit")),
    "write_file": lambda **kw: run_write(kw["path"], kw["content"]),
    "edit_file":  lambda **kw: run_edit(kw["path"], kw["old_text"], kw["new_text"]),
    "compact":    lambda **kw: "Manual compression requested.",  # Layer 3 触发标记
}


# ============================================================================
# 第八部分：工具定义列表
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

    # 新增：compact 工具（Layer 3）
    {
        "name": "compact",
        "description": "Trigger manual conversation compression.",
        "input_schema": {
            "type": "object",
            "properties": {
                "focus": {
                    "type": "string",
                    "description": "What to preserve in the summary"
                }
            }
            # 没有 required，focus 是可选参数
        }
    },
]


# ============================================================================
# 第九部分：序列化响应内容
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
# 第十部分：Agent 循环（包含三层压缩）
# ============================================================================

def agent_loop(messages: list):
    """
    Agent 核心循环，集成三层压缩。

    循环流程：
    1. [Layer 1] micro_compact - 每轮都压缩旧的工具结果
    2. [Layer 2] auto_compact - 检查 token，超过阈值则自动压缩
    3. 调用 LLM
    4. 执行工具
    5. [Layer 3] 如果调用了 compact 工具，立即压缩

    注意 messages[:] = ... 的用法：
        这是 Python 中"原地替换列表内容"的方式
        直接 messages = ... 只会改变局部变量
        messages[:] = ... 会修改原列表的内容
    """
    while True:
        # === Layer 1: micro_compact（每轮执行）===
        # 静默压缩旧的工具结果
        micro_compact(messages)

        # === Layer 2: auto_compact（超过阈值时触发）===
        if estimate_tokens(messages) > THRESHOLD:
            print("[auto_compact triggered]")
            # 用切片赋值原地替换列表内容
            messages[:] = auto_compact(messages)

        # === 调用 LLM ===
        response = client.messages.create(
            model=MODEL, system=SYSTEM, messages=messages,
            tools=TOOLS, max_tokens=8000,
        )

        # 保存响应（序列化以避免兼容性问题）
        messages.append({"role": "assistant", "content": serialize_content(response.content)})

        if response.stop_reason != "tool_use":
            return

        # === 执行工具调用 ===
        results = []
        manual_compact = False  # Layer 3 触发标记

        for block in response.content:
            if block.type == "tool_use":
                # === 特殊处理 compact 工具（Layer 3）===
                if block.name == "compact":
                    manual_compact = True
                    output = "Compressing..."
                else:
                    # 普通工具，通过分发映射执行
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

        # === Layer 3: 手动压缩（模型触发）===
        if manual_compact:
            print("[manual compact]")
            messages[:] = auto_compact(messages)


# ============================================================================
# 第十一部分：主程序入口
# ============================================================================

if __name__ == "__main__":
    """
    交互式命令行界面。

    使用示例：
        s06 >> 读取所有 Python 文件的内容并分析依赖关系

    观察：
        1. 执行多轮工具调用后，旧的工具结果会被压缩
        2. 如果 token 超过 50000，会自动触发压缩
        3. 压缩后对话历史会保存到 .transcripts/ 目录

    测试压缩：
        - 连续执行多个读取文件的操作
        - 观察 micro_compact 的效果（旧结果变成占位符）
        - 或者直接让模型调用 compact 工具
    """
    history = []

    print("=" * 60)
    print("s06_context_compact.py - 上下文压缩示例")
    print("新增功能: 三层压缩管道（micro_compact, auto_compact, compact tool）")
    print(f"阈值: {THRESHOLD} tokens")
    print("输入 'q' 或 'exit' 退出")
    print("=" * 60)

    while True:
        try:
            query = input("\033[36ms06 >> \033[0m")
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
