#!/usr/bin/env python3
"""
s04_subagent.py - Subagents (子代理)

=== 本课要点 ===

引入"子代理"概念：父代理可以派生一个独立的子代理来完成子任务。

核心洞察：
    "进程隔离免费提供上下文隔离。"

子代理的特点：
1. 独立的上下文 - messages=[] 从空开始
2. 共享文件系统 - 可以读写同一个工作目录
3. 只返回总结 - 父代理只收到最终结果，不关心过程

=== 为什么需要子代理？ ===

场景：用户说"分析这个项目的代码结构"

没有子代理：
    - 所有探索过程都在主对话中累积
    - 上下文迅速膨胀（读取大量文件）
    - 容易超出 token 限制
    - 主对话被"污染"，难以回顾

有子代理：
    - 派生子代理去探索
    - 子代理读取文件、分析代码
    - 子代理返回简洁的总结
    - 主对话保持干净

=== 架构图 ===

    Parent agent                     Subagent
    +------------------+             +------------------+
    | messages=[...]   |             | messages=[]      |  <-- 全新上下文
    |                  |  dispatch   |                  |
    | tool: task       | ----------> | while tool_use:  |
    |   prompt="..."   |             |   call tools     |
    |   description="" |             |   append results |
    |                  |  summary    |                  |
    |   result = "..." | <---------- | return last text |
    +------------------+             +------------------+
              |
    Parent context stays clean.     子代理上下文被丢弃
    Subagent context is discarded.

=== 工具差异 ===

    父代理工具: bash, read_file, write_file, edit_file, task
    子代理工具: bash, read_file, write_file, edit_file
                                                        ↑ 没有 task！

子代理没有 task 工具，这样可以防止无限递归派生。
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

# 父代理的系统提示词：强调可以委派任务
SYSTEM = f"You are a coding agent at {WORKDIR}. Use the task tool to delegate exploration or subtasks."

# 子代理的系统提示词：强调完成任务后要总结
SUBAGENT_SYSTEM = f"You are a coding subagent at {WORKDIR}. Complete the given task, then summarize your findings."


# ============================================================================
# 第三部分：工具实现函数（父子代理共享）
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
# 第四部分：工具分发映射（父子代理共享）
# ============================================================================

TOOL_HANDLERS = {
    "bash":       lambda **kw: run_bash(kw["command"]),
    "read_file":  lambda **kw: run_read(kw["path"], kw.get("limit")),
    "write_file": lambda **kw: run_write(kw["path"], kw["content"]),
    "edit_file":  lambda **kw: run_edit(kw["path"], kw["old_text"], kw["new_text"]),
}


# ============================================================================
# 第五部分：工具定义列表
# ============================================================================

# 子代理的工具列表：基础工具，没有 task（防止递归）
CHILD_TOOLS = [
    {"name": "bash", "description": "Run a shell command.",
     "input_schema": {"type": "object", "properties": {"command": {"type": "string"}}, "required": ["command"]}},
    {"name": "read_file", "description": "Read file contents.",
     "input_schema": {"type": "object", "properties": {"path": {"type": "string"}, "limit": {"type": "integer"}}, "required": ["path"]}},
    {"name": "write_file", "description": "Write content to file.",
     "input_schema": {"type": "object", "properties": {"path": {"type": "string"}, "content": {"type": "string"}}, "required": ["path", "content"]}},
    {"name": "edit_file", "description": "Replace exact text in file.",
     "input_schema": {"type": "object", "properties": {"path": {"type": "string"}, "old_text": {"type": "string"}, "new_text": {"type": "string"}}, "required": ["path", "old_text", "new_text"]}},
]

# 父代理的工具列表：基础工具 + task 工具
PARENT_TOOLS = CHILD_TOOLS + [
    {
        "name": "task",
        "description": "Spawn a subagent with fresh context. It shares the filesystem but not conversation history.",
        "input_schema": {
            "type": "object",
            "properties": {
                "prompt": {"type": "string"},
                "description": {"type": "string", "description": "Short description of the task"}
            },
            "required": ["prompt"]
        }
    },
]


# ============================================================================
# 第六部分：序列化响应内容
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
# 第七部分：子代理实现（本课核心）
# ============================================================================

def run_subagent(prompt: str) -> str:
    """
    运行一个子代理来完成指定任务。

    子代理的特点：
    1. 全新上下文 - sub_messages 从空列表开始
    2. 独立运行 - 有自己的 Agent 循环
    3. 有限工具 - 没有 task 工具，不能再派生子代理
    4. 只返回总结 - 父代理只收到最后的文字回复

    为什么这样设计？
    - 上下文隔离：子代理的探索过程不会污染父代理的上下文
    - 资源保护：30 轮限制防止无限循环
    - 信息压缩：复杂的探索过程被压缩成简洁的总结

    Args:
        prompt: 子代理要完成的任务描述

    Returns:
        子代理的最终总结（文字回复）

    示例：
        父代理调用:
            task(prompt="分析 src/ 目录下所有 Python 文件的依赖关系")

        子代理执行:
            1. ls src/
            2. read_file src/main.py
            3. read_file src/utils.py
            4. ... (多轮探索)
            5. 返回: "src/ 目录下有 3 个 Python 文件..."

        父代理收到:
            "src/ 目录下有 3 个 Python 文件..."
            （不包含子代理的探索过程）
    """
    # 全新的上下文！这是子代理的关键特点
    sub_messages = [{"role": "user", "content": prompt}]

    # 安全限制：最多 30 轮，防止无限循环
    for _ in range(30):
        response = client.messages.create(
            model=MODEL,
            system=SUBAGENT_SYSTEM,  # 使用子代理专用的系统提示词
            messages=sub_messages,
            tools=CHILD_TOOLS,        # 使用子代理的工具列表（没有 task）
            max_tokens=8000,
        )

        # 保存响应（序列化以避免兼容性问题）
        sub_messages.append({"role": "assistant", "content": serialize_content(response.content)})

        # 如果子代理决定停止，退出循环
        if response.stop_reason != "tool_use":
            break

        # 执行工具调用
        results = []
        for block in response.content:
            if block.type == "tool_use":
                handler = TOOL_HANDLERS.get(block.name)
                output = handler(**block.input) if handler else f"Unknown tool: {block.name}"
                results.append({
                    "type": "tool_result",
                    "tool_use_id": block.id,
                    "content": str(output)[:50000]
                })

        sub_messages.append({"role": "user", "content": results})

    # === 关键：只返回最终的文字回复 ===
    # 子代理的整个 sub_messages 历史被丢弃
    # 父代理只收到这个简洁的总结
    return "".join(b.text for b in response.content if hasattr(b, "text")) or "(no summary)"


# ============================================================================
# 第八部分：父代理循环
# ============================================================================

def agent_loop(messages: list):
    """
    父代理的核心循环。

    与 s02/s03 的区别：
    - 增加了 task 工具的处理
    - 当遇到 task 工具时，调用 run_subagent() 派生子代理

    工作流程：
    1. 用户请求 → 父代理
    2. 父代理决定是否需要委派 → 调用 task 工具
    3. 子代理执行任务 → 返回总结
    4. 父代理收到总结 → 继续工作或回复用户
    """
    while True:
        response = client.messages.create(
            model=MODEL, system=SYSTEM, messages=messages,
            tools=PARENT_TOOLS, max_tokens=8000,  # 父代理有 task 工具
        )

        # 保存响应（序列化）
        messages.append({"role": "assistant", "content": serialize_content(response.content)})

        if response.stop_reason != "tool_use":
            return

        results = []
        for block in response.content:
            if block.type == "tool_use":
                # === 特殊处理 task 工具 ===
                if block.name == "task":
                    desc = block.input.get("description", "subtask")
                    print(f"> task ({desc}): {block.input['prompt'][:80]}...")

                    # 派生子代理！
                    output = run_subagent(block.input["prompt"])
                else:
                    # 普通工具，直接执行
                    handler = TOOL_HANDLERS.get(block.name)
                    output = handler(**block.input) if handler else f"Unknown tool: {block.name}"

                print(f"  {str(output)[:200]}")
                results.append({
                    "type": "tool_result",
                    "tool_use_id": block.id,
                    "content": str(output)
                })

        messages.append({"role": "user", "content": results})


# ============================================================================
# 第九部分：主程序入口
# ============================================================================

if __name__ == "__main__":
    """
    交互式命令行界面。

    使用示例：
        s04 >> 使用 task 分析这个项目的目录结构，然后创建一个总结文档

    观察：
        1. 父代理可能会派生子代理去探索目录
        2. 子代理读取文件、分析结构
        3. 子代理返回简洁的总结
        4. 父代理基于总结创建文档
        5. 父代理的上下文保持干净（不包含子代理的探索过程）
    """
    history = []

    print("=" * 60)
    print("s04_subagent.py - 子代理示例")
    print("新增工具: task（派生子代理）")
    print("特性: 上下文隔离、只返回总结")
    print("输入 'q' 或 'exit' 退出")
    print("=" * 60)

    while True:
        try:
            query = input("\033[36ms04 >> \033[0m")
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
                if isinstance(block, dict) and block.get("type") == "text":
                    print(block["text"])

        print()
