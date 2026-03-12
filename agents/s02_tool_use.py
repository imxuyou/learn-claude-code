#!/usr/bin/env python3
"""
s02_tool_use.py - Tools (工具扩展与分发)

=== 本课要点 ===

s01 只有一个 bash 工具，本课扩展到 4 个工具，并引入"工具分发"模式。

核心洞察：
    "Agent 循环完全没变，我只是添加了更多工具。"

这说明 s01 的循环设计是通用的，可以支持任意数量的工具。

=== s01 vs s02 对比 ===

    s01: 1 个工具 (bash)
         硬编码调用 run_bash()

    s02: 4 个工具 (bash, read_file, write_file, edit_file)
         使用分发映射 TOOL_HANDLERS 路由调用

=== 架构图 ===

    +----------+      +-------+      +------------------+
    |   User   | ---> |  LLM  | ---> | Tool Dispatch    |
    |  prompt  |      |       |      | {                |
    +----------+      +---+---+      |   bash: run_bash |
                          ^          |   read: run_read |
                          |          |   write: run_wr  |
                          +----------+   edit: run_edit |
                          tool_result| }                |
                                     +------------------+

=== 为什么需要文件操作工具？ ===

虽然 bash 工具可以用 cat/echo 完成文件读写，但专用工具有优势：
1. 更安全 - 可以实现路径沙箱，防止访问敏感目录
2. 更高效 - 直接操作内存，无需启动子进程
3. 更可控 - 可以添加细粒度的权限控制
4. 更友好 - 更清晰的错误信息和返回值
"""

# ============================================================================
# 第一部分：导入依赖
# ============================================================================

import os           # 操作系统功能
import subprocess   # 执行系统命令
from pathlib import Path  # 现代化的路径处理（比 os.path 更好用）

from anthropic import Anthropic      # Claude API 客户端
from dotenv import load_dotenv       # 环境变量加载

# ============================================================================
# 第二部分：初始化配置
# ============================================================================

load_dotenv(override=True)

if os.getenv("ANTHROPIC_BASE_URL"):
    os.environ.pop("ANTHROPIC_AUTH_TOKEN", None)

# 工作目录：所有文件操作都限制在这个目录内
# Path.cwd() 返回当前工作目录的 Path 对象
WORKDIR = Path.cwd()

client = Anthropic(base_url=os.getenv("ANTHROPIC_BASE_URL"))
MODEL = os.environ["MODEL_ID"]

# 系统提示词：注意从 "Use bash" 改为 "Use tools"（复数）
SYSTEM = f"You are a coding agent at {WORKDIR}. Use tools to solve tasks. Act, don't explain."


# ============================================================================
# 第三部分：安全辅助函数
# ============================================================================

def safe_path(p: str) -> Path:
    """
    将用户提供的路径转换为安全的绝对路径。

    这是一个重要的安全机制！防止 LLM 访问工作目录之外的文件。

    例如：
        WORKDIR = /home/user/project
        safe_path("src/main.py")     → /home/user/project/src/main.py  ✓
        safe_path("../../../etc/passwd") → 抛出异常！              ✗

    Args:
        p: 用户提供的相对或绝对路径

    Returns:
        解析后的安全路径

    Raises:
        ValueError: 如果路径试图逃逸出工作目录
    """
    # 1. 将相对路径转为基于 WORKDIR 的绝对路径
    # 2. resolve() 会处理 .. 和符号链接，得到真实路径
    path = (WORKDIR / p).resolve()

    # 检查解析后的路径是否仍在 WORKDIR 内
    # is_relative_to() 是 Python 3.9+ 的方法
    if not path.is_relative_to(WORKDIR):
        raise ValueError(f"Path escapes workspace: {p}")

    return path


# ============================================================================
# 第四部分：工具实现函数
# ============================================================================

# -----------------------------------------------------------------------------
# 工具 1: bash - 执行 shell 命令（与 s01 相同）
# -----------------------------------------------------------------------------
def run_bash(command: str) -> str:
    """
    执行 bash 命令。

    这个函数与 s01 中的实现相同，提供基础的命令执行能力。
    """
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


# -----------------------------------------------------------------------------
# 工具 2: read_file - 读取文件内容
# -----------------------------------------------------------------------------
def run_read(path: str, limit: int = None) -> str:
    """
    读取文件内容。

    为什么不用 bash 的 cat？
    1. 更安全：使用 safe_path() 确保路径在沙箱内
    2. 更可控：可以限制读取的行数，避免超大文件消耗 token
    3. 更清晰：错误信息更友好

    Args:
        path: 要读取的文件路径（相对于 WORKDIR）
        limit: 可选，最多读取多少行

    Returns:
        文件内容（可能被截断）
    """
    try:
        # safe_path() 确保路径安全
        text = safe_path(path).read_text()
        lines = text.splitlines()

        # 如果指定了行数限制，截断内容
        if limit and limit < len(lines):
            lines = lines[:limit] + [f"... ({len(lines) - limit} more lines)"]

        # 最终输出也限制在 50000 字符内（保护 token 消耗）
        return "\n".join(lines)[:50000]
    except Exception as e:
        return f"Error: {e}"


# -----------------------------------------------------------------------------
# 工具 3: write_file - 写入文件
# -----------------------------------------------------------------------------
def run_write(path: str, content: str) -> str:
    """
    写入内容到文件。

    会自动创建不存在的父目录。

    Args:
        path: 目标文件路径
        content: 要写入的内容

    Returns:
        成功/失败信息
    """
    try:
        fp = safe_path(path)

        # 自动创建父目录（如 src/utils/ 目录不存在时自动创建）
        # parents=True: 递归创建所有需要的父目录
        # exist_ok=True: 目录已存在时不报错
        fp.parent.mkdir(parents=True, exist_ok=True)

        fp.write_text(content)
        return f"Wrote {len(content)} bytes to {path}"
    except Exception as e:
        return f"Error: {e}"


# -----------------------------------------------------------------------------
# 工具 4: edit_file - 编辑文件（查找替换）
# -----------------------------------------------------------------------------
def run_edit(path: str, old_text: str, new_text: str) -> str:
    """
    在文件中查找并替换文本。

    这是比 write_file 更精细的编辑方式：
    - write_file: 覆盖整个文件
    - edit_file: 只修改匹配的部分

    为什么需要 edit_file？
    1. 对于大文件，LLM 只需要提供要修改的片段，而非整个文件
    2. 减少 token 消耗
    3. 减少出错概率（不会意外删除其他内容）

    Args:
        path: 文件路径
        old_text: 要查找的原文本（必须精确匹配）
        new_text: 替换后的新文本

    Returns:
        成功/失败信息

    注意：只替换第一个匹配项（replace(..., 1)）
    """
    try:
        fp = safe_path(path)
        content = fp.read_text()

        # 检查原文本是否存在
        if old_text not in content:
            return f"Error: Text not found in {path}"

        # 只替换第一个匹配项，避免意外的全局替换
        fp.write_text(content.replace(old_text, new_text, 1))
        return f"Edited {path}"
    except Exception as e:
        return f"Error: {e}"


# ============================================================================
# 第五部分：工具分发映射（核心设计模式）
# ============================================================================

# 这是本课的核心概念：工具分发映射
#
# 为什么需要分发映射？
# - s01 只有一个工具，可以硬编码调用
# - 当工具增多时，需要一种通用的方式来路由调用
#
# 模式：{工具名称: 处理函数}
# 当 LLM 调用某个工具时，我们通过名称查找对应的处理函数

TOOL_HANDLERS = {
    # 工具名称 → 处理函数
    # lambda **kw 用于解包 LLM 传来的参数字典
    "bash":       lambda **kw: run_bash(kw["command"]),
    "read_file":  lambda **kw: run_read(kw["path"], kw.get("limit")),
    "write_file": lambda **kw: run_write(kw["path"], kw["content"]),
    "edit_file":  lambda **kw: run_edit(kw["path"], kw["old_text"], kw["new_text"]),
}

# ============================================================================
# 第六部分：工具定义列表
# ============================================================================

# 工具定义列表：告诉 LLM 有哪些工具可用
# 这个列表会在每次 API 调用时发送给模型
#
# 每个工具定义包含：
# - name: 工具名称（与 TOOL_HANDLERS 的 key 对应）
# - description: 描述（帮助 LLM 理解何时使用）
# - input_schema: 参数定义（JSON Schema 格式）

TOOLS = [
    # 工具 1: bash
    {
        "name": "bash",
        "description": "Run a shell command.",
        "input_schema": {
            "type": "object",
            "properties": {
                "command": {"type": "string"}
            },
            "required": ["command"]
        }
    },

    # 工具 2: read_file
    {
        "name": "read_file",
        "description": "Read file contents.",
        "input_schema": {
            "type": "object",
            "properties": {
                "path": {"type": "string"},
                "limit": {"type": "integer"}  # 可选参数，不在 required 中
            },
            "required": ["path"]
        }
    },

    # 工具 3: write_file
    {
        "name": "write_file",
        "description": "Write content to file.",
        "input_schema": {
            "type": "object",
            "properties": {
                "path": {"type": "string"},
                "content": {"type": "string"}
            },
            "required": ["path", "content"]
        }
    },

    # 工具 4: edit_file
    {
        "name": "edit_file",
        "description": "Replace exact text in file.",
        "input_schema": {
            "type": "object",
            "properties": {
                "path": {"type": "string"},
                "old_text": {"type": "string"},
                "new_text": {"type": "string"}
            },
            "required": ["path", "old_text", "new_text"]
        }
    },
]


# ============================================================================
# 第七部分：序列化响应内容
# ============================================================================

def serialize_content(content):
    """
    将 API 响应的 content 转换为可序列化的字典格式。

    重要：不能使用 model_dump()！
    model_dump() 输出的格式与 API 期望的格式不同，会导致错误。
    必须手动提取必要字段。

    详见 s01 中的注释说明。
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
# 第八部分：Agent 循环
# ============================================================================

def agent_loop(messages: list):
    """
    Agent 核心循环。

    与 s01 的区别：
    - s01: 硬编码调用 run_bash()
    - s02: 通过 TOOL_HANDLERS 分发调用

    核心流程没变：
    1. 调用 LLM
    2. 检查 stop_reason
    3. 执行工具
    4. 循环
    """
    while True:
        # 调用 API（传入工具列表）
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
        for block in response.content:
            if block.type == "tool_use":
                # === 关键变化：通过分发映射查找处理函数 ===
                handler = TOOL_HANDLERS.get(block.name)

                if handler:
                    # 找到处理函数，执行它
                    # **block.input 将参数字典解包传入
                    output = handler(**block.input)
                else:
                    # 未知工具（理论上不应该发生，除非工具列表配置错误）
                    output = f"Unknown tool: {block.name}"

                # 打印执行信息（方便调试）
                print(f"> {block.name}: {output[:200]}")

                results.append({
                    "type": "tool_result",
                    "tool_use_id": block.id,
                    "content": output
                })

        messages.append({"role": "user", "content": results})


# ============================================================================
# 第九部分：主程序入口
# ============================================================================

if __name__ == "__main__":
    """
    交互式命令行界面。

    使用示例：
        s02 >> 读取 README.md 文件
        s02 >> 创建一个 hello.py 文件，内容是打印 Hello World
        s02 >> 把 hello.py 中的 Hello 改成 Hi
        s02 >> 运行 hello.py
    """
    history = []

    print("=" * 60)
    print("s02_tool_use.py - 工具扩展示例")
    print("可用工具: bash, read_file, write_file, edit_file")
    print("输入 'q' 或 'exit' 退出")
    print("=" * 60)

    while True:
        try:
            query = input("\033[36ms02 >> \033[0m")
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
