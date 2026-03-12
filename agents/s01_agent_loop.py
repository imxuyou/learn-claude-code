#!/usr/bin/env python3
"""
s01_agent_loop.py - The Agent Loop (AI 智能体循环)

=== 写给新手的说明 ===

什么是 AI Agent（智能体）？
- 普通的 LLM 聊天：用户问 → 模型答 → 结束
- AI Agent：用户问 → 模型思考 → 调用工具 → 获取结果 → 继续思考 → ... → 最终回答

AI Agent 的核心秘密就是一个 while 循环：

    while stop_reason == "tool_use":   # 只要模型还想用工具
        response = LLM(messages, tools) # 调用大模型
        execute tools                    # 执行工具（如运行命令）
        append results                   # 把结果加入对话历史

架构图：
    +----------+      +-------+      +---------+
    |   User   | ---> |  LLM  | ---> |  Tool   |
    |  prompt  |      |       |      | execute |
    +----------+      +---+---+      +----+----+
                          ^               |
                          |   tool_result |
                          +---------------+
                          (循环继续直到模型决定停止)

这就是核心循环：将工具执行结果反馈给模型，直到模型决定停止。
生产级的 Agent 会在此基础上添加策略、钩子和生命周期控制。
"""

# ============================================================================
# 第一部分：导入依赖
# ============================================================================

import os           # 操作系统相关功能，如获取环境变量、当前目录等
import subprocess   # 用于执行系统命令（如 ls、python 等）

from anthropic import Anthropic      # Anthropic 官方 Python SDK，用于调用 Claude API
from dotenv import load_dotenv       # 从 .env 文件加载环境变量（如 API 密钥）

# ============================================================================
# 第二部分：初始化配置
# ============================================================================

# 加载 .env 文件中的环境变量
# override=True 表示 .env 中的值会覆盖已存在的环境变量
load_dotenv(override=True)

# 兼容性处理：如果使用自定义 API 地址，移除可能冲突的认证 token
# 这是为了支持第三方兼容 Anthropic API 的服务（如 MiniMax、Moonshot 等）
if os.getenv("ANTHROPIC_BASE_URL"):
    os.environ.pop("ANTHROPIC_AUTH_TOKEN", None)

# 创建 Anthropic 客户端实例
# base_url 参数允许指向不同的 API 端点（默认是 Anthropic 官方）
client = Anthropic(base_url=os.getenv("ANTHROPIC_BASE_URL"))

# 从环境变量获取要使用的模型 ID
# 例如：claude-sonnet-4-6, claude-opus-4-5-20251101 等
MODEL = os.environ["MODEL_ID"]

# ============================================================================
# 第三部分：系统提示词（System Prompt）
# ============================================================================

# System Prompt 是给模型的"角色设定"和"行为指南"
# 它会影响模型的整体行为风格，但不会出现在对话历史中
#
# 这里的提示词很简洁：
# 1. 告诉模型它是一个编程助手
# 2. 告诉它当前工作目录（这样模型知道文件路径的上下文）
# 3. "Act, don't explain" - 要求模型直接执行，而不是空谈
SYSTEM = f"You are a coding agent at {os.getcwd()}. Use bash to solve tasks. Act, don't explain."

# ============================================================================
# 第四部分：工具定义（Tools Definition）
# ============================================================================

# 工具定义告诉模型"你可以用什么工具"以及"如何使用它们"
#
# Anthropic 的 Tool Use 格式要求：
# - name: 工具名称，模型会用这个名字来调用
# - description: 工具描述，帮助模型理解何时使用这个工具
# - input_schema: JSON Schema 格式，定义工具接受的参数
#
# 这里我们只定义了一个 bash 工具，它可以执行任意 shell 命令
TOOLS = [{
    "name": "bash",                              # 工具名称
    "description": "Run a shell command.",       # 工具描述
    "input_schema": {                            # 参数定义（JSON Schema 格式）
        "type": "object",                        # 参数是一个对象
        "properties": {                          # 对象的属性
            "command": {"type": "string"}        # command 参数，类型是字符串
        },
        "required": ["command"],                 # command 是必需参数
    },
}]


# ============================================================================
# 第五部分：工具执行函数
# ============================================================================

def run_bash(command: str) -> str:
    """
    执行 bash 命令并返回输出结果。

    这是"工具"的实际实现。当模型决定调用 bash 工具时，
    我们就会调用这个函数来真正执行命令。

    Args:
        command: 要执行的 shell 命令，如 "ls -la" 或 "python script.py"

    Returns:
        命令的输出结果（stdout + stderr），或错误信息

    安全考虑：
        生产环境中需要更严格的安全控制，这里只是基础示例
    """

    # === 安全检查 ===
    # 阻止一些明显危险的命令
    # 注意：这只是演示级别的安全措施，生产环境需要更完善的沙箱机制
    dangerous = ["rm -rf /", "sudo", "shutdown", "reboot", "> /dev/"]
    if any(d in command for d in dangerous):
        return "Error: Dangerous command blocked"

    try:
        # === 执行命令 ===
        # subprocess.run() 是 Python 执行外部命令的推荐方式
        r = subprocess.run(
            command,              # 要执行的命令
            shell=True,           # 通过 shell 执行（支持管道、通配符等）
            cwd=os.getcwd(),      # 工作目录设为当前目录
            capture_output=True,  # 捕获 stdout 和 stderr
            text=True,            # 以文本模式返回（而非字节）
            timeout=120           # 超时时间：120秒
        )

        # 合并标准输出和错误输出
        out = (r.stdout + r.stderr).strip()

        # 限制输出长度，防止超长输出消耗过多 token
        # LLM 的上下文窗口是有限的，需要控制输入长度
        return out[:50000] if out else "(no output)"

    except subprocess.TimeoutExpired:
        # 命令执行超时
        return "Error: Timeout (120s)"


# ============================================================================
# 第六部分：序列化响应内容（关键！解决 SDK 兼容性问题）
# ============================================================================

def serialize_content(content):
    """
    将 API 响应的 content 转换为可序列化的字典格式。

    为什么需要这个函数？
    ========================
    Anthropic SDK 返回的 response.content 是 Pydantic 对象列表，
    包含 TextBlock、ToolUseBlock 等类型。当我们把它们放回 messages
    再次调用 API 时，可能会遇到兼容性问题。

    重要警告：不能使用 model_dump()！
    ================================
    model_dump() 输出的格式与 API 期望的格式不同：

        model_dump() 输出:
        {"id": "toolu_xxx", "caller": {...}, "input": {...}}

        API 期望的格式:
        {"type": "tool_use", "id": "toolu_xxx", "name": "bash", "input": {...}}

    必须手动提取必要字段，确保格式正确。

    Args:
        content: API 响应的 content 字段（Pydantic 对象列表）

    Returns:
        可序列化的字典列表
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
# 第七部分：核心 Agent 循环（最重要的部分！）
# ============================================================================

def agent_loop(messages: list):
    """
    AI Agent 的核心循环。

    这个函数实现了 Agent 的"思考-行动-观察"循环：
    1. 将对话历史发送给 LLM
    2. LLM 返回响应（可能包含工具调用）
    3. 如果 LLM 想调用工具，执行工具并将结果加入对话
    4. 重复以上步骤，直到 LLM 决定停止

    Args:
        messages: 对话历史列表，格式为：
            [
                {"role": "user", "content": "用户消息"},
                {"role": "assistant", "content": [...]},
                {"role": "user", "content": [工具结果]},
                ...
            ]

    关键概念 - stop_reason（停止原因）：
        - "tool_use": 模型想要调用工具 → 继续循环
        - "end_turn": 模型认为任务完成 → 退出循环
        - "max_tokens": 达到 token 上限 → 退出循环

    注意：messages 是引用传递，函数会直接修改这个列表
    """

    while True:  # 无限循环，直到模型决定停止

        # === 步骤 1: 调用 LLM API ===
        # 将对话历史、系统提示词、可用工具一起发送给模型
        response = client.messages.create(
            model=MODEL,          # 使用的模型（如 claude-sonnet-4-6）
            system=SYSTEM,        # 系统提示词（角色设定）
            messages=messages,    # 对话历史
            tools=TOOLS,          # 可用的工具列表
            max_tokens=8000,      # 最大输出 token 数
        )

        # === 步骤 2: 保存模型的响应 ===
        # 必须序列化！直接使用 response.content 会导致兼容性问题
        messages.append({"role": "assistant", "content": serialize_content(response.content)})

        # === 步骤 3: 检查是否应该停止 ===
        # stop_reason 是模型自己决定的！
        # 如果模型不想调用工具（即 stop_reason != "tool_use"），说明任务完成
        if response.stop_reason != "tool_use":
            return  # 退出循环，Agent 任务结束

        # === 步骤 4: 执行工具调用 ===
        # 如果代码执行到这里，说明模型想要调用工具
        results = []  # 用于收集所有工具的执行结果

        # 遍历响应中的每个内容块
        for block in response.content:
            # 只处理工具调用类型的块
            if block.type == "tool_use":
                # 打印正在执行的命令（黄色显示）
                # \033[33m 是 ANSI 转义码，表示黄色
                print(f"\033[33m$ {block.input['command']}\033[0m")

                # 实际执行命令
                output = run_bash(block.input["command"])

                # 打印输出的前 200 个字符（避免刷屏）
                print(output[:200])

                # 构造工具结果，格式是 Anthropic API 规定的
                # tool_use_id 用于关联请求和响应
                results.append({
                    "type": "tool_result",
                    "tool_use_id": block.id,
                    "content": output
                })

        # === 步骤 5: 将工具结果加入对话历史 ===
        # 工具结果作为 "user" 角色添加（这是 Anthropic API 的约定）
        # 这样在下一轮循环中，模型就能看到工具的执行结果
        messages.append({"role": "user", "content": results})

        # 循环继续：回到 while True，再次调用模型
        # 模型会基于新的工具结果，决定下一步行动


# ============================================================================
# 第八部分：主程序入口（交互式命令行）
# ============================================================================

if __name__ == "__main__":
    """
    当直接运行这个脚本时（python s01_agent_loop.py），执行以下代码。
    提供一个简单的命令行交互界面。
    """

    # 对话历史列表，会在整个会话中累积
    # 这使得模型能够"记住"之前的对话
    history = []

    print("=" * 60)
    print("s01_agent_loop.py - 最小 Agent 循环示例")
    print("可用工具: bash")
    print("输入 'q' 或 'exit' 退出")
    print("=" * 60)

    # 主交互循环
    while True:
        try:
            # 显示提示符，等待用户输入
            # \033[36m 是青色，\033[0m 重置颜色
            query = input("\033[36ms01 >> \033[0m")
        except (EOFError, KeyboardInterrupt):
            # 处理 Ctrl+D (EOF) 或 Ctrl+C (中断)
            break

        # 检查退出命令
        if query.strip().lower() in ("q", "exit", ""):
            break

        # 将用户输入添加到对话历史
        history.append({"role": "user", "content": query})

        # 运行 Agent 循环！
        # 这个调用可能会执行多轮工具调用，直到模型完成任务
        agent_loop(history)

        # 打印模型的最终文字回复
        # history[-1] 是最后一条消息（应该是 assistant 的回复）
        response_content = history[-1]["content"]
        if isinstance(response_content, list):
            for block in response_content:
                # 处理序列化后的字典格式
                if isinstance(block, dict) and block.get("type") == "text":
                    print(block["text"])

        print()  # 打印空行，美化输出
