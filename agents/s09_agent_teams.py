#!/usr/bin/env python3
"""
s09_agent_teams.py - Agent Teams (多 Agent 协作团队)

=== 本课要点 ===

引入"Agent 团队"概念：多个 Agent 可以像真实团队一样协作。

核心洞察：
    "队友之间可以互相对话。"

=== 生活中的比喻 ===

想象一个软件开发团队：
    - 项目经理（Lead）：分配任务、协调工作
    - 前端开发（Alice）：负责 UI 实现
    - 后端开发（Bob）：负责 API 开发
    - 测试工程师（Carol）：负责测试

他们通过邮件/消息互相沟通：
    Lead → Alice: "请实现登录页面"
    Alice → Lead: "登录页面完成了"
    Lead → Bob: "Alice 完成了前端，你可以开始 API 了"
    Bob → Carol: "API 写好了，帮忙测试一下"

s09 就是把这个模式用 Agent 实现！

=== 与 s04 子代理的对比 ===

    s04 子代理（Subagent）：
    ┌─────────────────────────────────────────────┐
    │  父代理: "分析这个项目"                       │
    │     │                                       │
    │     └──→ 子代理: 执行任务                    │
    │              │                              │
    │              └──→ 返回总结 → 销毁            │
    │                                             │
    │  特点：用完就扔，一次性的                     │
    └─────────────────────────────────────────────┘

    s09 队友（Teammate）：
    ┌─────────────────────────────────────────────┐
    │  Lead: "Alice，实现登录功能"                 │
    │     │                                       │
    │     └──→ Alice: 开始工作...                 │
    │              │                              │
    │              └──→ 完成 → 变成 idle（待命）   │
    │                                             │
    │  Lead: "Alice，再加个注册功能"               │
    │     │                                       │
    │     └──→ Alice: 又开始工作...（复用！）      │
    │                                             │
    │  特点：持久存在，可以反复调用                 │
    └─────────────────────────────────────────────┘

=== 通信机制：收件箱（Inbox）===

每个队友有自己的"收件箱"（JSONL 文件）：

    .team/inbox/
    ├── alice.jsonl    ← Alice 的收件箱
    ├── bob.jsonl      ← Bob 的收件箱
    └── lead.jsonl     ← Lead 的收件箱

发消息就是往文件里追加一行：
    send_message("alice", "请修复 bug")
    → 打开 alice.jsonl，追加 {"from":"lead", "content":"请修复 bug", ...}

读消息就是读取并清空文件：
    read_inbox("alice")
    → 读取 alice.jsonl 的所有行
    → 清空文件（表示已读）
    → 返回消息列表

=== 架构图 ===

    .team/config.json                    .team/inbox/
    +----------------------------+       +------------------+
    | {"team_name": "default",   |       | alice.jsonl      |
    |  "members": [              |       | bob.jsonl        |
    |    {"name":"alice",        |       | lead.jsonl       |
    |     "role":"coder",        |       +------------------+
    |     "status":"idle"}       |
    |  ]}                        |       Lead 发消息给 Alice:
    +----------------------------+         → 追加到 alice.jsonl

    spawn_teammate("alice", "coder", "...")
         │
         v
    线程: Alice                   线程: Bob
    +------------------+         +------------------+
    | agent_loop       |         | agent_loop       |
    | status: working  |         | status: idle     |
    | ... 执行工具 ... |         | ... 等待消息 ... |
    | status → idle    |         |                  |
    +------------------+         +------------------+

=== 消息类型 ===

    +-------------------------+-----------------------------------+
    | message                 | 普通文本消息                       |
    | broadcast               | 群发给所有队友                     |
    | shutdown_request        | 请求关闭（s10 扩展）               |
    | shutdown_response       | 同意/拒绝关闭（s10 扩展）          |
    | plan_approval_response  | 同意/拒绝计划（s10 扩展）          |
    +-------------------------+-----------------------------------+
"""

# ============================================================================
# 第一部分：导入依赖
# ============================================================================

import json
import os
import subprocess
import threading   # 多线程，每个队友一个线程
import time        # 时间戳
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

# 团队配置和收件箱目录
TEAM_DIR = WORKDIR / ".team"
INBOX_DIR = TEAM_DIR / "inbox"

# Lead 的系统提示词
SYSTEM = f"You are a team lead at {WORKDIR}. Spawn teammates and communicate via inboxes."

# 有效的消息类型
VALID_MSG_TYPES = {
    "message",               # 普通消息
    "broadcast",             # 广播消息
    "shutdown_request",      # 关闭请求（s10 用）
    "shutdown_response",     # 关闭响应（s10 用）
    "plan_approval_response",# 计划审批（s10 用）
}


# ============================================================================
# 第三部分：MessageBus 类（消息总线）
# ============================================================================

class MessageBus:
    """
    消息总线：管理所有队友的收件箱。

    核心思想：
        每个队友有一个 JSONL 文件作为"收件箱"
        发消息 = 往收件箱文件追加一行
        读消息 = 读取并清空收件箱

    为什么用 JSONL 文件？
        1. 简单：无需数据库或消息队列
        2. 持久：程序重启后消息还在
        3. 可读：人可以直接查看文件
        4. 并发安全：追加操作是原子的

    JSONL 格式示例（alice.jsonl）：
        {"type":"message","from":"lead","content":"请修复登录bug","timestamp":1234567890}
        {"type":"message","from":"bob","content":"API已经准备好了","timestamp":1234567891}
    """

    def __init__(self, inbox_dir: Path):
        """
        初始化消息总线。

        Args:
            inbox_dir: 收件箱目录路径
        """
        self.dir = inbox_dir
        # 确保目录存在
        self.dir.mkdir(parents=True, exist_ok=True)

    def send(self, sender: str, to: str, content: str,
             msg_type: str = "message", extra: dict = None) -> str:
        """
        发送消息给指定队友。

        这就像发邮件：
            - sender: 发件人
            - to: 收件人
            - content: 邮件内容
            - msg_type: 邮件类型（普通、广播等）

        实现原理：
            打开收件人的 .jsonl 文件，追加一行 JSON

        Args:
            sender: 发送者名称
            to: 接收者名称
            content: 消息内容
            msg_type: 消息类型（默认 "message"）
            extra: 额外的字段（可选）

        Returns:
            发送确认消息
        """
        # 验证消息类型
        if msg_type not in VALID_MSG_TYPES:
            return f"Error: Invalid type '{msg_type}'. Valid: {VALID_MSG_TYPES}"

        # 构造消息对象
        msg = {
            "type": msg_type,
            "from": sender,
            "content": content,
            "timestamp": time.time(),
        }

        # 合并额外字段
        if extra:
            msg.update(extra)

        # 追加到收件人的收件箱
        inbox_path = self.dir / f"{to}.jsonl"
        with open(inbox_path, "a") as f:
            f.write(json.dumps(msg) + "\n")

        return f"Sent {msg_type} to {to}"

    def read_inbox(self, name: str) -> list:
        """
        读取并清空指定队友的收件箱。

        这就像查看邮箱：
            - 读取所有未读邮件
            - 标记为已读（通过清空文件实现）

        为什么要清空？
            - 防止同一消息被重复处理
            - 类似于"已读即焚"

        Args:
            name: 队友名称

        Returns:
            消息列表（清空后返回）
        """
        inbox_path = self.dir / f"{name}.jsonl"

        if not inbox_path.exists():
            return []

        # 读取所有消息
        messages = []
        for line in inbox_path.read_text().strip().splitlines():
            if line:
                messages.append(json.loads(line))

        # 清空收件箱（表示已读）
        inbox_path.write_text("")

        return messages

    def broadcast(self, sender: str, content: str, teammates: list) -> str:
        """
        广播消息给所有队友。

        这就像群发邮件：
            - 遍历所有队友
            - 每人发一份（除了自己）

        Args:
            sender: 发送者
            content: 消息内容
            teammates: 队友名称列表

        Returns:
            广播确认消息
        """
        count = 0
        for name in teammates:
            if name != sender:  # 不发给自己
                self.send(sender, name, content, "broadcast")
                count += 1
        return f"Broadcast to {count} teammates"


# ============================================================================
# 第四部分：创建全局消息总线
# ============================================================================

BUS = MessageBus(INBOX_DIR)


# ============================================================================
# 第五部分：TeammateManager 类（队友管理器）
# ============================================================================

class TeammateManager:
    """
    队友管理器：创建和管理持久化的 Agent 队友。

    核心概念：
        - 每个队友是一个独立的线程
        - 每个队友有自己的 Agent 循环
        - 队友通过收件箱通信
        - 队友可以反复使用（idle → working → idle）

    配置文件结构（config.json）：
    {
        "team_name": "default",
        "members": [
            {"name": "alice", "role": "frontend", "status": "idle"},
            {"name": "bob", "role": "backend", "status": "working"}
        ]
    }

    状态流转：
        spawn() → working → 任务完成 → idle → 再次 spawn() → working → ...
    """

    def __init__(self, team_dir: Path):
        """
        初始化队友管理器。

        Args:
            team_dir: 团队配置目录
        """
        self.dir = team_dir
        self.dir.mkdir(exist_ok=True)

        # 配置文件路径
        self.config_path = self.dir / "config.json"
        # 加载或创建配置
        self.config = self._load_config()
        # 线程字典: {name: Thread}
        self.threads = {}

    def _load_config(self) -> dict:
        """加载团队配置，不存在则创建默认配置"""
        if self.config_path.exists():
            return json.loads(self.config_path.read_text())
        return {"team_name": "default", "members": []}

    def _save_config(self):
        """保存团队配置到文件"""
        self.config_path.write_text(json.dumps(self.config, indent=2))

    def _find_member(self, name: str) -> dict:
        """根据名称查找队友配置"""
        for m in self.config["members"]:
            if m["name"] == name:
                return m
        return None

    # -------------------------------------------------------------------------
    # 核心方法：spawn（创建或激活队友）
    # -------------------------------------------------------------------------

    def spawn(self, name: str, role: str, prompt: str) -> str:
        """
        创建或激活一个队友。

        这是 Agent 团队的核心操作！

        工作流程：
        1. 如果队友已存在且 idle → 激活它
        2. 如果队友不存在 → 创建新队友
        3. 启动一个新线程运行队友的 Agent 循环

        与 s04 子代理的区别：
        - 子代理：用完就销毁
        - 队友：任务完成后变成 idle，可以再次激活

        Args:
            name: 队友名称（如 "alice"）
            role: 队友角色（如 "frontend developer"）
            prompt: 初始任务提示

        Returns:
            创建/激活确认消息

        示例：
            spawn("alice", "frontend", "请实现用户登录页面")
            → 创建 Alice 线程
            → Alice 开始执行任务
            → 任务完成后 Alice 变成 idle
            → 以后可以再次 spawn("alice", ...) 给她新任务
        """
        member = self._find_member(name)

        if member:
            # 队友已存在
            if member["status"] not in ("idle", "shutdown"):
                return f"Error: '{name}' is currently {member['status']}"
            # 激活 idle 的队友
            member["status"] = "working"
            member["role"] = role
        else:
            # 创建新队友
            member = {"name": name, "role": role, "status": "working"}
            self.config["members"].append(member)

        self._save_config()

        # 启动队友线程
        thread = threading.Thread(
            target=self._teammate_loop,  # 线程执行的函数
            args=(name, role, prompt),   # 传递给函数的参数
            daemon=True,                 # 守护线程
        )
        self.threads[name] = thread
        thread.start()

        return f"Spawned '{name}' (role: {role})"

    # -------------------------------------------------------------------------
    # 队友的 Agent 循环
    # -------------------------------------------------------------------------

    def _teammate_loop(self, name: str, role: str, prompt: str):
        """
        队友的独立 Agent 循环。

        这个方法在队友的线程中运行，流程：
        1. 设置队友专属的系统提示词
        2. 开始 Agent 循环
        3. 每轮检查收件箱，处理新消息
        4. 执行工具调用
        5. 任务完成后变成 idle

        注意：
        - 最多 50 轮，防止无限循环
        - 每轮都会检查收件箱
        - 队友之间通过消息通信

        Args:
            name: 队友名称
            role: 队友角色
            prompt: 初始任务提示
        """
        # 队友专属的系统提示词
        sys_prompt = (
            f"You are '{name}', role: {role}, at {WORKDIR}. "
            f"Use send_message to communicate. Complete your task."
        )

        # 初始消息：用户给的任务
        messages = [{"role": "user", "content": prompt}]

        # 队友可用的工具
        tools = self._teammate_tools()

        # Agent 循环（最多 50 轮）
        for _ in range(50):
            # === 关键：每轮检查收件箱 ===
            inbox = BUS.read_inbox(name)
            for msg in inbox:
                # 把收到的消息注入到对话中
                messages.append({"role": "user", "content": json.dumps(msg)})

            # 调用 LLM
            try:
                response = client.messages.create(
                    model=MODEL,
                    system=sys_prompt,
                    messages=messages,
                    tools=tools,
                    max_tokens=8000,
                )
            except Exception:
                break

            messages.append({"role": "assistant", "content": response.content})

            # 任务完成？
            if response.stop_reason != "tool_use":
                break

            # 执行工具调用
            results = []
            for block in response.content:
                if block.type == "tool_use":
                    output = self._exec(name, block.name, block.input)
                    # 打印日志（带队友名称标识）
                    print(f"  [{name}] {block.name}: {str(output)[:120]}")
                    results.append({
                        "type": "tool_result",
                        "tool_use_id": block.id,
                        "content": str(output),
                    })

            messages.append({"role": "user", "content": results})

        # === 任务完成，变成 idle ===
        member = self._find_member(name)
        if member and member["status"] != "shutdown":
            member["status"] = "idle"
            self._save_config()

    def _exec(self, sender: str, tool_name: str, args: dict) -> str:
        """
        执行队友的工具调用。

        注意：sender 参数用于标识消息的发送者
        当队友调用 send_message 时，sender 就是队友的名字

        Args:
            sender: 调用者名称（队友名）
            tool_name: 工具名称
            args: 工具参数

        Returns:
            工具执行结果
        """
        # 基础工具
        if tool_name == "bash":
            return _run_bash(args["command"])
        if tool_name == "read_file":
            return _run_read(args["path"])
        if tool_name == "write_file":
            return _run_write(args["path"], args["content"])
        if tool_name == "edit_file":
            return _run_edit(args["path"], args["old_text"], args["new_text"])

        # 通信工具
        if tool_name == "send_message":
            # sender 是队友自己的名字
            return BUS.send(sender, args["to"], args["content"],
                           args.get("msg_type", "message"))
        if tool_name == "read_inbox":
            return json.dumps(BUS.read_inbox(sender), indent=2)

        return f"Unknown tool: {tool_name}"

    def _teammate_tools(self) -> list:
        """
        队友可用的工具列表。

        与 Lead 的区别：
        - 队友没有 spawn_teammate（不能创建新队友）
        - 队友没有 broadcast（不能群发）
        - 队友没有 list_teammates（看不到团队全貌）

        这体现了权限分层：Lead 是管理者，队友是执行者
        """
        return [
            # 基础工具
            {"name": "bash", "description": "Run a shell command.",
             "input_schema": {"type": "object", "properties": {"command": {"type": "string"}}, "required": ["command"]}},
            {"name": "read_file", "description": "Read file contents.",
             "input_schema": {"type": "object", "properties": {"path": {"type": "string"}}, "required": ["path"]}},
            {"name": "write_file", "description": "Write content to file.",
             "input_schema": {"type": "object", "properties": {"path": {"type": "string"}, "content": {"type": "string"}}, "required": ["path", "content"]}},
            {"name": "edit_file", "description": "Replace exact text in file.",
             "input_schema": {"type": "object", "properties": {"path": {"type": "string"}, "old_text": {"type": "string"}, "new_text": {"type": "string"}}, "required": ["path", "old_text", "new_text"]}},

            # 通信工具（队友版本）
            {"name": "send_message", "description": "Send message to a teammate.",
             "input_schema": {"type": "object", "properties": {"to": {"type": "string"}, "content": {"type": "string"}, "msg_type": {"type": "string", "enum": list(VALID_MSG_TYPES)}}, "required": ["to", "content"]}},
            {"name": "read_inbox", "description": "Read and drain your inbox.",
             "input_schema": {"type": "object", "properties": {}}},
        ]

    def list_all(self) -> str:
        """列出所有队友的状态"""
        if not self.config["members"]:
            return "No teammates."
        lines = [f"Team: {self.config['team_name']}"]
        for m in self.config["members"]:
            lines.append(f"  {m['name']} ({m['role']}): {m['status']}")
        return "\n".join(lines)

    def member_names(self) -> list:
        """获取所有队友的名称列表"""
        return [m["name"] for m in self.config["members"]]


# ============================================================================
# 第六部分：创建全局队友管理器
# ============================================================================

TEAM = TeammateManager(TEAM_DIR)


# ============================================================================
# 第七部分：基础工具实现函数
# ============================================================================

def _safe_path(p: str) -> Path:
    """路径安全检查"""
    path = (WORKDIR / p).resolve()
    if not path.is_relative_to(WORKDIR):
        raise ValueError(f"Path escapes workspace: {p}")
    return path


def _run_bash(command: str) -> str:
    """执行 bash 命令"""
    dangerous = ["rm -rf /", "sudo", "shutdown", "reboot"]
    if any(d in command for d in dangerous):
        return "Error: Dangerous command blocked"
    try:
        r = subprocess.run(
            command, shell=True, cwd=WORKDIR,
            capture_output=True, text=True, timeout=120,
        )
        out = (r.stdout + r.stderr).strip()
        return out[:50000] if out else "(no output)"
    except subprocess.TimeoutExpired:
        return "Error: Timeout (120s)"


def _run_read(path: str, limit: int = None) -> str:
    """读取文件"""
    try:
        lines = _safe_path(path).read_text().splitlines()
        if limit and limit < len(lines):
            lines = lines[:limit] + [f"... ({len(lines) - limit} more)"]
        return "\n".join(lines)[:50000]
    except Exception as e:
        return f"Error: {e}"


def _run_write(path: str, content: str) -> str:
    """写入文件"""
    try:
        fp = _safe_path(path)
        fp.parent.mkdir(parents=True, exist_ok=True)
        fp.write_text(content)
        return f"Wrote {len(content)} bytes"
    except Exception as e:
        return f"Error: {e}"


def _run_edit(path: str, old_text: str, new_text: str) -> str:
    """编辑文件"""
    try:
        fp = _safe_path(path)
        c = fp.read_text()
        if old_text not in c:
            return f"Error: Text not found in {path}"
        fp.write_text(c.replace(old_text, new_text, 1))
        return f"Edited {path}"
    except Exception as e:
        return f"Error: {e}"


# ============================================================================
# 第八部分：Lead 的工具分发映射
# ============================================================================

# Lead 有 9 个工具，比队友多 3 个（spawn、list、broadcast）
TOOL_HANDLERS = {
    # 基础工具（与队友相同）
    "bash":            lambda **kw: _run_bash(kw["command"]),
    "read_file":       lambda **kw: _run_read(kw["path"], kw.get("limit")),
    "write_file":      lambda **kw: _run_write(kw["path"], kw["content"]),
    "edit_file":       lambda **kw: _run_edit(kw["path"], kw["old_text"], kw["new_text"]),

    # 团队管理工具（Lead 专属）
    "spawn_teammate":  lambda **kw: TEAM.spawn(kw["name"], kw["role"], kw["prompt"]),
    "list_teammates":  lambda **kw: TEAM.list_all(),

    # 通信工具（Lead 版本，sender 固定为 "lead"）
    "send_message":    lambda **kw: BUS.send("lead", kw["to"], kw["content"], kw.get("msg_type", "message")),
    "read_inbox":      lambda **kw: json.dumps(BUS.read_inbox("lead"), indent=2),
    "broadcast":       lambda **kw: BUS.broadcast("lead", kw["content"], TEAM.member_names()),
}


# ============================================================================
# 第九部分：Lead 的工具定义列表
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

    # 团队管理工具（Lead 专属）
    {
        "name": "spawn_teammate",
        "description": "Spawn a persistent teammate that runs in its own thread.",
        "input_schema": {
            "type": "object",
            "properties": {
                "name": {"type": "string"},
                "role": {"type": "string"},
                "prompt": {"type": "string"}
            },
            "required": ["name", "role", "prompt"]
        }
    },
    {
        "name": "list_teammates",
        "description": "List all teammates with name, role, status.",
        "input_schema": {"type": "object", "properties": {}}
    },

    # 通信工具
    {
        "name": "send_message",
        "description": "Send a message to a teammate's inbox.",
        "input_schema": {
            "type": "object",
            "properties": {
                "to": {"type": "string"},
                "content": {"type": "string"},
                "msg_type": {"type": "string", "enum": list(VALID_MSG_TYPES)}
            },
            "required": ["to", "content"]
        }
    },
    {
        "name": "read_inbox",
        "description": "Read and drain the lead's inbox.",
        "input_schema": {"type": "object", "properties": {}}
    },
    {
        "name": "broadcast",
        "description": "Send a message to all teammates.",
        "input_schema": {
            "type": "object",
            "properties": {"content": {"type": "string"}},
            "required": ["content"]
        }
    },
]


# ============================================================================
# 第十部分：序列化响应内容
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
# 第十一部分：Lead 的 Agent 循环
# ============================================================================

def agent_loop(messages: list):
    """
    Lead 的 Agent 循环。

    与队友循环的区别：
    - Lead 是主线程
    - Lead 有更多工具（spawn、broadcast 等）
    - Lead 也会检查收件箱（队友可能给 Lead 发消息）

    典型协作流程：
    1. 用户: "开发一个登录系统"
    2. Lead: spawn_teammate("alice", "frontend", "实现登录页面")
    3. Lead: spawn_teammate("bob", "backend", "实现登录API")
    4. Alice 和 Bob 各自在自己的线程中工作
    5. Alice: send_message("lead", "登录页面完成了")
    6. Lead 下次循环时收到消息
    7. Lead: send_message("bob", "前端完成了，你可以开始集成了")
    8. ...
    """
    while True:
        # === Lead 也检查自己的收件箱 ===
        inbox = BUS.read_inbox("lead")
        if inbox:
            # 把收到的消息注入到对话中
            messages.append({
                "role": "user",
                "content": f"<inbox>{json.dumps(inbox, indent=2)}</inbox>",
            })
            messages.append({
                "role": "assistant",
                "content": "Noted inbox messages.",
            })

        # 调用 LLM
        response = client.messages.create(
            model=MODEL,
            system=SYSTEM,
            messages=messages,
            tools=TOOLS,
            max_tokens=8000,
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
                    "content": str(output),
                })

        messages.append({"role": "user", "content": results})


# ============================================================================
# 第十二部分：主程序入口
# ============================================================================

if __name__ == "__main__":
    """
    交互式命令行界面。

    使用示例：
        s09 >> 组建一个团队来开发一个简单的待办事项应用

    观察：
        1. Lead 会创建多个队友（如 alice, bob）
        2. 每个队友在自己的线程中独立工作
        3. 队友之间通过消息互相协作
        4. 查看 .team/inbox/ 可以看到消息文件

    快捷命令：
        /team  - 查看团队状态
        /inbox - 查看 Lead 的收件箱

    实际场景示例：
        s09 >> 我需要一个团队来重构这个项目的认证系统

        Lead 可能会：
        1. spawn_teammate("analyst", "architect", "分析现有认证代码")
        2. spawn_teammate("coder", "developer", "等待分析结果后重构")
        3. 协调两个队友的工作
        4. 汇总结果
    """
    history = []

    print("=" * 60)
    print("s09_agent_teams.py - 多 Agent 协作团队示例")
    print("新增工具: spawn_teammate, list_teammates, send_message,")
    print("         read_inbox, broadcast")
    print("快捷命令: /team (查看团队), /inbox (查看收件箱)")
    print("输入 'q' 或 'exit' 退出")
    print("=" * 60)

    while True:
        try:
            query = input("\033[36ms09 >> \033[0m")
        except (EOFError, KeyboardInterrupt):
            break

        if query.strip().lower() in ("q", "exit", ""):
            break

        # 快捷命令
        if query.strip() == "/team":
            print(TEAM.list_all())
            continue
        if query.strip() == "/inbox":
            print(json.dumps(BUS.read_inbox("lead"), indent=2))
            continue

        history.append({"role": "user", "content": query})
        agent_loop(history)

        # 打印模型的文字回复
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
