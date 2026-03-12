#!/usr/bin/env python3
"""
s10_team_protocols.py - Team Protocols (团队协议)

=== 本课要点 ===

引入"团队协议"概念：规范化的请求-响应模式，用于关键决策。

核心洞察：
    "相同的 request_id 关联模式，应用于不同领域。"

=== 为什么需要协议？ ===

s09 的问题：
    - 队友可以随意工作，没有监督
    - 无法优雅地关闭队友
    - 重大决策没有审批流程

s10 的解决方案：
    - 关闭协议（Shutdown Protocol）：优雅关闭队友
    - 计划审批协议（Plan Approval Protocol）：重大工作需要 Lead 批准

=== 生活中的比喻 ===

想象一个公司的审批流程：

    关闭协议（像是辞职流程）：
    ┌─────────────────────────────────────────────────────────┐
    │  老板: "Alice，项目结束了，你可以下班了"                  │
    │         （发送 shutdown_request）                        │
    │                                                         │
    │  Alice: "好的，我把手头的工作收尾一下就走"                │
    │         （发送 shutdown_response: approve=true）         │
    │                                                         │
    │  或者：                                                  │
    │  Alice: "不行，我还有一个紧急任务没完成"                  │
    │         （发送 shutdown_response: approve=false）        │
    └─────────────────────────────────────────────────────────┘

    计划审批协议（像是方案评审）：
    ┌─────────────────────────────────────────────────────────┐
    │  Alice: "老板，我打算用 React 重写前端，这是我的方案..."  │
    │         （发送 plan_approval: plan="..."）               │
    │                                                         │
    │  老板: "方案 OK，可以开始"                                │
    │         （发送 plan_approval_response: approve=true）    │
    │                                                         │
    │  或者：                                                  │
    │  老板: "预算不够，换个方案"                               │
    │         （发送 plan_approval_response: approve=false）   │
    └─────────────────────────────────────────────────────────┘

=== request_id 关联模式 ===

问题：异步通信中，如何知道响应对应哪个请求？

    时间线：
    Lead ──[请求A]──[请求B]──────────────[收到响应X]──
                                              │
                                              └─ 这是 A 的响应还是 B 的？

解决：每个请求带唯一 ID，响应时返回相同 ID

    Lead ──[请求A: id=abc]──[请求B: id=xyz]────[响应: id=abc]──
                                                     │
                                                     └─ 是 A 的响应！

=== 协议状态机 ===

    关闭协议状态机：
    ┌─────────┐  shutdown_request   ┌─────────┐
    │ (none)  │ ─────────────────→  │ pending │
    └─────────┘                     └────┬────┘
                                         │
                     ┌───────────────────┼───────────────────┐
                     │                   │                   │
                     ▼                   │                   ▼
               ┌──────────┐              │            ┌──────────┐
               │ approved │              │            │ rejected │
               └──────────┘              │            └──────────┘
                     │                   │
                     ▼                   │
              队友线程退出               │
                                        │
    计划审批状态机：                      │
    同样的模式 ────────────────────────-─┘

=== 架构图 ===

    关闭协议：
    Lead                              Teammate
    +---------------------+          +---------------------+
    | shutdown_request     |          |                     |
    | {                    | -------> | 收到请求            |
    |   request_id: abc    |          | 决定：同意关闭吗？   |
    | }                    |          |                     |
    +---------------------+          +---------------------+
                                            |
    +---------------------+          +------v--------------+
    | shutdown_response    | <------- | shutdown_response   |
    | {                    |          | {                   |
    |   request_id: abc    |          |   request_id: abc   |
    |   approve: true      |          |   approve: true     |
    | }                    |          | }                   |
    +---------------------+          +---------------------+
            |
            v
    状态 -> "shutdown", 线程停止

    计划审批协议：
    Teammate                          Lead
    +---------------------+          +---------------------+
    | plan_approval        |          |                     |
    | submit: {plan:"..."} | -------> | 审查计划文本        |
    +---------------------+          | 同意/拒绝？          |
                                     +---------------------+
                                            |
    +---------------------+          +------v--------------+
    | plan_approval_resp   | <------- | plan_approval       |
    | {approve: true}      |          | review: {req_id,    |
    +---------------------+          |   approve: true}     |
                                     +---------------------+

    请求跟踪器：
    {request_id: {"target|from": name, "status": "pending|approved|rejected"}}
"""

# ============================================================================
# 第一部分：导入依赖
# ============================================================================

import json
import os
import subprocess
import threading
import time
import uuid          # 生成唯一请求 ID
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
SYSTEM = f"You are a team lead at {WORKDIR}. Manage teammates with shutdown and plan approval protocols."

# 有效的消息类型
VALID_MSG_TYPES = {
    "message",                  # 普通消息
    "broadcast",                # 广播消息
    "shutdown_request",         # 关闭请求（Lead → 队友）
    "shutdown_response",        # 关闭响应（队友 → Lead）
    "plan_approval_response",   # 计划审批响应（双向）
}


# ============================================================================
# 第三部分：请求跟踪器（核心数据结构）
# ============================================================================

# 关闭请求跟踪器
# 格式: {request_id: {"target": teammate_name, "status": "pending|approved|rejected"}}
shutdown_requests = {}

# 计划审批请求跟踪器
# 格式: {request_id: {"from": teammate_name, "plan": plan_text, "status": "pending|approved|rejected"}}
plan_requests = {}

# 线程锁：保护跟踪器的并发访问
_tracker_lock = threading.Lock()


# ============================================================================
# 第四部分：MessageBus 类（消息总线，与 s09 相同）
# ============================================================================

class MessageBus:
    """
    消息总线：管理所有队友的收件箱。

    与 s09 完全相同，这里不再重复注释。
    """

    def __init__(self, inbox_dir: Path):
        self.dir = inbox_dir
        self.dir.mkdir(parents=True, exist_ok=True)

    def send(self, sender: str, to: str, content: str,
             msg_type: str = "message", extra: dict = None) -> str:
        """发送消息给指定队友"""
        if msg_type not in VALID_MSG_TYPES:
            return f"Error: Invalid type '{msg_type}'. Valid: {VALID_MSG_TYPES}"
        msg = {
            "type": msg_type,
            "from": sender,
            "content": content,
            "timestamp": time.time(),
        }
        if extra:
            msg.update(extra)
        inbox_path = self.dir / f"{to}.jsonl"
        with open(inbox_path, "a") as f:
            f.write(json.dumps(msg) + "\n")
        return f"Sent {msg_type} to {to}"

    def read_inbox(self, name: str) -> list:
        """读取并清空指定队友的收件箱"""
        inbox_path = self.dir / f"{name}.jsonl"
        if not inbox_path.exists():
            return []
        messages = []
        for line in inbox_path.read_text().strip().splitlines():
            if line:
                messages.append(json.loads(line))
        inbox_path.write_text("")
        return messages

    def broadcast(self, sender: str, content: str, teammates: list) -> str:
        """广播消息给所有队友"""
        count = 0
        for name in teammates:
            if name != sender:
                self.send(sender, name, content, "broadcast")
                count += 1
        return f"Broadcast to {count} teammates"


# ============================================================================
# 第五部分：创建全局消息总线
# ============================================================================

BUS = MessageBus(INBOX_DIR)


# ============================================================================
# 第六部分：TeammateManager 类（支持协议）
# ============================================================================

class TeammateManager:
    """
    队友管理器：支持关闭协议和计划审批协议。

    与 s09 的区别：
    1. 队友的系统提示词要求遵守协议
    2. 队友有新工具：shutdown_response, plan_approval
    3. 队友循环中处理关闭请求
    """

    def __init__(self, team_dir: Path):
        self.dir = team_dir
        self.dir.mkdir(exist_ok=True)
        self.config_path = self.dir / "config.json"
        self.config = self._load_config()
        self.threads = {}

    def _load_config(self) -> dict:
        if self.config_path.exists():
            return json.loads(self.config_path.read_text())
        return {"team_name": "default", "members": []}

    def _save_config(self):
        self.config_path.write_text(json.dumps(self.config, indent=2))

    def _find_member(self, name: str) -> dict:
        for m in self.config["members"]:
            if m["name"] == name:
                return m
        return None

    def spawn(self, name: str, role: str, prompt: str) -> str:
        """创建或激活队友（与 s09 相同）"""
        member = self._find_member(name)
        if member:
            if member["status"] not in ("idle", "shutdown"):
                return f"Error: '{name}' is currently {member['status']}"
            member["status"] = "working"
            member["role"] = role
        else:
            member = {"name": name, "role": role, "status": "working"}
            self.config["members"].append(member)
        self._save_config()
        thread = threading.Thread(
            target=self._teammate_loop,
            args=(name, role, prompt),
            daemon=True,
        )
        self.threads[name] = thread
        thread.start()
        return f"Spawned '{name}' (role: {role})"

    # -------------------------------------------------------------------------
    # 队友的 Agent 循环（支持协议）
    # -------------------------------------------------------------------------

    def _teammate_loop(self, name: str, role: str, prompt: str):
        """
        队友的 Agent 循环，支持关闭协议和计划审批。

        与 s09 的区别：
        1. 系统提示词要求遵守协议
        2. 检测 shutdown_response(approve=true) 时退出
        3. 可以提交计划等待审批

        关闭流程：
        1. Lead 发送 shutdown_request
        2. 队友收到请求（通过收件箱）
        3. 队友决定是否同意
        4. 队友调用 shutdown_response(approve=true/false)
        5. 如果 approve=true，设置 should_exit=True
        6. 下一轮循环检测到 should_exit，退出循环
        """
        # 系统提示词：强调协议
        sys_prompt = (
            f"You are '{name}', role: {role}, at {WORKDIR}. "
            f"Submit plans via plan_approval before major work. "
            f"Respond to shutdown_request with shutdown_response."
        )

        messages = [{"role": "user", "content": prompt}]
        tools = self._teammate_tools()

        # === 关键：退出标志 ===
        should_exit = False

        for _ in range(50):
            # 检查收件箱
            inbox = BUS.read_inbox(name)
            for msg in inbox:
                messages.append({"role": "user", "content": json.dumps(msg)})

            # 检查是否应该退出
            if should_exit:
                break

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

            if response.stop_reason != "tool_use":
                break

            results = []
            for block in response.content:
                if block.type == "tool_use":
                    output = self._exec(name, block.name, block.input)
                    print(f"  [{name}] {block.name}: {str(output)[:120]}")
                    results.append({
                        "type": "tool_result",
                        "tool_use_id": block.id,
                        "content": str(output),
                    })

                    # === 关键：检测关闭批准 ===
                    # 如果队友调用 shutdown_response(approve=true)，设置退出标志
                    if block.name == "shutdown_response" and block.input.get("approve"):
                        should_exit = True

            messages.append({"role": "user", "content": results})

        # 更新状态
        member = self._find_member(name)
        if member:
            # 根据退出原因设置状态
            member["status"] = "shutdown" if should_exit else "idle"
            self._save_config()

    # -------------------------------------------------------------------------
    # 队友的工具执行
    # -------------------------------------------------------------------------

    def _exec(self, sender: str, tool_name: str, args: dict) -> str:
        """
        执行队友的工具调用。

        新增工具：
        - shutdown_response: 响应关闭请求
        - plan_approval: 提交计划等待审批
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
        if tool_name == "send_message":
            return BUS.send(sender, args["to"], args["content"], args.get("msg_type", "message"))
        if tool_name == "read_inbox":
            return json.dumps(BUS.read_inbox(sender), indent=2)

        # === 新增：关闭响应 ===
        if tool_name == "shutdown_response":
            req_id = args["request_id"]
            approve = args["approve"]

            # 更新跟踪器状态
            with _tracker_lock:
                if req_id in shutdown_requests:
                    shutdown_requests[req_id]["status"] = "approved" if approve else "rejected"

            # 发送响应给 Lead
            BUS.send(
                sender, "lead", args.get("reason", ""),
                "shutdown_response", {"request_id": req_id, "approve": approve},
            )
            return f"Shutdown {'approved' if approve else 'rejected'}"

        # === 新增：计划审批提交 ===
        if tool_name == "plan_approval":
            plan_text = args.get("plan", "")

            # 生成请求 ID
            req_id = str(uuid.uuid4())[:8]

            # 记录到跟踪器
            with _tracker_lock:
                plan_requests[req_id] = {
                    "from": sender,
                    "plan": plan_text,
                    "status": "pending"
                }

            # 发送给 Lead 审批
            BUS.send(
                sender, "lead", plan_text, "plan_approval_response",
                {"request_id": req_id, "plan": plan_text},
            )
            return f"Plan submitted (request_id={req_id}). Waiting for lead approval."

        return f"Unknown tool: {tool_name}"

    def _teammate_tools(self) -> list:
        """
        队友可用的工具列表。

        新增工具：
        - shutdown_response: 响应关闭请求（同意或拒绝）
        - plan_approval: 提交计划等待 Lead 审批
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
            {"name": "send_message", "description": "Send message to a teammate.",
             "input_schema": {"type": "object", "properties": {"to": {"type": "string"}, "content": {"type": "string"}, "msg_type": {"type": "string", "enum": list(VALID_MSG_TYPES)}}, "required": ["to", "content"]}},
            {"name": "read_inbox", "description": "Read and drain your inbox.",
             "input_schema": {"type": "object", "properties": {}}},

            # 新增：关闭响应
            {
                "name": "shutdown_response",
                "description": "Respond to a shutdown request. Approve to shut down, reject to keep working.",
                "input_schema": {
                    "type": "object",
                    "properties": {
                        "request_id": {"type": "string"},
                        "approve": {"type": "boolean"},
                        "reason": {"type": "string"}
                    },
                    "required": ["request_id", "approve"]
                }
            },

            # 新增：计划审批提交
            {
                "name": "plan_approval",
                "description": "Submit a plan for lead approval. Provide plan text.",
                "input_schema": {
                    "type": "object",
                    "properties": {
                        "plan": {"type": "string"}
                    },
                    "required": ["plan"]
                }
            },
        ]

    def list_all(self) -> str:
        if not self.config["members"]:
            return "No teammates."
        lines = [f"Team: {self.config['team_name']}"]
        for m in self.config["members"]:
            lines.append(f"  {m['name']} ({m['role']}): {m['status']}")
        return "\n".join(lines)

    def member_names(self) -> list:
        return [m["name"] for m in self.config["members"]]


# ============================================================================
# 第七部分：创建全局队友管理器
# ============================================================================

TEAM = TeammateManager(TEAM_DIR)


# ============================================================================
# 第八部分：基础工具实现函数
# ============================================================================

def _safe_path(p: str) -> Path:
    path = (WORKDIR / p).resolve()
    if not path.is_relative_to(WORKDIR):
        raise ValueError(f"Path escapes workspace: {p}")
    return path


def _run_bash(command: str) -> str:
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
    try:
        lines = _safe_path(path).read_text().splitlines()
        if limit and limit < len(lines):
            lines = lines[:limit] + [f"... ({len(lines) - limit} more)"]
        return "\n".join(lines)[:50000]
    except Exception as e:
        return f"Error: {e}"


def _run_write(path: str, content: str) -> str:
    try:
        fp = _safe_path(path)
        fp.parent.mkdir(parents=True, exist_ok=True)
        fp.write_text(content)
        return f"Wrote {len(content)} bytes"
    except Exception as e:
        return f"Error: {e}"


def _run_edit(path: str, old_text: str, new_text: str) -> str:
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
# 第九部分：Lead 专用的协议处理函数
# ============================================================================

def handle_shutdown_request(teammate: str) -> str:
    """
    Lead 发起关闭请求。

    工作流程：
    1. 生成唯一的 request_id
    2. 记录到 shutdown_requests 跟踪器
    3. 发送 shutdown_request 消息给队友
    4. 返回 request_id 供后续查询

    Args:
        teammate: 要关闭的队友名称

    Returns:
        请求确认消息，包含 request_id
    """
    # 生成唯一请求 ID
    req_id = str(uuid.uuid4())[:8]

    # 记录到跟踪器
    with _tracker_lock:
        shutdown_requests[req_id] = {"target": teammate, "status": "pending"}

    # 发送关闭请求
    BUS.send(
        "lead", teammate, "Please shut down gracefully.",
        "shutdown_request", {"request_id": req_id},
    )

    return f"Shutdown request {req_id} sent to '{teammate}' (status: pending)"


def handle_plan_review(request_id: str, approve: bool, feedback: str = "") -> str:
    """
    Lead 审批计划。

    工作流程：
    1. 根据 request_id 查找计划请求
    2. 更新状态为 approved 或 rejected
    3. 发送响应给提交计划的队友

    Args:
        request_id: 计划请求 ID
        approve: 是否批准
        feedback: 反馈意见（可选）

    Returns:
        审批结果消息
    """
    # 查找请求
    with _tracker_lock:
        req = plan_requests.get(request_id)

    if not req:
        return f"Error: Unknown plan request_id '{request_id}'"

    # 更新状态
    with _tracker_lock:
        req["status"] = "approved" if approve else "rejected"

    # 发送响应
    BUS.send(
        "lead", req["from"], feedback, "plan_approval_response",
        {"request_id": request_id, "approve": approve, "feedback": feedback},
    )

    return f"Plan {req['status']} for '{req['from']}'"


def _check_shutdown_status(request_id: str) -> str:
    """查询关闭请求状态"""
    with _tracker_lock:
        return json.dumps(shutdown_requests.get(request_id, {"error": "not found"}))


# ============================================================================
# 第十部分：Lead 的工具分发映射
# ============================================================================

# Lead 有 12 个工具（比 s09 多 3 个协议相关工具）
TOOL_HANDLERS = {
    # 基础工具
    "bash":              lambda **kw: _run_bash(kw["command"]),
    "read_file":         lambda **kw: _run_read(kw["path"], kw.get("limit")),
    "write_file":        lambda **kw: _run_write(kw["path"], kw["content"]),
    "edit_file":         lambda **kw: _run_edit(kw["path"], kw["old_text"], kw["new_text"]),

    # 团队管理工具
    "spawn_teammate":    lambda **kw: TEAM.spawn(kw["name"], kw["role"], kw["prompt"]),
    "list_teammates":    lambda **kw: TEAM.list_all(),

    # 通信工具
    "send_message":      lambda **kw: BUS.send("lead", kw["to"], kw["content"], kw.get("msg_type", "message")),
    "read_inbox":        lambda **kw: json.dumps(BUS.read_inbox("lead"), indent=2),
    "broadcast":         lambda **kw: BUS.broadcast("lead", kw["content"], TEAM.member_names()),

    # 协议工具（新增）
    "shutdown_request":  lambda **kw: handle_shutdown_request(kw["teammate"]),
    "shutdown_response": lambda **kw: _check_shutdown_status(kw.get("request_id", "")),
    "plan_approval":     lambda **kw: handle_plan_review(kw["request_id"], kw["approve"], kw.get("feedback", "")),
}


# ============================================================================
# 第十一部分：Lead 的工具定义列表
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

    # 团队管理工具
    {"name": "spawn_teammate", "description": "Spawn a persistent teammate.",
     "input_schema": {"type": "object", "properties": {"name": {"type": "string"}, "role": {"type": "string"}, "prompt": {"type": "string"}}, "required": ["name", "role", "prompt"]}},
    {"name": "list_teammates", "description": "List all teammates.",
     "input_schema": {"type": "object", "properties": {}}},

    # 通信工具
    {"name": "send_message", "description": "Send a message to a teammate.",
     "input_schema": {"type": "object", "properties": {"to": {"type": "string"}, "content": {"type": "string"}, "msg_type": {"type": "string", "enum": list(VALID_MSG_TYPES)}}, "required": ["to", "content"]}},
    {"name": "read_inbox", "description": "Read and drain the lead's inbox.",
     "input_schema": {"type": "object", "properties": {}}},
    {"name": "broadcast", "description": "Send a message to all teammates.",
     "input_schema": {"type": "object", "properties": {"content": {"type": "string"}}, "required": ["content"]}},

    # 协议工具（新增）
    {
        "name": "shutdown_request",
        "description": "Request a teammate to shut down gracefully. Returns a request_id for tracking.",
        "input_schema": {
            "type": "object",
            "properties": {"teammate": {"type": "string"}},
            "required": ["teammate"]
        }
    },
    {
        "name": "shutdown_response",
        "description": "Check the status of a shutdown request by request_id.",
        "input_schema": {
            "type": "object",
            "properties": {"request_id": {"type": "string"}},
            "required": ["request_id"]
        }
    },
    {
        "name": "plan_approval",
        "description": "Approve or reject a teammate's plan. Provide request_id + approve + optional feedback.",
        "input_schema": {
            "type": "object",
            "properties": {
                "request_id": {"type": "string"},
                "approve": {"type": "boolean"},
                "feedback": {"type": "string"}
            },
            "required": ["request_id", "approve"]
        }
    },
]


# ============================================================================
# 第十二部分：序列化响应内容
# ============================================================================

def serialize_content(content):
    """
    将 API 响应的 content 转换为可序列化的字典格式。

    重要：不能使用 model_dump()！必须手动提取字段。
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
# 第十三部分：Lead 的 Agent 循环
# ============================================================================

def agent_loop(messages: list):
    """
    Lead 的 Agent 循环。

    与 s09 的区别：
    - 可以发起关闭请求
    - 可以审批队友的计划
    - 收件箱中会收到协议相关的消息
    """
    while True:
        # 检查收件箱
        inbox = BUS.read_inbox("lead")
        if inbox:
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

        # 保存响应
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
# 第十四部分：主程序入口
# ============================================================================

if __name__ == "__main__":
    """
    交互式命令行界面。

    使用示例：
        s10 >> 创建一个队友 alice，让她先提交一个实现计划

    观察协议流程：
        1. Lead spawn alice
        2. Alice 提交 plan_approval
        3. Lead 收到计划，审批（approve/reject）
        4. Alice 收到审批结果，继续或修改

    关闭协议示例：
        s10 >> 请求 alice 关闭

    观察：
        1. Lead 发送 shutdown_request
        2. Alice 收到请求，决定是否同意
        3. Alice 发送 shutdown_response
        4. 如果同意，Alice 的线程退出
        5. Alice 状态变为 "shutdown"

    快捷命令：
        /team  - 查看团队状态
        /inbox - 查看 Lead 的收件箱
    """
    history = []

    print("=" * 60)
    print("s10_team_protocols.py - 团队协议示例")
    print("新增功能: shutdown_request/response, plan_approval")
    print("核心模式: request_id 关联")
    print("快捷命令: /team (查看团队), /inbox (查看收件箱)")
    print("输入 'q' 或 'exit' 退出")
    print("=" * 60)

    while True:
        try:
            query = input("\033[36ms10 >> \033[0m")
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
                if isinstance(block, dict) and block.get("type") == "text":
                    print(block["text"])
                elif hasattr(block, "text"):
                    print(block.text)

        print()
