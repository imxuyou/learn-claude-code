#!/usr/bin/env python3
"""
s05_skill_loading.py - Skills (技能动态加载)

=== 本课要点 ===

引入"技能"概念：将专业知识模块化，按需加载。

核心洞察：
    "不要把所有东西都放在系统提示词中，按需加载。"

=== 为什么需要技能系统？ ===

问题：如果把所有专业知识都放在系统提示词中...
    - 系统提示词会变得巨大（几万 token）
    - 每次 API 调用都要发送这些内容
    - 大部分知识在当前任务中用不到
    - 成本高、速度慢

解决：两层加载策略
    - Layer 1: 只在系统提示词中放技能名称和简短描述（~100 tokens/skill）
    - Layer 2: 当模型需要时，通过工具调用加载完整内容

=== 两层加载架构 ===

    skills/
      pdf/
        SKILL.md          <-- YAML frontmatter + 完整指令
      code-review/
        SKILL.md

    Layer 1 - 系统提示词（始终存在）:
    +--------------------------------------+
    | You are a coding agent.              |
    | Skills available:                    |
    |   - pdf: Process PDF files...        |  <-- 只有名称和简短描述
    |   - code-review: Review code...      |
    +--------------------------------------+

    Layer 2 - 按需加载（通过 tool_result）:
    +--------------------------------------+
    | 当模型调用 load_skill("pdf"):         |
    |                                      |
    | tool_result:                         |
    | <skill>                              |
    |   Full PDF processing instructions   |  <-- 完整的技能内容
    |   Step 1: ...                        |
    |   Step 2: ...                        |
    | </skill>                             |
    +--------------------------------------+

=== SKILL.md 文件格式 ===

    ---
    name: pdf
    description: Process and analyze PDF files
    tags: document, extraction
    ---

    # PDF Processing Skill

    When working with PDF files, follow these steps:

    1. First, check if poppler-utils is installed...
    2. Use pdftotext for text extraction...
    ...

前半部分是 YAML frontmatter（元数据），后半部分是技能正文。
"""

# ============================================================================
# 第一部分：导入依赖
# ============================================================================

import os
import re           # 用于解析 YAML frontmatter
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

# 技能目录：存放所有 SKILL.md 文件
SKILLS_DIR = WORKDIR / "skills"


# ============================================================================
# 第三部分：SkillLoader 类（本课核心）
# ============================================================================

class SkillLoader:
    """
    技能加载器：扫描并管理技能文件。

    职责：
    1. 扫描 skills/ 目录下的所有 SKILL.md 文件
    2. 解析 YAML frontmatter 提取元数据
    3. 提供 Layer 1 描述（用于系统提示词）
    4. 提供 Layer 2 内容（用于按需加载）

    目录结构示例：
        skills/
        ├── pdf/
        │   └── SKILL.md
        ├── code-review/
        │   └── SKILL.md
        └── agent-builder/
            └── SKILL.md

    SKILL.md 格式：
        ---
        name: pdf
        description: Process PDF files
        tags: document
        ---

        [技能正文内容...]
    """

    def __init__(self, skills_dir: Path):
        self.skills_dir = skills_dir
        self.skills = {}  # {name: {"meta": {...}, "body": "...", "path": "..."}}
        self._load_all()

    def _load_all(self):
        """
        扫描并加载所有技能文件。

        使用 rglob 递归搜索所有 SKILL.md 文件，
        这样技能可以嵌套在子目录中。
        """
        if not self.skills_dir.exists():
            return

        # 递归查找所有 SKILL.md 文件，按路径排序确保一致性
        for f in sorted(self.skills_dir.rglob("SKILL.md")):
            text = f.read_text()

            # 解析 frontmatter 和正文
            meta, body = self._parse_frontmatter(text)

            # 技能名称：优先使用 frontmatter 中的 name，否则使用目录名
            name = meta.get("name", f.parent.name)

            self.skills[name] = {
                "meta": meta,    # 元数据字典
                "body": body,    # 技能正文
                "path": str(f)   # 文件路径（用于调试）
            }

    def _parse_frontmatter(self, text: str) -> tuple:
        """
        解析 YAML frontmatter。

        YAML frontmatter 是文件开头用 --- 包围的部分：

            ---
            name: pdf
            description: Process PDF files
            tags: document, extraction
            ---

            [正文内容...]

        这是一种常见的格式，Jekyll、Hugo 等静态网站生成器都使用它。

        Args:
            text: 完整的文件内容

        Returns:
            (meta_dict, body_text) 元组
        """
        # 正则匹配：开头的 ---，然后是任意内容，然后是 ---，然后是剩余内容
        match = re.match(r"^---\n(.*?)\n---\n(.*)", text, re.DOTALL)

        if not match:
            # 没有 frontmatter，整个文件都是正文
            return {}, text

        # 解析 YAML（简化版，只支持 key: value 格式）
        meta = {}
        for line in match.group(1).strip().splitlines():
            if ":" in line:
                key, val = line.split(":", 1)
                meta[key.strip()] = val.strip()

        return meta, match.group(2).strip()

    def get_descriptions(self) -> str:
        """
        获取所有技能的简短描述（Layer 1）。

        这个内容会被注入到系统提示词中，让模型知道有哪些技能可用。

        返回格式：
            - pdf: Process PDF files [document, extraction]
            - code-review: Review code quality [quality]

        Returns:
            格式化的技能列表字符串
        """
        if not self.skills:
            return "(no skills available)"

        lines = []
        for name, skill in self.skills.items():
            desc = skill["meta"].get("description", "No description")
            tags = skill["meta"].get("tags", "")

            line = f"  - {name}: {desc}"
            if tags:
                line += f" [{tags}]"

            lines.append(line)

        return "\n".join(lines)

    def get_content(self, name: str) -> str:
        """
        获取指定技能的完整内容（Layer 2）。

        当模型调用 load_skill 工具时，返回完整的技能正文。
        使用 <skill> 标签包裹，方便模型识别。

        Args:
            name: 技能名称

        Returns:
            <skill name="xxx">...</skill> 格式的完整内容
        """
        skill = self.skills.get(name)

        if not skill:
            available = ', '.join(self.skills.keys())
            return f"Error: Unknown skill '{name}'. Available: {available}"

        # 使用 XML 标签包裹技能内容
        return f"<skill name=\"{name}\">\n{skill['body']}\n</skill>"


# ============================================================================
# 第四部分：初始化技能加载器和系统提示词
# ============================================================================

# 创建全局技能加载器实例
SKILL_LOADER = SkillLoader(SKILLS_DIR)

# Layer 1: 技能元数据注入系统提示词
# 注意：这里只包含技能名称和简短描述，不包含完整内容
SYSTEM = f"""You are a coding agent at {WORKDIR}.
Use load_skill to access specialized knowledge before tackling unfamiliar topics.

Skills available:
{SKILL_LOADER.get_descriptions()}"""


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
        content = fp.read_text()
        if old_text not in content:
            return f"Error: Text not found in {path}"
        fp.write_text(content.replace(old_text, new_text, 1))
        return f"Edited {path}"
    except Exception as e:
        return f"Error: {e}"


# ============================================================================
# 第六部分：工具分发映射
# ============================================================================

TOOL_HANDLERS = {
    "bash":       lambda **kw: run_bash(kw["command"]),
    "read_file":  lambda **kw: run_read(kw["path"], kw.get("limit")),
    "write_file": lambda **kw: run_write(kw["path"], kw["content"]),
    "edit_file":  lambda **kw: run_edit(kw["path"], kw["old_text"], kw["new_text"]),
    "load_skill": lambda **kw: SKILL_LOADER.get_content(kw["name"]),  # ← 新增！
}


# ============================================================================
# 第七部分：工具定义列表
# ============================================================================

TOOLS = [
    # 基础工具（与之前相同）
    {"name": "bash", "description": "Run a shell command.",
     "input_schema": {"type": "object", "properties": {"command": {"type": "string"}}, "required": ["command"]}},
    {"name": "read_file", "description": "Read file contents.",
     "input_schema": {"type": "object", "properties": {"path": {"type": "string"}, "limit": {"type": "integer"}}, "required": ["path"]}},
    {"name": "write_file", "description": "Write content to file.",
     "input_schema": {"type": "object", "properties": {"path": {"type": "string"}, "content": {"type": "string"}}, "required": ["path", "content"]}},
    {"name": "edit_file", "description": "Replace exact text in file.",
     "input_schema": {"type": "object", "properties": {"path": {"type": "string"}, "old_text": {"type": "string"}, "new_text": {"type": "string"}}, "required": ["path", "old_text", "new_text"]}},

    # 新增：load_skill 工具
    {
        "name": "load_skill",
        "description": "Load specialized knowledge by name.",
        "input_schema": {
            "type": "object",
            "properties": {
                "name": {
                    "type": "string",
                    "description": "Skill name to load"
                }
            },
            "required": ["name"]
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

    工作流程示例：
    1. 用户: "帮我处理这个 PDF 文件"
    2. 模型看到系统提示词中有 pdf 技能可用
    3. 模型调用 load_skill("pdf")
    4. 系统返回完整的 PDF 处理指令
    5. 模型按照指令处理 PDF
    """
    while True:
        response = client.messages.create(
            model=MODEL, system=SYSTEM, messages=messages,
            tools=TOOLS, max_tokens=8000,
        )

        # 保存响应（序列化）
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
        s05 >> 帮我处理 document.pdf 文件，提取其中的文本

    观察：
        1. 模型看到系统提示词中有 pdf 技能
        2. 模型调用 load_skill("pdf") 获取详细指令
        3. 模型按照指令处理 PDF 文件
    """
    history = []

    print("=" * 60)
    print("s05_skill_loading.py - 技能动态加载示例")
    print("新增工具: load_skill（按需加载专业知识）")
    print(f"可用技能: {', '.join(SKILL_LOADER.skills.keys()) or '(无)'}")
    print("输入 'q' 或 'exit' 退出")
    print("=" * 60)

    while True:
        try:
            query = input("\033[36ms05 >> \033[0m")
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
