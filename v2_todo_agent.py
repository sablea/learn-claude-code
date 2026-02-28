#!/usr/bin/env python3
"""
v2_todo_agent.py - Mini Claude Code：结构化规划（约 300 行）

核心理念：“让计划可见”
=====================
v1 在简单任务上表现很好。但当你要求它“重构鉴权、补测试、更新文档”时，
问题就会出现。没有显式规划时，模型会：
    - 在任务之间随机跳转
    - 忘记已经完成的步骤
    - 中途失去焦点

问题——“上下文衰减”：
--------------------
在 v1 中，计划只存在于模型的“脑海”里：

        v1: “我先做 A，再做 B，然后做 C”（不可见）
                10 次工具调用后：“等等，我刚刚在做什么？”

解决方案——TodoWrite 工具：
--------------------------
v2 新增了一个工具，但它从根本上改变了代理的工作方式：

        v2:
            [ ] 重构鉴权模块
            [>] 添加单元测试         <- 当前正在做
            [ ] 更新文档

现在你和模型都能看到计划。模型可以：
    - 在执行过程中更新状态
    - 明确知道已完成项和下一步
    - 一次只专注于一个任务

关键约束（并非随意设定——它们是护栏）：
----------------------------------------
        | 规则              | 原因                            |
        |-------------------|---------------------------------|
        | 最多 20 项        | 防止生成无限任务清单            |
        | 仅一个 in_progress| 强制一次只专注一件事            |
        | 必填字段          | 保证结构化输出                  |

深层洞察：
----------
> “结构既约束，也赋能。”

Todo 约束（最大数量、仅一个 in_progress）会“赋能”（计划可见、进度可跟踪）。

这种模式在代理设计中随处可见：
    - max_tokens 的约束 -> 使响应可控
    - 工具 schema 的约束 -> 使调用结构化
    - Todo 的约束 -> 支撑复杂任务完成

好的约束不是限制，而是脚手架。

用法：
        python v2_todo_agent.py
"""

import os
import subprocess
import sys
import json
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()

try:
    from openai import OpenAI
except ImportError:
    sys.exit("Please install: pip install openai python-dotenv")


# =============================================================================
# 配置
# =============================================================================

API_KEY = os.getenv("OPENAI_API_KEY")
BASE_URL = os.getenv("BASE_URL")
MODEL = os.getenv("MODEL_NAME", "gpt-4.1")
WORKDIR = Path.cwd()

if not API_KEY:
    sys.exit("OPENAI_API_KEY is required")

client = OpenAI(api_key=API_KEY, base_url=BASE_URL) if BASE_URL else OpenAI(api_key=API_KEY)


# =============================================================================
# TodoManager - v2 的核心新增
# =============================================================================

class TodoManager:
    """
    管理带有强约束的结构化任务清单。

    关键设计决策：
    --------------------
    1. 最多 20 项：防止模型生成无穷无尽的清单
    2. 仅一个 in_progress：强制聚焦——同一时间只能做一件事
    3. 必填字段：每个条目都需要 content、status、activeForm

    activeForm 字段需要特别说明：
    - 它表示“当前动作”的现在进行式
    - 当 status 为 "in_progress" 时展示
    - 示例：content="添加测试"，activeForm="正在添加单元测试..."

    这样就能实时看到代理正在做什么。
    """

    def __init__(self):
        self.items = []

    def update(self, items: list) -> str:
        """
        校验并更新 todo 清单。

        模型每次都会发送一份完整的新清单。我们进行校验、存储，
        并返回渲染后的视图供模型读取。

        校验规则：
        - 每个条目必须包含：content、status、activeForm
        - status 必须是：pending | in_progress | completed
        - 同一时间只能有一个条目为 in_progress
        - 最多允许 20 个条目

        Returns:
            todo 清单的渲染文本视图
        """
        validated = []
        in_progress_count = 0

        for i, item in enumerate(items):
            # 提取并校验字段
            content = str(item.get("content", "")).strip()
            status = str(item.get("status", "pending")).lower()
            active_form = str(item.get("activeForm", "")).strip()

            # 校验检查
            if not content:
                raise ValueError(f"Item {i}: content required")
            if status not in ("pending", "in_progress", "completed"):
                raise ValueError(f"Item {i}: invalid status '{status}'")
            if not active_form:
                raise ValueError(f"Item {i}: activeForm required")

            if status == "in_progress":
                in_progress_count += 1

            validated.append({
                "content": content,
                "status": status,
                "activeForm": active_form
            })

        # 强制约束
        if len(validated) > 20:
            raise ValueError("Max 20 todos allowed")
        if in_progress_count > 1:
            raise ValueError("Only one task can be in_progress at a time")

        self.items = validated
        return self.render()

    def render(self) -> str:
        """
        将 todo 清单渲染为人类可读文本。

        格式：
            [x] 已完成任务
            [>] 进行中任务 <- 正在做某事...
            [ ] 待处理任务

            （2/3 已完成）

        这段渲染文本会作为工具结果返回给模型。
        模型随后可基于当前状态继续更新清单。
        """
        if not self.items:
            return "No todos."

        lines = []
        for item in self.items:
            if item["status"] == "completed":
                lines.append(f"[x] {item['content']}")
            elif item["status"] == "in_progress":
                lines.append(f"[>] {item['content']} <- {item['activeForm']}")
            else:
                lines.append(f"[ ] {item['content']}")

        completed = sum(1 for t in self.items if t["status"] == "completed")
        lines.append(f"\n({completed}/{len(self.items)} completed)")

        return "\n".join(lines)


# 全局 Todo 管理器实例
TODO = TodoManager()


# =============================================================================
# 系统提示词 - v2 更新版
# =============================================================================

SYSTEM = f"""You are a coding agent at {WORKDIR}.

Loop: plan -> act with tools -> update todos -> report.

Rules:
- Use TodoWrite to track multi-step tasks
- Mark tasks in_progress before starting, completed when done
- Prefer tools over prose. Act, don't just explain.
- After finishing, summarize what changed."""


# =============================================================================
# 系统提醒 - 用于鼓励使用 todo 的软提示
# =============================================================================

# 在对话开始时显示
INITIAL_REMINDER = "<reminder>Use TodoWrite for multi-step tasks.</reminder>"

# 当模型较长时间未更新 todo 时显示
NAG_REMINDER = "<reminder>10+ turns without todo update. Please update todos.</reminder>"


# =============================================================================
# 工具定义（v1 工具 + TodoWrite）
# =============================================================================

TOOLS = [
    # v1 工具（不变）
    {
        "type": "function",
        "function": {
            "name": "bash",
            "description": "Run a shell command.",
            "parameters": {
                "type": "object",
                "properties": {"command": {"type": "string"}},
                "required": ["command"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "read_file",
            "description": "Read file contents.",
            "parameters": {
                "type": "object",
                "properties": {
                    "path": {"type": "string"},
                    "limit": {"type": "integer"}
                },
                "required": ["path"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "write_file",
            "description": "Write content to file.",
            "parameters": {
                "type": "object",
                "properties": {
                    "path": {"type": "string"},
                    "content": {"type": "string"}
                },
                "required": ["path", "content"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "edit_file",
            "description": "Replace exact text in file.",
            "parameters": {
                "type": "object",
                "properties": {
                    "path": {"type": "string"},
                    "old_text": {"type": "string"},
                    "new_text": {"type": "string"},
                },
                "required": ["path", "old_text", "new_text"],
            },
        },
    },

    # v2 新增：TodoWrite
    # 这是实现结构化规划的关键新增
    {
        "type": "function",
        "function": {
            "name": "TodoWrite",
            "description": "Update the task list. Use to plan and track progress.",
            "parameters": {
                "type": "object",
                "properties": {
                    "items": {
                        "type": "array",
                        "description": "Complete list of tasks (replaces existing)",
                        "items": {
                            "type": "object",
                            "properties": {
                                "content": {
                                    "type": "string",
                                    "description": "Task description"
                                },
                                "status": {
                                    "type": "string",
                                    "enum": ["pending", "in_progress", "completed"],
                                    "description": "Task status"
                                },
                                "activeForm": {
                                    "type": "string",
                                    "description": "Present tense action, e.g. 'Reading files'"
                                },
                            },
                            "required": ["content", "status", "activeForm"],
                        },
                    }
                },
                "required": ["items"],
            },
        },
    },
]


# =============================================================================
# 工具实现（v1 + TodoWrite）
# =============================================================================

def safe_path(p: str) -> Path:
    """确保路径始终位于工作区内。"""
    path = (WORKDIR / p).resolve()
    if not path.is_relative_to(WORKDIR):
        raise ValueError(f"Path escapes workspace: {p}")
    return path


def run_bash(cmd: str) -> str:
    """执行带安全检查的 shell 命令。"""
    dangerous = ["rm -rf /", "sudo", "shutdown", "reboot"]
    if any(d in cmd for d in dangerous):
        return "Error: Dangerous command blocked"
    try:
        result = subprocess.run(
            cmd, shell=True, cwd=WORKDIR,
            capture_output=True, text=True, timeout=60
        )
        output = (result.stdout + result.stderr).strip()
        return output[:50000] if output else "(no output)"
    except subprocess.TimeoutExpired:
        return "Error: Timeout"
    except Exception as e:
        return f"Error: {e}"


def run_read(path: str, limit: int = None) -> str:
    """读取文件内容。"""
    try:
        text = safe_path(path).read_text()
        lines = text.splitlines()
        if limit and limit < len(lines):
            lines = lines[:limit] + [f"... ({len(text.splitlines()) - limit} more)"]
        return "\n".join(lines)[:50000]
    except Exception as e:
        return f"Error: {e}"


def run_write(path: str, content: str) -> str:
    """将内容写入文件。"""
    try:
        fp = safe_path(path)
        fp.parent.mkdir(parents=True, exist_ok=True)
        fp.write_text(content)
        return f"Wrote {len(content)} bytes to {path}"
    except Exception as e:
        return f"Error: {e}"


def run_edit(path: str, old_text: str, new_text: str) -> str:
    """在文件中精确替换文本。"""
    try:
        fp = safe_path(path)
        content = fp.read_text()
        if old_text not in content:
            return f"Error: Text not found in {path}"
        fp.write_text(content.replace(old_text, new_text, 1))
        return f"Edited {path}"
    except Exception as e:
        return f"Error: {e}"


def run_todo(items: list) -> str:
    """
    更新 todo 清单。

    模型会发送完整的新清单（而非 diff）。
    我们进行校验并返回渲染视图。
    """
    try:
        return TODO.update(items)
    except Exception as e:
        return f"Error: {e}"


def execute_tool(name: str, args: dict) -> str:
    """将工具调用分发到对应实现。"""
    if name == "bash":
        return run_bash(args["command"])
    if name == "read_file":
        return run_read(args["path"], args.get("limit"))
    if name == "write_file":
        return run_write(args["path"], args["content"])
    if name == "edit_file":
        return run_edit(args["path"], args["old_text"], args["new_text"])
    if name == "TodoWrite":
        return run_todo(args["items"])
    return f"Unknown tool: {name}"


# =============================================================================
# 代理循环（含 todo 跟踪）
# =============================================================================

# 记录距离上次更新 todo 经过了多少轮
rounds_without_todo = 0


def agent_loop(messages: list) -> list:
    """
    带有 todo 使用跟踪的代理循环。

    核心循环与 v1 相同，但现在会跟踪模型是否使用了 todo。
    如果长时间未更新，会在 main() 中注入提醒。
    """
    global rounds_without_todo

    while True:
        response = client.chat.completions.create(
            model=MODEL,
            messages=messages,
            tools=TOOLS,
            tool_choice="auto",
            max_tokens=8000,
        )

        assistant_message = response.choices[0].message
        tool_calls = assistant_message.tool_calls or []

        if assistant_message.content:
            print(assistant_message.content)

        if not tool_calls:
            messages.append({
                "role": "assistant",
                "content": assistant_message.content or ""
            })
            return messages

        messages.append(assistant_message.model_dump(exclude_none=True))

        results = []
        used_todo = False

        for tc in tool_calls:
            tool_name = tc.function.name
            try:
                tool_args = json.loads(tc.function.arguments or "{}")
            except json.JSONDecodeError:
                tool_args = {}

            print(f"\n> {tool_name}")
            output = execute_tool(tool_name, tool_args)
            preview = output[:300] + "..." if len(output) > 300 else output
            print(f"  {preview}")

            results.append({
                "role": "tool",
                "tool_call_id": tc.id,
                "content": output,
            })

            # 跟踪 todo 使用情况
            if tool_name == "TodoWrite":
                used_todo = True

        # 更新计数器：如果用了 todo 就重置，否则递增
        if used_todo:
            rounds_without_todo = 0
        else:
            rounds_without_todo += 1

        messages.extend(results)


# =============================================================================
# 主 REPL
# =============================================================================

def main():
    """
    带提醒注入的 REPL。

    v2 的关键新增：注入“提醒”消息，以鼓励使用 todo，
    但不进行强制。这属于软约束。

    提醒会作为用户消息的一部分注入，而不是独立系统提示词。
    模型会看到它们，但不会直接回复这些提醒。
    """
    global rounds_without_todo

    print(f"Mini Claude Code v2 (with Todos) - {WORKDIR}")
    print("Type 'exit' to quit.\n")

    history = []
    first_message = True

    # OpenAI 要求在对话中显式包含 system 消息
    history.append({"role": "system", "content": SYSTEM})

    while True:
        try:
            user_input = input("You: ").strip()
        except (EOFError, KeyboardInterrupt):
            break

        if not user_input or user_input.lower() in ("exit", "quit", "q"):
            break

        # 构建用户消息内容
        # 可能包含作为上下文提示的提醒
        content = []

        if first_message:
            # 开始时给一个温和提醒
            content.append(INITIAL_REMINDER)
            first_message = False
        elif rounds_without_todo > 10:
            # 若模型一段时间未使用 todo，则进行催促
            content.append(NAG_REMINDER)

        content.append(user_input)
        history.append({"role": "user", "content": "\n".join(content)})

        try:
            agent_loop(history)
        except Exception as e:
            print(f"Error: {e}")

        print()


if __name__ == "__main__":
    main()
