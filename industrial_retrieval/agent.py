"""
Agent 核心模块。

包含系统提示词构建、Agent 主循环和交互式会话入口。
Agent 循环遵循标准的 tool_use 模式：
    while stop_reason == "tool_use":
        执行工具 → 追加结果 → 继续调用模型
"""

from anthropic import Anthropic

from industrial_retrieval.config import ANTHROPIC_BASE_URL, MODEL
from industrial_retrieval.skills.runner import get_loader
from industrial_retrieval.tools.definitions import build_tools
from industrial_retrieval.tools.executor import execute_tool

# ── Anthropic 客户端（模块级单例）────────────────────────────────────────────
_client = Anthropic(base_url=ANTHROPIC_BASE_URL)


def build_system_prompt() -> str:
    """
    构建系统提示词。

    系统提示词保持静态（只包含技能名称和描述），技能正文通过工具调用
    按需注入，从而保持提示词前缀不变，最大化利用 Anthropic 提示词缓存。
    """
    skill_descriptions = get_loader().get_descriptions()

    return f"""\
你是工业设备管理系统的智能检索助手。

你的职责：帮助技术员快速查询设备信息、故障记录、维护历史和操作规程。

**可用技能**（调用 Skill 工具加载）：
{skill_descriptions}

**检索工具**：
- sql_query:      查询结构化数据（设备台账、维护记录、故障代码、零件库存）
- rag_search:     搜索文档（故障报告、SOP、设备手册）
- list_tables:    查看数据库表结构
- list_documents: 查看可检索的文档列表

**工作准则**：
1. 先用 Skill 工具加载 industrial-retrieval 技能，获取数据库结构和查询指导
2. 根据问题类型选择合适工具：结构化数据用 sql_query，文档用 rag_search
3. 复杂问题同时使用 SQL 和 RAG，综合多源信息回答
4. 用清晰的中文回答，关键数据用表格展示
5. 如果找不到信息，明确告知用户\
"""


def agent_loop(messages: list[dict]) -> list[dict]:
    """
    核心 Agent 循环。

    模型持续调用工具直到 stop_reason 不再是 "tool_use"，
    每轮工具结果作为用户消息追加到对话历史中。

    Args:
        messages: 对话历史（会原地追加，调用方可保留引用以维护多轮上下文）。

    Returns:
        更新后的对话历史。
    """
    system  = build_system_prompt()
    tools   = build_tools()

    while True:
        response = _client.messages.create(
            model=MODEL,
            system=system,
            messages=messages,
            tools=tools,
            max_tokens=8000,
        )

        # 打印模型的文本输出
        for block in response.content:
            if hasattr(block, "text"):
                print(block.text)

        # 无工具调用 → 任务完成
        if response.stop_reason != "tool_use":
            messages.append({"role": "assistant", "content": response.content})
            return messages

        # 执行工具并收集结果
        tool_results = _execute_tool_calls(response.content)

        messages.append({"role": "assistant", "content": response.content})
        messages.append({"role": "user",      "content": tool_results})


def _execute_tool_calls(content_blocks: list) -> list[dict]:
    """
    执行响应中所有工具调用，返回 tool_result 列表。

    Args:
        content_blocks: 模型响应的 content 块列表。

    Returns:
        tool_result 格式的结果列表。
    """
    results = []
    for block in content_blocks:
        if block.type != "tool_use":
            continue

        print(f"\n> {block.name}: {block.input}")
        output  = execute_tool(block.name, block.input)
        preview = output[:200] + "..." if len(output) > 200 else output
        print(f"  {preview}")

        results.append({
            "type":        "tool_result",
            "tool_use_id": block.id,
            "content":     output,
        })

    return results


def run_interactive_session() -> None:
    """
    启动交互式命令行检索会话。

    维护多轮对话历史，支持持续对话。
    输入 'exit' 或 Ctrl+C 退出。
    """
    from industrial_retrieval.config import DB_PATH, DOCS_DIR
    from industrial_retrieval.database.schema import init_database
    from industrial_retrieval.rag.documents import init_documents

    # 确保数据库和文档库已初始化
    init_database(DB_PATH)
    init_documents(DOCS_DIR)

    print("=" * 60)
    print("工业设备管理系统 - 智能检索助手")
    print("=" * 60)
    print(f"数据库：{DB_PATH}")
    print(f"文档库：{DOCS_DIR}")
    print("\n示例查询：")
    print("  - CNC001设备当前状态如何？")
    print("  - 查询所有处于故障状态的设备")
    print("  - E003故障码的处理方法是什么？")
    print("  - 液压压力机紧急停机的操作流程")
    print("  - 哪些零件库存不足需要补货？")
    print("\n输入 'exit' 退出\n")

    history: list[dict] = []

    while True:
        try:
            user_input = input("技术员: ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\n退出系统。")
            break

        if not user_input or user_input.lower() in ("exit", "quit", "q", "退出"):
            print("退出系统。")
            break

        history.append({"role": "user", "content": user_input})

        try:
            agent_loop(history)
        except Exception as e:
            print(f"错误：{e}")

        print()
