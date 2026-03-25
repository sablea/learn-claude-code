"""
LLM 集成测试（需要 ANTHROPIC_API_KEY）。

测试 Agent 能否正确调用 SQL 和 RAG 工具响应自然语言查询。
没有 API Key 时自动跳过所有测试。
"""

import os
import pytest

pytestmark = pytest.mark.skipif(
    not os.getenv("ANTHROPIC_API_KEY") and not os.getenv("TEST_API_KEY"),
    reason="需要 ANTHROPIC_API_KEY 才能运行 LLM 集成测试",
)


@pytest.fixture()
def llm_client():
    from anthropic import Anthropic

    api_key = os.getenv("TEST_API_KEY") or os.getenv("ANTHROPIC_API_KEY")
    base_url = os.getenv("TEST_BASE_URL") or os.getenv("ANTHROPIC_BASE_URL")
    os.environ.pop("ANTHROPIC_AUTH_TOKEN", None)
    return Anthropic(api_key=api_key, base_url=base_url)


def _run_single_turn(client, user_msg: str, tools: list, system: str = "") -> tuple:
    """运行单轮 Agent，返回 (text, tool_calls)。"""
    model = os.getenv("TEST_MODEL") or os.getenv("MODEL_ID") or "glm-5"
    messages = [{"role": "user", "content": user_msg}]

    for _ in range(5):
        resp = client.messages.create(
            model=model,
            system=system or "你是数据检索助手，使用工具回答问题。",
            messages=messages,
            tools=tools,
            max_tokens=2000,
        )
        if resp.stop_reason != "tool_use":
            text = "".join(
                block.text for block in resp.content if hasattr(block, "text")
            )
            calls = [(b.name, b.input) for b in resp.content if b.type == "tool_use"]
            return text, calls

        tool_uses = [b for b in resp.content if b.type == "tool_use"]
        results = [
            {"type": "tool_result", "tool_use_id": tc.id, "content": "mock result"}
            for tc in tool_uses
        ]
        messages.append({"role": "assistant", "content": resp.content})
        messages.append({"role": "user",      "content": results})

    return "", []


def test_llm_calls_sql_query(llm_client):
    """LLM 应能调用 sql_query 工具查询设备状态。"""
    tools = [{
        "name": "sql_query",
        "description": "Execute SQL SELECT query on industrial database",
        "input_schema": {
            "type": "object",
            "properties": {"query": {"type": "string"}},
            "required": ["query"],
        },
    }]

    _, calls = _run_single_turn(
        llm_client,
        "Use sql_query: SELECT id, status FROM equipment WHERE status = 'fault'",
        tools,
    )

    sql_calls = [c for c in calls if c[0] == "sql_query"]
    assert len(sql_calls) >= 1, f"应调用 sql_query，实际：{calls}"


def test_llm_calls_rag_search(llm_client):
    """LLM 应能调用 rag_search 工具搜索文档。"""
    tools = [{
        "name": "rag_search",
        "description": "Search documents for fault reports and SOPs",
        "input_schema": {
            "type": "object",
            "properties": {"query": {"type": "string"}},
            "required": ["query"],
        },
    }]

    _, calls = _run_single_turn(
        llm_client,
        "Use rag_search to search for: 紧急停机操作流程",
        tools,
    )

    rag_calls = [c for c in calls if c[0] == "rag_search"]
    assert len(rag_calls) >= 1, f"应调用 rag_search，实际：{calls}"
