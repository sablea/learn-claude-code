"""
Tests for industrial_retrieval_agent.py - 工业数据检索 Agent

Tests cover:
  - Database initialization and schema
  - RAG index construction and TF-IDF search
  - SQL query execution (read-only safety)
  - Tool dispatch and output format
  - Skill loader integration
  - LLM integration tests (skipped without API key)
"""

import os
import sys
import sqlite3
import tempfile
from pathlib import Path

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from industrial_retrieval_agent import (
    RAGIndex,
    SkillLoader,
    _tokenize,
    execute_tool,
    init_database,
    init_documents,
    run_list_documents,
    run_list_tables,
    run_rag_search,
    run_sql_query,
    run_skill,
    SAMPLE_DOCUMENTS,
    SAMPLE_EQUIPMENT,
    SAMPLE_FAULT_CODES,
    SYSTEM,
)
from tests.helpers import get_client, run_agent, run_tests


# =============================================================================
# 单元测试：分词
# =============================================================================


def test_tokenize_chinese():
    tokens = _tokenize("液压压力机故障")
    assert "液" in tokens or len(tokens) > 0, "应能对中文分词"
    print("PASS: test_tokenize_chinese")
    return True


def test_tokenize_english():
    tokens = _tokenize("CNC001 hydraulic fault E003")
    assert "cnc001" in tokens
    assert "hydraulic" in tokens
    assert "e003" in tokens
    print("PASS: test_tokenize_english")
    return True


def test_tokenize_mixed():
    tokens = _tokenize("CNC001主轴过热E001报警")
    assert "cnc001" in tokens
    assert "e001" in tokens
    assert len(tokens) > 5
    print("PASS: test_tokenize_mixed")
    return True


# =============================================================================
# 单元测试：数据库初始化
# =============================================================================


def test_db_init_creates_tables():
    with tempfile.TemporaryDirectory() as tmpdir:
        db_path = Path(tmpdir) / "test.db"
        init_database(db_path)

        conn = sqlite3.connect(db_path)
        c = conn.cursor()
        c.execute("SELECT name FROM sqlite_master WHERE type='table'")
        tables = {row[0] for row in c.fetchall()}
        conn.close()

        assert "equipment" in tables
        assert "maintenance_records" in tables
        assert "fault_codes" in tables
        assert "parts_inventory" in tables
    print("PASS: test_db_init_creates_tables")
    return True


def test_db_init_populates_data():
    with tempfile.TemporaryDirectory() as tmpdir:
        db_path = Path(tmpdir) / "test.db"
        init_database(db_path)

        conn = sqlite3.connect(db_path)
        c = conn.cursor()
        c.execute("SELECT COUNT(*) FROM equipment")
        eq_count = c.fetchone()[0]
        c.execute("SELECT COUNT(*) FROM fault_codes")
        fc_count = c.fetchone()[0]
        conn.close()

        assert eq_count == len(SAMPLE_EQUIPMENT), f"设备数量不符：{eq_count}"
        assert fc_count == len(SAMPLE_FAULT_CODES), f"故障代码数量不符：{fc_count}"
    print("PASS: test_db_init_populates_data")
    return True


def test_db_init_idempotent():
    """多次初始化不应重复插入数据。"""
    with tempfile.TemporaryDirectory() as tmpdir:
        db_path = Path(tmpdir) / "test.db"
        init_database(db_path)
        init_database(db_path)  # 第二次

        conn = sqlite3.connect(db_path)
        c = conn.cursor()
        c.execute("SELECT COUNT(*) FROM equipment")
        count = c.fetchone()[0]
        conn.close()

        assert count == len(SAMPLE_EQUIPMENT), f"重复初始化导致数据重复：{count}"
    print("PASS: test_db_init_idempotent")
    return True


# =============================================================================
# 单元测试：SQL 查询
# =============================================================================


def test_sql_query_select_all_equipment():
    result = run_sql_query("SELECT id, name, status FROM equipment")
    assert "CNC001" in result
    assert "operational" in result or "fault" in result
    print("PASS: test_sql_query_select_all_equipment")
    return True


def test_sql_query_with_where_clause():
    result = run_sql_query("SELECT id, name FROM equipment WHERE status = 'fault'")
    assert "CNC002" in result
    print("PASS: test_sql_query_with_where_clause")
    return True


def test_sql_query_with_params():
    result = run_sql_query(
        "SELECT code, description FROM fault_codes WHERE code = ?",
        params=["E003"]
    )
    assert "E003" in result
    assert "主轴驱动故障" in result
    print("PASS: test_sql_query_with_params")
    return True


def test_sql_query_blocks_non_select():
    result = run_sql_query("DELETE FROM equipment WHERE id = 'CNC001'")
    assert "错误" in result or "Error" in result.lower()
    print("PASS: test_sql_query_blocks_non_select")
    return True


def test_sql_query_blocks_dangerous_keywords():
    result = run_sql_query("SELECT * FROM equipment; DROP TABLE equipment;--")
    # The query contains DROP so should be blocked
    assert "错误" in result or "Error" in result.lower() or "不允许" in result
    print("PASS: test_sql_query_blocks_dangerous_keywords")
    return True


def test_sql_query_empty_result():
    result = run_sql_query("SELECT * FROM equipment WHERE id = 'NONEXISTENT'")
    assert "空" in result or "empty" in result.lower() or "0" in result
    print("PASS: test_sql_query_empty_result")
    return True


def test_sql_query_aggregation():
    result = run_sql_query(
        "SELECT type, COUNT(*) as cnt FROM maintenance_records GROUP BY type"
    )
    assert "preventive" in result or "corrective" in result
    print("PASS: test_sql_query_aggregation")
    return True


# =============================================================================
# 单元测试：RAG 文档索引
# =============================================================================


def test_rag_index_loads_documents():
    with tempfile.TemporaryDirectory() as tmpdir:
        docs_dir = Path(tmpdir)
        init_documents(docs_dir)
        index = RAGIndex(docs_dir)

        assert len(index.documents) == len(SAMPLE_DOCUMENTS)
        assert len(index.idf) > 0
    print("PASS: test_rag_index_loads_documents")
    return True


def test_rag_index_empty_dir():
    with tempfile.TemporaryDirectory() as tmpdir:
        docs_dir = Path(tmpdir) / "empty"
        docs_dir.mkdir()
        index = RAGIndex(docs_dir)

        assert len(index.documents) == 0
        result = index.search("anything")
        assert result == []
    print("PASS: test_rag_index_empty_dir")
    return True


def test_rag_search_returns_relevant():
    with tempfile.TemporaryDirectory() as tmpdir:
        docs_dir = Path(tmpdir)
        init_documents(docs_dir)
        index = RAGIndex(docs_dir)

        results = index.search("主轴过热 CNC 故障")
        assert len(results) > 0
        # 故障报告应该排在前面
        top_fname = results[0]["filename"]
        assert "fault_report" in top_fname or "cnc" in top_fname.lower()
    print("PASS: test_rag_search_returns_relevant")
    return True


def test_rag_search_sop_query():
    with tempfile.TemporaryDirectory() as tmpdir:
        docs_dir = Path(tmpdir)
        init_documents(docs_dir)
        index = RAGIndex(docs_dir)

        results = index.search("液压系统紧急停机操作流程")
        assert len(results) > 0
        top_fname = results[0]["filename"].lower()
        is_relevant = "sop" in top_fname or "hydraulic" in top_fname or "emergency" in top_fname
        assert is_relevant, f"Top result '{top_fname}' should be a SOP or hydraulic document"
    print("PASS: test_rag_search_sop_query")
    return True


def test_rag_search_top_k_limit():
    with tempfile.TemporaryDirectory() as tmpdir:
        docs_dir = Path(tmpdir)
        init_documents(docs_dir)
        index = RAGIndex(docs_dir)

        results_3 = index.search("故障维修", top_k=3)
        results_2 = index.search("故障维修", top_k=2)

        assert len(results_3) <= 3
        assert len(results_2) <= 2
    print("PASS: test_rag_search_top_k_limit")
    return True


def test_rag_search_snippet_included():
    with tempfile.TemporaryDirectory() as tmpdir:
        docs_dir = Path(tmpdir)
        init_documents(docs_dir)
        index = RAGIndex(docs_dir)

        results = index.search("预防性维护 CNC")
        if results:
            assert "snippet" in results[0]
            assert "full_content" in results[0]
            assert len(results[0]["snippet"]) > 0
    print("PASS: test_rag_search_snippet_included")
    return True


# =============================================================================
# 单元测试：run_rag_search 工具函数
# =============================================================================


def test_run_rag_search_output_format():
    result = run_rag_search("CNC故障报告")
    assert "搜索" in result or "找到" in result or "未找到" in result
    print("PASS: test_run_rag_search_output_format")
    return True


def test_run_rag_search_no_results():
    # Use pure ASCII gibberish that cannot match any document token
    result = run_rag_search("zxqvwjkxqzxqvwjkxqzxqvwjkxqnonexistentterm999999")
    assert "未找到" in result
    print("PASS: test_run_rag_search_no_results")
    return True


# =============================================================================
# 单元测试：list_tables / list_documents
# =============================================================================


def test_list_tables_all():
    result = run_list_tables()
    assert "equipment" in result
    assert "maintenance_records" in result
    assert "fault_codes" in result
    assert "parts_inventory" in result
    print("PASS: test_list_tables_all")
    return True


def test_list_tables_specific():
    result = run_list_tables("equipment")
    assert "id" in result
    assert "name" in result
    assert "status" in result
    print("PASS: test_list_tables_specific")
    return True


def test_list_tables_nonexistent():
    result = run_list_tables("no_such_table")
    assert "不存在" in result or "exist" in result.lower()
    print("PASS: test_list_tables_nonexistent")
    return True


def test_list_documents():
    result = run_list_documents()
    assert "fault_report" in result
    assert "sop" in result
    print("PASS: test_list_documents")
    return True


# =============================================================================
# 单元测试：Skill 加载
# =============================================================================


def test_skill_loader_finds_industrial_skill():
    """验证 industrial-retrieval 技能已正确加载。"""
    skills_dir = Path(__file__).parent.parent / "skills"
    loader = SkillLoader(skills_dir)
    assert "industrial-retrieval" in loader.skills, (
        "industrial-retrieval 技能未找到，请确认 skills/industrial-retrieval/SKILL.md 存在"
    )
    print("PASS: test_skill_loader_finds_industrial_skill")
    return True


def test_skill_content_not_in_system_prompt():
    """验证技能内容不在系统提示词中（缓存友好）。"""
    skill_body_marker = "SQL查询最佳实践"
    assert skill_body_marker not in SYSTEM, (
        "技能正文不应出现在系统提示词中（破坏缓存）"
    )
    print("PASS: test_skill_content_not_in_system_prompt")
    return True


def test_run_skill_loads_content():
    result = run_skill("industrial-retrieval")
    assert "<skill-loaded" in result
    assert "industrial-retrieval" in result
    assert "数据库" in result or "SQL" in result
    print("PASS: test_run_skill_loads_content")
    return True


def test_run_skill_unknown():
    result = run_skill("nonexistent-skill")
    assert "不存在" in result or "not found" in result.lower()
    print("PASS: test_run_skill_unknown")
    return True


# =============================================================================
# 单元测试：工具分发
# =============================================================================


def test_execute_tool_sql_query():
    result = execute_tool("sql_query", {"query": "SELECT COUNT(*) as total FROM equipment"})
    assert "total" in result
    assert str(len(SAMPLE_EQUIPMENT)) in result
    print("PASS: test_execute_tool_sql_query")
    return True


def test_execute_tool_rag_search():
    result = execute_tool("rag_search", {"query": "液压泵维修"})
    assert isinstance(result, str)
    assert len(result) > 0
    print("PASS: test_execute_tool_rag_search")
    return True


def test_execute_tool_list_tables():
    result = execute_tool("list_tables", {})
    assert "equipment" in result
    print("PASS: test_execute_tool_list_tables")
    return True


def test_execute_tool_list_documents():
    result = execute_tool("list_documents", {})
    assert "fault_report" in result
    print("PASS: test_execute_tool_list_documents")
    return True


def test_execute_tool_unknown():
    result = execute_tool("unknown_tool", {})
    assert "未知" in result or "Unknown" in result
    print("PASS: test_execute_tool_unknown")
    return True


# =============================================================================
# LLM 集成测试（需要 API Key）
# =============================================================================


def test_llm_sql_query_equipment_status():
    """LLM 能正确使用 sql_query 工具查询设备状态。"""
    client = get_client()
    if not client:
        print("SKIP: No API key")
        return True

    tools = [
        {
            "name": "sql_query",
            "description": "Execute SQL SELECT query on industrial database",
            "input_schema": {
                "type": "object",
                "properties": {
                    "query": {"type": "string"},
                    "params": {"type": "array", "items": {"type": "string"}},
                },
                "required": ["query"],
            },
        }
    ]

    text, calls, _ = run_agent(
        client,
        "Query the equipment table and find all equipment with status 'fault'. "
        "Use sql_query tool with: SELECT id, name, status FROM equipment WHERE status = 'fault'",
        tools,
        system="You are a data retrieval agent. Use the sql_query tool to answer questions.",
    )

    sql_calls = [c for c in calls if c[0] == "sql_query"]
    assert len(sql_calls) >= 1, f"应调用 sql_query，实际调用：{calls}"
    print("PASS: test_llm_sql_query_equipment_status")
    return True


def test_llm_rag_search_sop():
    """LLM 能正确使用 rag_search 工具查询 SOP 文档。"""
    client = get_client()
    if not client:
        print("SKIP: No API key")
        return True

    tools = [
        {
            "name": "rag_search",
            "description": "Search documents using TF-IDF similarity",
            "input_schema": {
                "type": "object",
                "properties": {
                    "query": {"type": "string"},
                    "top_k": {"type": "integer"},
                },
                "required": ["query"],
            },
        }
    ]

    text, calls, _ = run_agent(
        client,
        "Use rag_search to find documents about emergency shutdown procedures. "
        "Search for: '紧急停机 操作流程'",
        tools,
        system="You are a document retrieval agent. Use rag_search to find relevant documents.",
    )

    rag_calls = [c for c in calls if c[0] == "rag_search"]
    assert len(rag_calls) >= 1, f"应调用 rag_search，实际调用：{calls}"
    print("PASS: test_llm_rag_search_sop")
    return True


# =============================================================================
# 测试运行器
# =============================================================================


if __name__ == "__main__":
    sys.exit(0 if run_tests([
        # 分词
        test_tokenize_chinese,
        test_tokenize_english,
        test_tokenize_mixed,
        # 数据库
        test_db_init_creates_tables,
        test_db_init_populates_data,
        test_db_init_idempotent,
        # SQL 查询
        test_sql_query_select_all_equipment,
        test_sql_query_with_where_clause,
        test_sql_query_with_params,
        test_sql_query_blocks_non_select,
        test_sql_query_blocks_dangerous_keywords,
        test_sql_query_empty_result,
        test_sql_query_aggregation,
        # RAG 索引
        test_rag_index_loads_documents,
        test_rag_index_empty_dir,
        test_rag_search_returns_relevant,
        test_rag_search_sop_query,
        test_rag_search_top_k_limit,
        test_rag_search_snippet_included,
        test_run_rag_search_output_format,
        test_run_rag_search_no_results,
        # 表/文档列表
        test_list_tables_all,
        test_list_tables_specific,
        test_list_tables_nonexistent,
        test_list_documents,
        # Skill 加载
        test_skill_loader_finds_industrial_skill,
        test_skill_content_not_in_system_prompt,
        test_run_skill_loads_content,
        test_run_skill_unknown,
        # 工具分发
        test_execute_tool_sql_query,
        test_execute_tool_rag_search,
        test_execute_tool_list_tables,
        test_execute_tool_list_documents,
        test_execute_tool_unknown,
        # LLM 集成
        test_llm_sql_query_equipment_status,
        test_llm_rag_search_sop,
    ]) else 1)
