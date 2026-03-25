"""
Tools 模块测试。

覆盖：executor（execute_tool 分发）、definitions（build_tools 输出格式）。
"""

import pytest

from industrial_retrieval.tools.executor import execute_tool
from industrial_retrieval.tools.definitions import build_tools
from industrial_retrieval.database.sample_data import EQUIPMENT


# ── execute_tool 分发测试 ─────────────────────────────────────────────────────

class TestExecuteTool:
    def test_sql_query_dispatches_correctly(self, tmp_db, monkeypatch):
        # 将全局 DB_PATH 指向临时数据库
        import industrial_retrieval.database.query as query_module
        monkeypatch.setattr(query_module, "DB_PATH", tmp_db)

        result = execute_tool(
            "sql_query",
            {"query": "SELECT COUNT(*) as total FROM equipment"},
        )
        assert "total" in result
        assert str(len(EQUIPMENT)) in result

    def test_list_tables_dispatches_correctly(self, tmp_db, monkeypatch):
        import industrial_retrieval.database.query as query_module
        monkeypatch.setattr(query_module, "DB_PATH", tmp_db)

        result = execute_tool("list_tables", {})
        assert "equipment" in result

    def test_rag_search_dispatches_correctly(self, tmp_docs, monkeypatch):
        from industrial_retrieval.rag import search as search_module
        from industrial_retrieval.rag.index import RAGIndex
        search_module._index = RAGIndex(tmp_docs)

        result = execute_tool("rag_search", {"query": "液压泵维修"})
        assert isinstance(result, str)
        assert len(result) > 0

    def test_list_documents_dispatches_correctly(self, tmp_docs, monkeypatch):
        from industrial_retrieval.rag import search as search_module
        from industrial_retrieval.rag.index import RAGIndex
        search_module._index = RAGIndex(tmp_docs)

        result = execute_tool("list_documents", {})
        assert "fault_report" in result

    def test_skill_dispatches_correctly(self, tmp_skills, monkeypatch):
        from industrial_retrieval.skills import runner as runner_module
        from industrial_retrieval.skills.loader import SkillLoader
        runner_module._loader = SkillLoader(tmp_skills)

        result = execute_tool("Skill", {"skill": "test-skill"})
        assert "<skill-loaded" in result

    def test_unknown_tool_returns_error(self):
        result = execute_tool("nonexistent_tool", {})
        assert "未知工具" in result

    def test_missing_required_param_returns_error(self, tmp_db, monkeypatch):
        import industrial_retrieval.database.query as query_module
        monkeypatch.setattr(query_module, "DB_PATH", tmp_db)

        # sql_query 需要 query 参数
        result = execute_tool("sql_query", {})
        assert "缺少" in result or "错误" in result or "error" in result.lower()


# ── build_tools 测试 ──────────────────────────────────────────────────────────

class TestBuildTools:
    def test_returns_list(self):
        tools = build_tools()
        assert isinstance(tools, list)

    def test_contains_expected_tools(self):
        tool_names = {t["name"] for t in build_tools()}
        expected = {"sql_query", "rag_search", "list_tables", "list_documents", "Skill"}
        assert expected <= tool_names

    def test_each_tool_has_required_fields(self):
        for tool in build_tools():
            assert "name" in tool, f"工具缺少 name 字段：{tool}"
            assert "description" in tool, f"工具缺少 description 字段：{tool}"
            assert "input_schema" in tool, f"工具缺少 input_schema 字段：{tool}"

    def test_each_tool_schema_is_valid(self):
        for tool in build_tools():
            schema = tool["input_schema"]
            assert schema.get("type") == "object", \
                f"工具 {tool['name']} 的 input_schema type 应为 object"
            assert "properties" in schema, \
                f"工具 {tool['name']} 的 input_schema 缺少 properties"
