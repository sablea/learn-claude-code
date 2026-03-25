"""
RAG 模块测试。

覆盖：tokenizer（分词）、index（RAGIndex 构建和检索）、search（工具函数输出）。
"""

import pytest

from industrial_retrieval.rag.tokenizer import extract_snippet, tokenize
from industrial_retrieval.rag.index import RAGIndex
from industrial_retrieval.rag.documents import SAMPLE_DOCUMENTS, init_documents
from industrial_retrieval.rag.search import run_rag_search, run_list_documents


# ── 分词测试 ──────────────────────────────────────────────────────────────────

class TestTokenize:
    def test_extracts_english_words(self):
        tokens = tokenize("CNC001 hydraulic fault E003")
        assert "cnc001" in tokens
        assert "hydraulic" in tokens
        assert "e003" in tokens

    def test_extracts_chinese_characters(self):
        tokens = tokenize("液压压力机故障")
        assert len(tokens) > 0
        assert all(isinstance(t, str) for t in tokens)

    def test_mixed_chinese_english(self):
        tokens = tokenize("CNC001主轴过热E001报警")
        assert "cnc001" in tokens
        assert "e001" in tokens
        assert len(tokens) > 5

    def test_empty_string(self):
        assert tokenize("") == []

    def test_lowercase_normalization(self):
        tokens = tokenize("CNC001")
        assert "cnc001" in tokens
        assert "CNC001" not in tokens


class TestExtractSnippet:
    def test_returns_non_empty_string(self):
        content = "第一行\n第二行 CNC001\n第三行\n第四行"
        snippet = extract_snippet(content, ["cnc001"])
        assert len(snippet) > 0

    def test_respects_max_chars(self):
        content = "x" * 2000
        snippet = extract_snippet(content, ["x"], max_chars=100)
        assert len(snippet) <= 103  # 100 + "..."

    def test_single_line_content(self):
        snippet = extract_snippet("仅有一行", ["一"])
        assert "一行" in snippet


# ── RAGIndex 测试 ─────────────────────────────────────────────────────────────

class TestRAGIndex:
    def test_loads_documents(self, tmp_docs):
        index = RAGIndex(tmp_docs)
        assert len(index.documents) == len(SAMPLE_DOCUMENTS)

    def test_builds_idf(self, tmp_docs):
        index = RAGIndex(tmp_docs)
        assert len(index.idf) > 0

    def test_empty_directory(self, tmp_path):
        empty_dir = tmp_path / "empty"
        empty_dir.mkdir()
        index = RAGIndex(empty_dir)
        assert index.search("任何内容") == []

    def test_nonexistent_directory(self, tmp_path):
        index = RAGIndex(tmp_path / "nonexistent")
        assert index.search("任何内容") == []

    def test_search_returns_relevant_fault_report(self, tmp_docs):
        index = RAGIndex(tmp_docs)
        results = index.search("主轴过热 CNC 故障")
        assert len(results) > 0
        top_fname = results[0]["filename"]
        assert "fault_report" in top_fname

    def test_search_returns_relevant_sop(self, tmp_docs):
        index = RAGIndex(tmp_docs)
        results = index.search("液压系统紧急停机操作流程")
        assert len(results) > 0
        top_fname = results[0]["filename"].lower()
        is_relevant = "sop" in top_fname or "hydraulic" in top_fname or "emergency" in top_fname
        assert is_relevant, f"Top result '{top_fname}' should be a SOP or hydraulic document"

    def test_search_respects_top_k(self, tmp_docs):
        index = RAGIndex(tmp_docs)
        assert len(index.search("故障维修", top_k=2)) <= 2
        assert len(index.search("故障维修", top_k=3)) <= 3

    def test_search_no_results_for_gibberish(self, tmp_docs):
        index = RAGIndex(tmp_docs)
        results = index.search("zxqvwjkxqnonexistentterm99999")
        assert results == []

    def test_result_contains_required_keys(self, tmp_docs):
        index = RAGIndex(tmp_docs)
        results = index.search("故障")
        if results:
            for key in ("filename", "score", "snippet", "full_content"):
                assert key in results[0], f"结果缺少键：{key}"

    def test_reload_rebuilds_index(self, tmp_docs):
        index = RAGIndex(tmp_docs)
        original_count = len(index.documents)
        index.reload()
        assert len(index.documents) == original_count

    def test_list_documents(self, tmp_docs):
        index = RAGIndex(tmp_docs)
        docs = index.list_documents()
        assert len(docs) == len(SAMPLE_DOCUMENTS)
        for doc in docs:
            assert "filename" in doc
            assert "size_chars" in doc
            assert "preview" in doc


# ── 工具函数测试 ──────────────────────────────────────────────────────────────

class TestRunRagSearch:
    def test_returns_formatted_output(self, tmp_docs, monkeypatch):
        """通过 monkeypatch 替换全局索引，使用临时文档目录。"""
        from industrial_retrieval.rag import search as search_module
        search_module._index = RAGIndex(tmp_docs)

        result = run_rag_search("CNC故障报告")
        assert "搜索" in result or "找到" in result

    def test_no_results_message(self, tmp_docs, monkeypatch):
        from industrial_retrieval.rag import search as search_module
        search_module._index = RAGIndex(tmp_docs)

        result = run_rag_search("zxqvwjkxqnonexistentterm99999")
        assert "未找到" in result

    def test_top_k_clamped_to_max(self, tmp_docs, monkeypatch):
        from industrial_retrieval.rag import search as search_module
        search_module._index = RAGIndex(tmp_docs)

        # top_k=100 不应崩溃，仍返回不超过文档总数的结果
        result = run_rag_search("故障", top_k=100)
        assert isinstance(result, str)


class TestRunListDocuments:
    def test_lists_sample_documents(self, tmp_docs, monkeypatch):
        from industrial_retrieval.rag import search as search_module
        search_module._index = RAGIndex(tmp_docs)

        result = run_list_documents()
        assert "fault_report" in result
        assert "sop" in result

    def test_empty_corpus(self, tmp_path, monkeypatch):
        from industrial_retrieval.rag import search as search_module
        empty_dir = tmp_path / "empty"
        empty_dir.mkdir()
        search_module._index = RAGIndex(empty_dir)

        result = run_list_documents()
        assert "空" in result
