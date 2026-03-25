"""
TF-IDF 文档检索引擎。

无外部依赖，基于 Python 标准库实现，支持中英文混合文档检索。

算法：
  1. 加载文档 → 分词 → 构建词频索引
  2. 计算 IDF（逆文档频率，平滑处理）
  3. 查询时计算每份文档的 TF-IDF 得分之和
  4. 按得分排序，返回 Top-K 结果
"""

import math
from collections import Counter
from pathlib import Path

from industrial_retrieval.rag.tokenizer import extract_snippet, tokenize


class RAGIndex:
    """
    基于 TF-IDF 的轻量级文档检索引擎。

    使用方式：
        index = RAGIndex(docs_dir)
        results = index.search("液压系统紧急停机", top_k=3)
    """

    def __init__(self, docs_dir: Path) -> None:
        """
        初始化并构建索引。

        Args:
            docs_dir: 包含 .txt 文档的目录路径。
        """
        self.docs_dir = docs_dir
        self.documents: dict[str, str]        = {}  # filename → full text
        self.doc_tokens: dict[str, list[str]] = {}  # filename → tokens
        self.idf: dict[str, float]            = {}  # term → IDF score
        self._build_index()

    # ── 公共接口 ────────────────────────────────────────────────────────────

    def search(self, query: str, top_k: int = 3) -> list[dict]:
        """
        检索与查询最相关的文档片段。

        Args:
            query:  自然语言查询字符串。
            top_k:  最多返回的文档数量。

        Returns:
            结果列表，每项包含 filename、score、snippet、full_content。
            若无匹配（score=0）则返回空列表。
        """
        if not self.documents:
            return []

        query_tokens = tokenize(query)
        if not query_tokens:
            return []

        scored = self._score_documents(query_tokens)
        return [
            {
                "filename":     fname,
                "score":        round(score, 4),
                "snippet":      extract_snippet(self.documents[fname], query_tokens),
                "full_content": self.documents[fname],
            }
            for fname, score in scored[:top_k]
            if score > 0
        ]

    def list_documents(self) -> list[dict]:
        """
        列出所有已索引文档的基本信息。

        Returns:
            列表，每项包含 filename、size_chars、preview（前 100 字符）。
        """
        return [
            {
                "filename":   fname,
                "size_chars": len(content),
                "preview":    " ".join(content.splitlines()[:3])[:100],
            }
            for fname, content in sorted(self.documents.items())
        ]

    def reload(self) -> None:
        """重新扫描目录并重建索引（文档有增删时调用）。"""
        self.documents.clear()
        self.doc_tokens.clear()
        self.idf.clear()
        self._build_index()

    # ── 私有方法 ────────────────────────────────────────────────────────────

    def _build_index(self) -> None:
        """扫描目录，加载文档，计算 IDF。"""
        if not self.docs_dir.exists():
            return

        for path in sorted(self.docs_dir.glob("*.txt")):
            content = path.read_text(encoding="utf-8")
            self.documents[path.name]   = content
            self.doc_tokens[path.name]  = tokenize(content)

        self._compute_idf()

    def _compute_idf(self) -> None:
        """计算语料库的逆文档频率（加 1 平滑，防零除）。"""
        n = len(self.doc_tokens)
        if n == 0:
            return

        all_terms = {t for tokens in self.doc_tokens.values() for t in tokens}
        for term in all_terms:
            df = sum(1 for tokens in self.doc_tokens.values() if term in tokens)
            self.idf[term] = math.log((n + 1) / (df + 1)) + 1

    def _score_documents(self, query_tokens: list[str]) -> list[tuple[str, float]]:
        """计算每份文档对查询的 TF-IDF 得分并排序（降序）。"""
        scored = []
        for fname, tokens in self.doc_tokens.items():
            tf    = Counter(tokens)
            total = max(len(tokens), 1)
            score = sum(
                (tf.get(t, 0) / total) * self.idf.get(t, 0)
                for t in query_tokens
            )
            scored.append((fname, score))

        scored.sort(key=lambda x: x[1], reverse=True)
        return scored
