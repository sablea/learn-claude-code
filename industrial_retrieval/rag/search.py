"""
RAG 检索工具函数。

将 RAGIndex 封装为面向 Agent 工具调用的接口，
返回格式化文本供 LLM 理解。
"""

from industrial_retrieval.config import DOCS_DIR
from industrial_retrieval.rag.index import RAGIndex

# 模块级索引单例（延迟初始化）
_index: RAGIndex | None = None


def get_index() -> RAGIndex:
    """
    获取（或懒加载初始化）全局 RAGIndex 单例。

    首次调用时从磁盘加载文档并构建 TF-IDF 索引。
    """
    global _index
    if _index is None:
        _index = RAGIndex(DOCS_DIR)
    return _index


def run_rag_search(
    query: str,
    top_k: int = 3,
    full_content: bool = False,
) -> str:
    """
    搜索文档库，返回与查询最相关的内容。

    Args:
        query:        自然语言搜索查询。
        top_k:        返回文档数量上限（1-5）。
        full_content: True 返回完整文档，False 返回摘要片段。

    Returns:
        格式化检索结果字符串，供 LLM 阅读。
    """
    top_k   = min(max(1, top_k), 5)
    results = get_index().search(query, top_k=top_k)

    if not results:
        return f"未找到与 '{query}' 相关的文档。"

    lines = [f"搜索 '{query}' 找到 {len(results)} 个相关文档：\n"]
    for i, r in enumerate(results, 1):
        lines.append("=" * 60)
        lines.append(f"[{i}] {r['filename']}  （相关度：{r['score']}）")
        lines.append("-" * 60)
        lines.append(r["full_content"] if full_content else r["snippet"])

    lines.append("=" * 60)
    return "\n".join(lines)


def run_list_documents() -> str:
    """
    列出文档库中所有已索引文档的清单和预览。

    Returns:
        格式化文档列表字符串。
    """
    docs = get_index().list_documents()

    if not docs:
        return "文档库为空。"

    lines = [f"文档库共 {len(docs)} 个文档：", "=" * 60]
    for doc in docs:
        lines.append(f"\n📄 {doc['filename']}（{doc['size_chars']} 字符）")
        lines.append(f"   {doc['preview']}...")

    return "\n".join(lines)
