"""
文本分词模块。

提供适用于中英文混合工业文档的轻量级分词实现：
- 英文和数字保留完整词（小写化）
- 中文按字符切分（简化实现，无需分词库依赖）
"""

import re


def tokenize(text: str) -> list[str]:
    """
    对中英文混合文本进行分词。

    Args:
        text: 待分词文本。

    Returns:
        Token 列表（英文已小写化）。
    """
    tokens: list[str] = []

    # 英文单词、数字和带连字符/下划线的标识符
    tokens.extend(re.findall(r"[a-zA-Z0-9_\-]+", text.lower()))

    # 中文字符（Unicode CJK 统一汉字块）
    tokens.extend(re.findall(r"[\u4e00-\u9fff]", text))

    return tokens


def extract_snippet(content: str, query_tokens: list[str], max_chars: int = 500) -> str:
    """
    从文档中提取与查询词最相关的段落片段。

    策略：找到与查询词重叠最多的行，取其周围若干行作为上下文。

    Args:
        content:      文档全文。
        query_tokens: 查询词 token 列表。
        max_chars:    返回片段的最大字符数。

    Returns:
        最相关的文本片段（超长时截断并加 "..."）。
    """
    lines = content.splitlines()
    query_set = set(query_tokens)

    best_idx = 0
    best_overlap = -1

    for i, line in enumerate(lines):
        overlap = len(query_set & set(tokenize(line)))
        if overlap > best_overlap:
            best_overlap = overlap
            best_idx = i

    start = max(0, best_idx - 2)
    end   = min(len(lines), best_idx + 8)
    snippet = "\n".join(lines[start:end])

    if len(snippet) > max_chars:
        snippet = snippet[:max_chars] + "..."

    return snippet
