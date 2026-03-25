"""
Agent 工具定义。

以 Anthropic tool_use API 格式定义所有工具的名称、描述和参数 schema。
工具描述对 LLM 的决策质量有直接影响，应清晰、准确地表达工具的用途和约束。
"""

from industrial_retrieval.skills.runner import get_loader


def build_tools() -> list[dict]:
    """
    构建工具定义列表。

    每次调用时动态获取最新的技能描述，以保证工具描述与已加载技能同步。

    Returns:
        Anthropic tool_use 格式的工具定义列表。
    """
    skill_descriptions = get_loader().get_descriptions()

    return [
        _sql_query_tool(),
        _rag_search_tool(),
        _list_tables_tool(),
        _list_documents_tool(),
        _skill_tool(skill_descriptions),
    ]


# ── 工具定义辅助函数 ──────────────────────────────────────────────────────────

def _sql_query_tool() -> dict:
    return {
        "name": "sql_query",
        "description": (
            "执行 SQL SELECT 查询，检索设备台账、维护记录、故障代码或零件库存。\n"
            "只允许 SELECT 语句；使用 params 参数传入查询参数（防 SQL 注入）。"
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "query": {
                    "type": "string",
                    "description": "SQL SELECT 语句",
                },
                "params": {
                    "type": "array",
                    "items": {"type": "string"},
                    "description": "参数化查询参数列表，对应语句中的 ? 占位符",
                },
            },
            "required": ["query"],
        },
    }


def _rag_search_tool() -> dict:
    return {
        "name": "rag_search",
        "description": (
            "在文档库中全文检索，适合查找故障报告、操作规程（SOP）和设备手册。\n"
            "输入自然语言查询，返回 TF-IDF 相关度最高的文档片段。"
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "query": {
                    "type": "string",
                    "description": "自然语言搜索查询",
                },
                "top_k": {
                    "type": "integer",
                    "description": "返回文档数量上限（默认 3，最大 5）",
                },
                "full_content": {
                    "type": "boolean",
                    "description": "True 返回完整文档，False 返回摘要片段（默认 false）",
                },
            },
            "required": ["query"],
        },
    }


def _list_tables_tool() -> dict:
    return {
        "name": "list_tables",
        "description": (
            "查看数据库表结构。不传参数时列出所有表；"
            "传入 table_name 时显示该表的字段定义和示例数据。"
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "table_name": {
                    "type": "string",
                    "description": "（可选）要查看的表名",
                },
            },
        },
    }


def _list_documents_tool() -> dict:
    return {
        "name": "list_documents",
        "description": "列出文档库中所有可检索文档的清单和内容预览。",
        "input_schema": {
            "type": "object",
            "properties": {},
        },
    }


def _skill_tool(skill_descriptions: str) -> dict:
    return {
        "name": "Skill",
        "description": (
            f"加载领域知识技能，获取专业查询指导。\n\n"
            f"可用技能：\n{skill_descriptions}"
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "skill": {
                    "type": "string",
                    "description": "技能名称（如 industrial-retrieval）",
                },
            },
            "required": ["skill"],
        },
    }
