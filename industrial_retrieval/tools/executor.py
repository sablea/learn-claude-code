"""
工具执行分发器。

将 LLM 的工具调用请求路由到对应的实现函数。
每个工具名映射一个纯函数，清晰、可测试。
"""

from industrial_retrieval.database.query import run_list_tables, run_sql_query
from industrial_retrieval.rag.search import run_list_documents, run_rag_search
from industrial_retrieval.skills.runner import run_skill

# 工具名 → 处理函数的映射（参数由调用方展开）
_TOOL_HANDLERS: dict = {
    "sql_query":      lambda args: run_sql_query(
                          args["query"], args.get("params")
                      ),
    "rag_search":     lambda args: run_rag_search(
                          args["query"],
                          top_k=args.get("top_k", 3),
                          full_content=args.get("full_content", False),
                      ),
    "list_tables":    lambda args: run_list_tables(args.get("table_name")),
    "list_documents": lambda _:    run_list_documents(),
    "Skill":          lambda args: run_skill(args["skill"]),
}


def execute_tool(name: str, args: dict) -> str:
    """
    根据工具名称执行对应的工具函数。

    Args:
        name: 工具名称（与工具定义中的 name 字段一致）。
        args: 工具参数字典（由 LLM 生成）。

    Returns:
        工具执行结果字符串，供 LLM 读取。
    """
    handler = _TOOL_HANDLERS.get(name)
    if handler is None:
        return f"未知工具：{name}"

    try:
        return handler(args)
    except KeyError as e:
        return f"工具 '{name}' 缺少必需参数：{e}"
    except Exception as e:
        return f"工具 '{name}' 执行错误：{e}"
