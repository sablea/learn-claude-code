"""
SQL 查询工具。

提供面向 Agent 工具调用的只读 SQL 查询接口，以及数据库表结构查看功能。
所有查询操作均为只读（SELECT），禁止任何数据修改操作。
"""

import sqlite3
from pathlib import Path

from industrial_retrieval.config import DB_PATH

# 禁止的 SQL 关键词（防止绕过 SELECT 检查）
_FORBIDDEN_KEYWORDS = frozenset(
    ["DROP", "DELETE", "UPDATE", "INSERT", "ALTER", "CREATE", "ATTACH"]
)


def run_sql_query(query: str, params: list | None = None, db_path: Path | None = None) -> str:
    """
    执行只读 SQL 查询并返回格式化文本表格。

    安全约束：
      - 仅允许 SELECT 语句
      - 阻止包含 DDL/DML 关键词的查询

    Args:
        query:   SQL SELECT 语句。
        params:  参数化查询的参数列表（防 SQL 注入）。
        db_path: 数据库文件路径（默认使用全局配置）。

    Returns:
        格式化文本表格，或"查询结果为空"，或错误消息。
    """
    if db_path is None:
        db_path = DB_PATH
    normalized = query.strip().upper()

    if not normalized.startswith("SELECT"):
        return "错误：只允许执行 SELECT 查询，不允许修改数据。"

    if any(kw in normalized for kw in _FORBIDDEN_KEYWORDS):
        return "错误：查询包含不允许的操作关键词。"

    try:
        with sqlite3.connect(db_path) as conn:
            conn.row_factory = sqlite3.Row
            rows = conn.execute(query, params or []).fetchall()

        if not rows:
            return "查询结果为空。"

        return _format_table(rows)

    except sqlite3.Error as e:
        return f"SQL 执行错误：{e}"


def run_list_tables(table_name: str | None = None, db_path: Path | None = None) -> str:
    """
    列出数据库表结构，可选查看单张表的详细信息。

    Args:
        table_name: 若提供则显示该表的字段结构和示例数据，否则列出全部表。
        db_path:    数据库文件路径（默认使用全局配置）。
    """
    if db_path is None:
        db_path = DB_PATH
    try:
        with sqlite3.connect(db_path) as conn:
            conn.row_factory = sqlite3.Row
            if table_name:
                return _describe_table(conn, table_name)
            return _list_all_tables(conn)
    except Exception as e:
        return f"错误：{e}"


# ── 私有格式化辅助函数 ────────────────────────────────────────────────────────

def _format_table(rows: list[sqlite3.Row]) -> str:
    """将查询结果格式化为人类可读的文本表格。"""
    columns = rows[0].keys()
    widths = {col: len(col) for col in columns}
    for row in rows:
        for col in columns:
            widths[col] = max(widths[col], len(str(row[col] or "")))

    header    = " | ".join(col.ljust(widths[col]) for col in columns)
    separator = "-+-".join("-" * widths[col] for col in columns)
    data_rows = [
        " | ".join(str(row[col] or "").ljust(widths[col]) for col in columns)
        for row in rows
    ]

    return "\n".join([header, separator, *data_rows, f"\n共 {len(rows)} 条记录"])


def _describe_table(conn: sqlite3.Connection, table_name: str) -> str:
    """返回指定表的字段结构和前 3 行示例数据。"""
    cols = conn.execute(f"PRAGMA table_info({table_name})").fetchall()
    if not cols:
        return f"表 '{table_name}' 不存在。"

    lines = [f"表：{table_name}", "-" * 40,
             f"{'列名':<25} {'类型':<10} {'非空':<6} {'默认值'}",
             "-" * 60]
    for col in cols:
        not_null = "YES" if col["notnull"] else "NO"
        default  = col["dflt_value"] or ""
        lines.append(f"{col['name']:<25} {col['type']:<10} {not_null:<6} {default}")

    samples = conn.execute(f"SELECT * FROM {table_name} LIMIT 3").fetchall()
    if samples:
        keys = samples[0].keys()
        lines.append(f"\n示例数据（前 {len(samples)} 条）：")
        lines.append(" | ".join(str(k)[:15] for k in keys))
        for row in samples:
            lines.append(" | ".join(str(row[k] or "")[:15] for k in keys))

    return "\n".join(lines)


def _list_all_tables(conn: sqlite3.Connection) -> str:
    """列出所有表的名称、记录数和字段列表。"""
    tables = [
        row[0] for row in
        conn.execute("SELECT name FROM sqlite_master WHERE type='table' ORDER BY name")
    ]

    lines = ["数据库中的表：", "=" * 40]
    for tbl in tables:
        count     = conn.execute(f"SELECT COUNT(*) FROM {tbl}").fetchone()[0]
        col_names = [col[1] for col in conn.execute(f"PRAGMA table_info({tbl})")]
        lines.append(f"\n📋 {tbl}（{count} 条记录）")
        lines.append(f"   字段：{', '.join(col_names)}")

    return "\n".join(lines)
