"""
数据库模式与初始化。

负责创建 SQLite 表结构并填充示例数据（幂等操作，已存在则跳过）。
"""

import sqlite3
from pathlib import Path

from industrial_retrieval.database.sample_data import (
    EQUIPMENT,
    FAULT_CODES,
    MAINTENANCE_RECORDS,
    PARTS_INVENTORY,
)

_CREATE_TABLES_SQL = """
    CREATE TABLE IF NOT EXISTS equipment (
        id                    TEXT PRIMARY KEY,
        name                  TEXT NOT NULL,
        model                 TEXT,
        location              TEXT,
        status                TEXT,
        install_date          TEXT,
        last_maintenance_date TEXT
    );

    CREATE TABLE IF NOT EXISTS maintenance_records (
        id              INTEGER PRIMARY KEY AUTOINCREMENT,
        equipment_id    TEXT,
        date            TEXT,
        type            TEXT,
        description     TEXT,
        technician      TEXT,
        cost            REAL,
        duration_hours  REAL
    );

    CREATE TABLE IF NOT EXISTS fault_codes (
        code               TEXT PRIMARY KEY,
        equipment_type     TEXT,
        description        TEXT,
        severity           TEXT,
        recommended_action TEXT
    );

    CREATE TABLE IF NOT EXISTS parts_inventory (
        part_id              TEXT PRIMARY KEY,
        name                 TEXT,
        quantity             INTEGER,
        compatible_equipment TEXT,
        reorder_point        INTEGER
    );
"""


def init_database(db_path: Path) -> None:
    """
    创建数据库表并填充示例数据。

    幂等操作：表已存在或数据已存在时跳过，安全重复调用。

    Args:
        db_path: SQLite 数据库文件路径（父目录会自动创建）。
    """
    db_path.parent.mkdir(parents=True, exist_ok=True)

    with sqlite3.connect(db_path) as conn:
        conn.executescript(_CREATE_TABLES_SQL)
        _seed_if_empty(conn)
        conn.commit()


def _seed_if_empty(conn: sqlite3.Connection) -> None:
    """仅在表为空时插入示例数据。"""
    c = conn.cursor()

    if not c.execute("SELECT 1 FROM equipment LIMIT 1").fetchone():
        c.executemany("INSERT INTO equipment VALUES (?,?,?,?,?,?,?)", EQUIPMENT)

    if not c.execute("SELECT 1 FROM maintenance_records LIMIT 1").fetchone():
        c.executemany(
            "INSERT INTO maintenance_records "
            "(equipment_id, date, type, description, technician, cost, duration_hours) "
            "VALUES (?,?,?,?,?,?,?)",
            MAINTENANCE_RECORDS,
        )

    if not c.execute("SELECT 1 FROM fault_codes LIMIT 1").fetchone():
        c.executemany("INSERT INTO fault_codes VALUES (?,?,?,?,?)", FAULT_CODES)

    if not c.execute("SELECT 1 FROM parts_inventory LIMIT 1").fetchone():
        c.executemany("INSERT INTO parts_inventory VALUES (?,?,?,?,?)", PARTS_INVENTORY)
