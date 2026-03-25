"""
数据库模块测试。

覆盖：schema（表创建和幂等性）、sample_data（常量正确性）、query（SQL 查询和安全性）。
"""

import sqlite3

import pytest

from industrial_retrieval.database.sample_data import (
    EQUIPMENT,
    FAULT_CODES,
    MAINTENANCE_RECORDS,
    PARTS_INVENTORY,
)
from industrial_retrieval.database.schema import init_database
from industrial_retrieval.database.query import run_sql_query, run_list_tables


# ── schema 测试 ───────────────────────────────────────────────────────────────

class TestInitDatabase:
    def test_creates_all_tables(self, tmp_db):
        with sqlite3.connect(tmp_db) as conn:
            tables = {
                row[0] for row in
                conn.execute("SELECT name FROM sqlite_master WHERE type='table'")
            }
        assert {"equipment", "maintenance_records", "fault_codes", "parts_inventory"} <= tables

    def test_seeds_equipment_data(self, tmp_db):
        with sqlite3.connect(tmp_db) as conn:
            count = conn.execute("SELECT COUNT(*) FROM equipment").fetchone()[0]
        assert count == len(EQUIPMENT)

    def test_seeds_fault_codes_data(self, tmp_db):
        with sqlite3.connect(tmp_db) as conn:
            count = conn.execute("SELECT COUNT(*) FROM fault_codes").fetchone()[0]
        assert count == len(FAULT_CODES)

    def test_seeds_maintenance_records(self, tmp_db):
        with sqlite3.connect(tmp_db) as conn:
            count = conn.execute("SELECT COUNT(*) FROM maintenance_records").fetchone()[0]
        assert count == len(MAINTENANCE_RECORDS)

    def test_seeds_parts_inventory(self, tmp_db):
        with sqlite3.connect(tmp_db) as conn:
            count = conn.execute("SELECT COUNT(*) FROM parts_inventory").fetchone()[0]
        assert count == len(PARTS_INVENTORY)

    def test_idempotent_multiple_calls(self, tmp_path):
        db_path = tmp_path / "idempotent.db"
        init_database(db_path)
        init_database(db_path)  # 第二次调用不应重复插入

        with sqlite3.connect(db_path) as conn:
            count = conn.execute("SELECT COUNT(*) FROM equipment").fetchone()[0]
        assert count == len(EQUIPMENT)

    def test_creates_parent_directory(self, tmp_path):
        db_path = tmp_path / "nested" / "dir" / "test.db"
        init_database(db_path)
        assert db_path.exists()


# ── sample_data 测试 ──────────────────────────────────────────────────────────

class TestSampleData:
    def test_equipment_tuples_have_seven_fields(self):
        for row in EQUIPMENT:
            assert len(row) == 7, f"equipment 行字段数错误：{row}"

    def test_fault_codes_tuples_have_five_fields(self):
        for row in FAULT_CODES:
            assert len(row) == 5, f"fault_codes 行字段数错误：{row}"

    def test_maintenance_records_tuples_have_seven_fields(self):
        for row in MAINTENANCE_RECORDS:
            assert len(row) == 7, f"maintenance_records 行字段数错误：{row}"

    def test_parts_inventory_tuples_have_five_fields(self):
        for row in PARTS_INVENTORY:
            assert len(row) == 5, f"parts_inventory 行字段数错误：{row}"

    def test_equipment_ids_unique(self):
        ids = [row[0] for row in EQUIPMENT]
        assert len(ids) == len(set(ids)), "设备 ID 有重复"

    def test_fault_codes_unique(self):
        codes = [row[0] for row in FAULT_CODES]
        assert len(codes) == len(set(codes)), "故障代码有重复"


# ── SQL 查询测试 ──────────────────────────────────────────────────────────────

class TestRunSqlQuery:
    def test_select_all_equipment(self, tmp_db):
        result = run_sql_query("SELECT id, name FROM equipment", db_path=tmp_db)
        assert "CNC001" in result
        assert "CNC002" in result

    def test_where_clause_filter(self, tmp_db):
        result = run_sql_query(
            "SELECT id FROM equipment WHERE status = 'fault'", db_path=tmp_db
        )
        assert "CNC002" in result

    def test_parameterized_query(self, tmp_db):
        result = run_sql_query(
            "SELECT code, description FROM fault_codes WHERE code = ?",
            params=["E003"],
            db_path=tmp_db,
        )
        assert "E003" in result
        assert "主轴驱动故障" in result

    def test_aggregation_query(self, tmp_db):
        result = run_sql_query(
            "SELECT type, COUNT(*) as cnt FROM maintenance_records GROUP BY type",
            db_path=tmp_db,
        )
        assert "preventive" in result or "corrective" in result

    def test_empty_result(self, tmp_db):
        result = run_sql_query(
            "SELECT * FROM equipment WHERE id = 'NONEXISTENT'", db_path=tmp_db
        )
        assert "空" in result

    def test_blocks_non_select(self, tmp_db):
        result = run_sql_query("DELETE FROM equipment WHERE id = 'CNC001'", db_path=tmp_db)
        assert "错误" in result

    def test_blocks_dangerous_keywords(self, tmp_db):
        result = run_sql_query(
            "SELECT * FROM equipment; DROP TABLE equipment;--", db_path=tmp_db
        )
        assert "错误" in result or "不允许" in result

    def test_result_includes_row_count(self, tmp_db):
        result = run_sql_query("SELECT id FROM equipment", db_path=tmp_db)
        assert "条记录" in result


# ── list_tables 测试 ──────────────────────────────────────────────────────────

class TestRunListTables:
    def test_lists_all_tables(self, tmp_db):
        result = run_list_tables(db_path=tmp_db)
        for table in ("equipment", "maintenance_records", "fault_codes", "parts_inventory"):
            assert table in result

    def test_describe_specific_table(self, tmp_db):
        result = run_list_tables("equipment", db_path=tmp_db)
        assert "id" in result
        assert "status" in result

    def test_nonexistent_table(self, tmp_db):
        result = run_list_tables("no_such_table", db_path=tmp_db)
        assert "不存在" in result
