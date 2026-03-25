"""
pytest 共享 fixtures。

所有测试模块可直接使用此处定义的 fixtures，无需额外导入。
"""

import pytest
from pathlib import Path


@pytest.fixture()
def tmp_db(tmp_path: Path):
    """提供一个在临时目录中初始化的测试数据库路径。"""
    from industrial_retrieval.database.schema import init_database

    db_path = tmp_path / "test.db"
    init_database(db_path)
    return db_path


@pytest.fixture()
def tmp_docs(tmp_path: Path):
    """提供一个在临时目录中初始化了示例文档的文档目录。"""
    from industrial_retrieval.rag.documents import init_documents

    docs_dir = tmp_path / "documents"
    init_documents(docs_dir)
    return docs_dir


@pytest.fixture()
def tmp_skills(tmp_path: Path):
    """提供一个包含测试技能的临时技能目录。"""
    skill_dir = tmp_path / "test-skill"
    skill_dir.mkdir()
    (skill_dir / "SKILL.md").write_text(
        "---\n"
        "name: test-skill\n"
        "description: 测试技能\n"
        "---\n\n"
        "这是测试技能的正文内容。\n"
        "步骤1：分析问题\n"
        "步骤2：执行方案\n",
        encoding="utf-8",
    )
    return tmp_path
