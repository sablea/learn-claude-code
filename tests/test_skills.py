"""
Skills 模块测试。

覆盖：SkillLoader（解析、加载、获取描述和内容）、runner（run_skill）。
"""

import pytest
from pathlib import Path

from industrial_retrieval.skills.loader import SkillLoader
from industrial_retrieval.skills.runner import run_skill
from industrial_retrieval.agent import build_system_prompt


# ── SkillLoader 测试 ──────────────────────────────────────────────────────────

class TestSkillLoader:
    def test_empty_directory_loads_zero_skills(self, tmp_path):
        loader = SkillLoader(tmp_path)
        assert loader.skills == {}

    def test_nonexistent_directory(self, tmp_path):
        loader = SkillLoader(tmp_path / "nonexistent")
        assert loader.skills == {}

    def test_parses_valid_skill_md(self, tmp_skills):
        loader = SkillLoader(tmp_skills)
        assert "test-skill" in loader.skills

    def test_parsed_skill_has_required_fields(self, tmp_skills):
        loader = SkillLoader(tmp_skills)
        skill = loader.skills["test-skill"]
        assert "name" in skill
        assert "description" in skill
        assert "body" in skill

    def test_parsed_skill_body_contains_content(self, tmp_skills):
        loader = SkillLoader(tmp_skills)
        assert "步骤1" in loader.skills["test-skill"]["body"]

    def test_rejects_skill_md_without_frontmatter(self, tmp_path):
        bad_dir = tmp_path / "bad-skill"
        bad_dir.mkdir()
        (bad_dir / "SKILL.md").write_text("无前置元数据的普通文本", encoding="utf-8")

        loader = SkillLoader(tmp_path)
        assert "bad-skill" not in loader.skills

    def test_rejects_skill_md_missing_name(self, tmp_path):
        skill_dir = tmp_path / "no-name"
        skill_dir.mkdir()
        (skill_dir / "SKILL.md").write_text(
            "---\ndescription: 缺少名称字段\n---\n\n正文", encoding="utf-8"
        )
        loader = SkillLoader(tmp_path)
        assert len(loader.skills) == 0

    def test_get_descriptions_returns_string(self, tmp_skills):
        loader = SkillLoader(tmp_skills)
        desc = loader.get_descriptions()
        assert isinstance(desc, str)
        assert "test-skill" in desc

    def test_get_descriptions_empty(self, tmp_path):
        loader = SkillLoader(tmp_path)
        desc = loader.get_descriptions()
        assert "暂无" in desc

    def test_get_skill_content_returns_body(self, tmp_skills):
        loader = SkillLoader(tmp_skills)
        content = loader.get_skill_content("test-skill")
        assert content is not None
        assert "步骤1" in content
        assert "Skill: test-skill" in content

    def test_get_skill_content_nonexistent(self, tmp_skills):
        loader = SkillLoader(tmp_skills)
        assert loader.get_skill_content("nonexistent") is None

    def test_list_skills(self, tmp_skills):
        loader = SkillLoader(tmp_skills)
        skills = loader.list_skills()
        assert isinstance(skills, list)
        assert "test-skill" in skills

    def test_industrial_retrieval_skill_exists(self):
        """验证项目内置的 industrial-retrieval 技能已正确存在。"""
        from industrial_retrieval.config import SKILLS_DIR
        loader = SkillLoader(SKILLS_DIR)
        assert "industrial-retrieval" in loader.skills, (
            "skills/industrial-retrieval/SKILL.md 不存在或格式错误"
        )


# ── run_skill 测试 ────────────────────────────────────────────────────────────

class TestRunSkill:
    def test_returns_skill_loaded_tag(self, tmp_skills, monkeypatch):
        from industrial_retrieval.skills import runner as runner_module
        runner_module._loader = SkillLoader(tmp_skills)

        result = run_skill("test-skill")
        assert "<skill-loaded" in result
        assert "test-skill" in result
        assert "</skill-loaded>" in result

    def test_includes_skill_body(self, tmp_skills, monkeypatch):
        from industrial_retrieval.skills import runner as runner_module
        runner_module._loader = SkillLoader(tmp_skills)

        result = run_skill("test-skill")
        assert "步骤1" in result

    def test_unknown_skill_returns_error(self, tmp_skills, monkeypatch):
        from industrial_retrieval.skills import runner as runner_module
        runner_module._loader = SkillLoader(tmp_skills)

        result = run_skill("nonexistent-skill")
        assert "不存在" in result

    def test_skill_body_not_in_system_prompt(self):
        """
        验证技能正文不出现在系统提示词中（保持缓存友好）。

        系统提示词只能包含技能的一句话 description（来自 SKILL.md 前置元数据）。
        技能正文（如 SQL 示例、检索流程）通过 Skill 工具调用按需注入，
        以确保 system prompt 前缀保持不变，最大化利用 Anthropic 提示词缓存。

        断言逻辑：取技能正文的前几个词——这些词只会出现在正文里，
        不会出现在 description 单行描述中。
        """
        from industrial_retrieval.config import SKILLS_DIR
        from industrial_retrieval.skills.loader import SkillLoader

        loader = SkillLoader(SKILLS_DIR)
        system = build_system_prompt()

        # 获取 industrial-retrieval 的正文第一行，验证它不在 system prompt 中
        skill = loader.skills.get("industrial-retrieval")
        if skill:
            # 正文的第一个非空段落行（不应出现在 system prompt）
            first_body_line = next(
                (line.strip() for line in skill["body"].splitlines() if line.strip()),
                None,
            )
            if first_body_line:
                assert first_body_line not in system, (
                    "技能正文首行不应出现在 system prompt 中（会破坏提示词缓存）"
                )
