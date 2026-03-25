"""
Skills 加载器。

从 SKILL.md 文件（YAML 前置元数据 + Markdown 正文）中加载领域知识，
实现按需注入专业知识的缓存友好机制。

SKILL.md 格式：
    ---
    name: skill-name
    description: 一句话描述，何时应加载此技能
    ---

    # 技能正文（Markdown）
    ...

缓存友好注入原理：
    技能正文通过 tool_result（用户消息的一部分）注入，
    而非写入 system prompt，从而不破坏提示词前缀的缓存。
"""

import re
from pathlib import Path


class SkillLoader:
    """
    扫描技能目录，解析并管理所有 SKILL.md 文件。

    Args:
        skills_dir: 包含技能子目录的根目录（每个子目录有一个 SKILL.md）。
    """

    def __init__(self, skills_dir: Path) -> None:
        self.skills_dir = skills_dir
        self.skills: dict[str, dict] = {}
        self._load_all()

    # ── 公共接口 ────────────────────────────────────────────────────────────

    def get_descriptions(self) -> str:
        """
        返回所有已加载技能的名称和一句话描述（用于 system prompt）。
        """
        if not self.skills:
            return "(暂无可用技能)"
        return "\n".join(
            f"- {name}: {skill['description']}"
            for name, skill in self.skills.items()
        )

    def get_skill_content(self, name: str) -> str | None:
        """
        返回指定技能的完整 Markdown 正文（用于注入上下文）。

        Returns:
            技能内容字符串，或 None（技能不存在时）。
        """
        if name not in self.skills:
            return None
        skill = self.skills[name]
        return f"# Skill: {skill['name']}\n\n{skill['body']}"

    def list_skills(self) -> list[str]:
        """返回所有已加载技能的名称列表。"""
        return list(self.skills.keys())

    # ── 私有方法 ────────────────────────────────────────────────────────────

    def _load_all(self) -> None:
        """扫描 skills_dir，加载全部有效 SKILL.md 文件。"""
        if not self.skills_dir.exists():
            return

        for skill_dir in self.skills_dir.iterdir():
            if not skill_dir.is_dir():
                continue
            skill_md = skill_dir / "SKILL.md"
            if not skill_md.exists():
                continue
            parsed = self._parse(skill_md)
            if parsed:
                self.skills[parsed["name"]] = parsed

    def _parse(self, path: Path) -> dict | None:
        """
        解析单个 SKILL.md 文件。

        Returns:
            包含 name、description、body 的字典，解析失败返回 None。
        """
        content = path.read_text(encoding="utf-8")
        match = re.match(r"^---\s*\n(.*?)\n---\s*\n(.*)$", content, re.DOTALL)
        if not match:
            return None

        frontmatter, body = match.groups()
        meta = self._parse_frontmatter(frontmatter)

        if "name" not in meta or "description" not in meta:
            return None

        return {
            "name":        meta["name"],
            "description": meta["description"],
            "body":        body.strip(),
        }

    @staticmethod
    def _parse_frontmatter(frontmatter: str) -> dict[str, str]:
        """解析 YAML 风格的前置元数据（简单 key: value 格式）。"""
        meta: dict[str, str] = {}
        for line in frontmatter.strip().splitlines():
            if ":" in line:
                key, _, value = line.partition(":")
                meta[key.strip()] = value.strip().strip("\"'")
        return meta
