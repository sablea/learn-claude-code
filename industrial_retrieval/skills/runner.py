"""
Skills 运行工具。

将 SkillLoader 封装为面向 Agent 工具调用的接口，
以缓存友好的方式将技能内容注入对话上下文。
"""

from industrial_retrieval.config import SKILLS_DIR
from industrial_retrieval.skills.loader import SkillLoader

# 模块级技能加载器单例
_loader: SkillLoader | None = None


def get_loader() -> SkillLoader:
    """获取（或懒加载初始化）全局 SkillLoader 单例。"""
    global _loader
    if _loader is None:
        _loader = SkillLoader(SKILLS_DIR)
    return _loader


def run_skill(skill_name: str) -> str:
    """
    加载指定技能并以缓存友好格式返回内容。

    内容通过 tool_result（用户消息）注入上下文，而不是修改 system prompt，
    确保提示词前缀缓存不被破坏。

    Args:
        skill_name: 技能名称（与 SKILL.md 中 name 字段一致）。

    Returns:
        包含 <skill-loaded> 标签的字符串，或错误提示。
    """
    loader  = get_loader()
    content = loader.get_skill_content(skill_name)

    if content is None:
        available = ", ".join(loader.list_skills()) or "（无可用技能）"
        return f"技能 '{skill_name}' 不存在。可用技能：{available}"

    return (
        f'<skill-loaded name="{skill_name}">\n'
        f"{content}\n"
        f"</skill-loaded>\n\n"
        f"已加载技能 '{skill_name}'，请按照技能指导进行操作。"
    )
