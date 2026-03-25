"""
配置模块：项目路径、环境变量和模型设置。

所有配置集中在此处管理，避免在各模块中硬编码路径或参数。
"""

import os
from pathlib import Path

from dotenv import load_dotenv

load_dotenv(override=True)

# 解决第三方 API 端点与 SDK 的认证冲突
if os.getenv("ANTHROPIC_BASE_URL"):
    os.environ.pop("ANTHROPIC_AUTH_TOKEN", None)

# ── 目录路径 ────────────────────────────────────────────────────────────────
# 项目根目录（config.py 所在包的上级）
PROJECT_ROOT = Path(__file__).parent.parent

SKILLS_DIR = PROJECT_ROOT / "skills"
DATA_DIR = PROJECT_ROOT / "data"
DOCS_DIR = DATA_DIR / "documents"
DB_PATH = DATA_DIR / "industrial.db"

# ── 模型配置 ─────────────────────────────────────────────────────────────────
MODEL = os.getenv("MODEL_ID", "claude-sonnet-4-5-20250929")
ANTHROPIC_BASE_URL = os.getenv("ANTHROPIC_BASE_URL")
