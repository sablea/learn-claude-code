# 工业设备智能检索系统

面向工业场景的 AI Agent，技术员可通过自然语言查询：
- **设备台账**：设备状态、安装信息、维护日期
- **维护记录**：历史维护操作、费用和技术员
- **故障代码**：故障原因和建议处理措施
- **零件库存**：库存数量和补货预警
- **故障报告**：历史故障分析文档（RAG 检索）
- **操作规程**：SOP 和设备手册（RAG 检索）

## 项目结构

```
industrial_retrieval/      # 主包
├── config.py              # 路径、环境变量和模型配置
├── agent.py               # Agent 循环和交互式会话
├── database/
│   ├── sample_data.py     # 示例数据常量（设备、故障码等）
│   ├── schema.py          # 数据库表创建和数据初始化
│   └── query.py           # SQL 查询工具（只读安全）
├── rag/
│   ├── documents.py       # 示例文档语料库
│   ├── tokenizer.py       # 中英文混合分词
│   ├── index.py           # TF-IDF 检索引擎
│   └── search.py          # RAG 搜索工具函数
├── skills/
│   ├── loader.py          # SKILL.md 解析器
│   └── runner.py          # 技能注入工具
└── tools/
    ├── definitions.py     # Agent 工具定义（Anthropic schema）
    └── executor.py        # 工具调用分发器

skills/
└── industrial-retrieval/
    └── SKILL.md           # 工业检索领域知识

tests/
├── conftest.py            # pytest fixtures
├── test_database.py       # 数据库测试
├── test_rag.py            # RAG 检索测试
├── test_skills.py         # Skills 加载测试
├── test_tools.py          # 工具执行测试
└── test_integration.py    # LLM 集成测试（需 API Key）

main.py                    # 程序入口
```

## 快速开始

```bash
# 安装依赖
pip install -r requirements.txt

# 配置 API Key
cp .env.example .env
# 编辑 .env 填入 ANTHROPIC_API_KEY

# 启动检索助手
python main.py
```

## 运行测试

```bash
# 运行所有单元测试（无需 API Key）
pytest tests/ -v -k "not integration"

# 运行全部测试（包含 LLM 集成，需要 API Key）
pytest tests/ -v
```

## 示例查询

```
技术员: CNC001设备当前状态如何？
技术员: 查询所有处于故障状态的设备
技术员: E003故障码的处理方法是什么？
技术员: 液压压力机紧急停机的操作流程
技术员: 哪些零件库存不足需要补货？
```

## 配置说明

```bash
# .env 文件
ANTHROPIC_API_KEY=sk-ant-xxx      # 必需
ANTHROPIC_BASE_URL=https://...    # 可选（第三方 API 代理）
MODEL_ID=claude-sonnet-4-5-20250929  # 可选（默认模型）
```
