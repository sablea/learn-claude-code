# README-zh.md 要点提炼

1. **项目目标**：从 0 到 1 构建 nano Claude Code-like agent，学习 AI 编程代理的核心模式

2. **核心模式**：Agent Pattern 是最小循环，包含 用户消息 → LLM → 响应 的流程

3. **关键判断**：当 stop_reason == "tool_use" 时执行工具并追加结果，否则返回文本
