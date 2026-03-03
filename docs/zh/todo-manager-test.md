# README-zh.md 要点总结

## 3 条核心要点

1. **项目目标**：从 0 到 1 构建一个类似 Claude Code 的最小化 AI 编程 Agent

2. **核心模式**：Agent Pattern 是最小循环，包含用户输入→消息数组→LLM→响应的流程

3. **关键逻辑**：当 stop_reason 为"tool_use"时执行工具并循环，否则返回文本
