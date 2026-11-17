# 🤖 AI Agent 开发实战项目集

本项目是基于大语言模型（LLM）开发的 Agent 实战应用集。两个核心项目均采用纯 Python 手写核心逻辑（不依赖黑盒 Agent 框架），深度还原了 ReAct 范式、Function Calling 机制及 RAG 检索增强生成流程。

## 📂 项目一：企业私有知识库智能客服 (RAG + Function Calling)
- **技术栈**: Python, Streamlit, OpenAI SDK, LangChain, ChromaDB
- **核心功能**: 
  - 基于 ReAct 范式手写 Agent 核心循环，实现大模型自主决策与多工具调用。
  - 构建 RAG 管线，通过 LangChain 加载私有文档，切片后存入 Chroma 向量库，实现私有知识的精准问答。
  - 引入基于 `tiktoken` 的动态滑动窗口裁剪策略，防止多轮对话下的 Token 超限报错。
  - 设计 Prompt 护栏，当检索不到相关内容时，模型拒绝编造（防幻觉）。
  - 基于 Streamlit 构建 Web 交互界面，利用 `@st.cache_resource` 缓存向量库优化响应速度。

## 📂 项目二：自动化数据分析与报告生成 Agent
- **技术栈**: Python, Pandas, OpenAI SDK, tiktoken
- **核心功能**:
  - 赋予大模型读取 CSV 数据、Pandas 聚合计算、模拟发送邮件等 3 种以上工具能力。
  - 实现从“数据读取 -> 逻辑分析（判断增长率） -> 动作执行（发邮件）”的复杂长链路自主规划。
  - 实现容错机制：通过 `try-except` 捕获工具执行报错，并将 Error Log 喂回大模型触发自我修正。
  - 通过严格约束 Tool Schema 的 Description，解决大模型“只输出文本不调用工具”的幻觉问题。

---

## 🚀 快速开始 (Quick Start)

1. 克隆仓库并安装依赖：
   ```bash
   git clone https://github.com/2092990276/Agent-Development-Demos.git
   pip install -r requirements.txt
2. 配置环境变量：
在根目录新建 .env 文件，填入你的 API Key（本项目使用智谱 GLM-4）：

  ```env
  ZHIPU_API_KEY=your_api_key_here
3. 运行项目一（智能客服 Web 端）：

  ```bash
  streamlit run app.py
4. 运行项目二（数据分析 Agent 端）：

  ```bash
  python agent.py
