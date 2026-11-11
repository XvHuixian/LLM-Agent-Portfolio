# 智能招聘中枢 (AI Resume Agent)

基于大语言模型（LLM）与向量检索（RAG）技术，从零手写的简历批量解析与智能初筛系统。

## ✨ 核心功能
- 多格式文件流式解析：支持 TXT、PDF、DOCX 格式，利用多线程并发解析，极速构建简历文本库。
- 对话式 RAG（检索增强生成）：基于 TF-IDF 构建本地向量空间，使用余弦相似度进行精准召回，避免超长文本导致的 Token 爆炸。
- 支持多轮追问：Agent 具备上下文记忆，可针对特定候选人连续追问（如：“张三有几年经验？”）。
- 多模型无缝切换：内置 DeepSeek、通义千问、Kimi 等主流服务商路由，自动拉取可用模型列表。
- 高可用容灾机制：封装 API 指数退避重试机制，具备日志监控（可观测性）。

## 🛠️ 技术栈
- 前端框架：Streamlit
- 大模型交互：OpenAI SDK (兼容所有国内主流大模型)
- 文件解析：pypdf, python-docx
- 向量检索：scikit-learn (TF-IDF + Cosine Similarity)
- 并发处理：concurrent.futures

## 🚀 如何运行
1. 克隆本项目：`git clone https://github.com/XvHuixian/LLM-Agent-Portfolio.git`
2. 安装依赖：`pip install streamlit openai pypdf python-docx scikit-learn`
3. 启动应用：`streamlit run my_agent.py`
