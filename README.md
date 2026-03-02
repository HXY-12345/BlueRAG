<p align="center">
  <img src="assets/logo.png" alt="BlueRAG Logo" width="160" />
</p>

<h1 align="center">BlueRAG</h1>

<p align="center">
  <b>面向安全知识的检索增强生成（RAG）问答系统</b><br/>
  用混合检索（BM25 + 向量召回）+ 证据注入生成，实现可追溯、低幻觉的安全知识问答。
</p>

<p align="center">
  <img src="https://img.shields.io/badge/Python-3.10%2B-blue" />
  <img src="https://img.shields.io/badge/License-MIT-green" />
  <img src="https://img.shields.io/badge/RAG-BM25%20%2B%20FAISS-orange" />
</p>


## ✨ 亮点特性
- **知识入库**：支持 PDF / Word / 网页文本等多源资料解析与清洗
- **混合检索（Hybrid Retrieval）**：BM25 关键词召回 + FAISS 向量召回提升命中与语义匹配
- **元数据过滤（Metadata Filter）**：按领域/来源等条件过滤检索结果（可扩展）
- **生成集成（Grounded Generation）**：将检索证据按模板注入 LLM，减少幻觉并支持引用来源
- **流式输出**：支持 SSE/Chunked 方式逐步输出回答，便于集成到业务系统

> 本项目以“检索 → 证据拼接 → 生成”的标准 RAG 链路组织代码，适合作为安全知识库问答/排障助手的参考实现。

---

## 📦 目录结构
```text
.
├─ main.py                 # 主入口：交互式问答 + 端到端RAG流程
├─ config.py               # 配置：数据路径、embedding模型、LLM模型、top_k等
├─ requirements.txt
├─ data/                   # 示例数据目录
├─ rag_modules/            # 数据准备/索引构建/检索优化/生成集成等模块
└─ vector_index/           # 向量索引落盘目录（默认）
````

---

## 🚀 快速开始

### 1) 安装依赖

```bash
pip install -r requirements.txt
```

### 2) 准备数据

把你的知识库文件放到 `./data` 下（或在 `config.py` 修改 `data_path`）。

### 3) 配置环境变量

项目运行需要一个 API Key（用于调用 LLM 推理服务，按你的服务端实现而定）：

```bash
# Windows PowerShell
setx RAG_API_KEY "YOUR_API_KEY"
```

> 也可以使用 `.env` 文件（项目中已引入 dotenv 加载）。

### 4) 运行

```bash
python main.py
```

进入交互式问答后，输入问题即可；可选择是否启用流式输出。

---

## ⚙️ 配置说明

默认配置在 `config.py`，常用项：

* `data_path`：知识库文件目录（默认 `./data`）
* `index_save_path`：向量索引落盘目录（默认 `./vector_index`）
* `embedding_model`：Embedding 模型（默认 `BAAI/bge-small-zh-v1.5`）
* `llm_model`：LLM 模型名（示例默认 `qwen/...`）
* `top_k`：召回数量
* `temperature / max_tokens`：生成参数

---

## 🧠 RAG 流程简述

1. **Ingestion**：加载并清洗文档，保留来源等元数据
2. **Indexing**：文本分块 → Embedding → FAISS 向量索引落盘
3. **Retrieval**：混合检索（BM25 + 向量）→ Top-K 证据集合
4. **Generation**：将证据注入 Prompt → LLM 生成回答（可附带引用/来源字段）

---

## 📝 License

本项目采用 **MIT License** 开源协议。

* 你可以自由使用、修改、分发本项目代码（需保留原始版权声明与许可声明）。

---

## 🙌 致谢

* LangChain / Unstructured / FAISS / BM25 等开源生态

```

---
