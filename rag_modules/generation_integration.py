"""
生成集成模块
"""

import os
import logging
from typing import List

from langchain_core.prompts import ChatPromptTemplate, PromptTemplate
from langchain_community.chat_models.moonshot import MoonshotChat
from langchain_core.documents import Document
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser

logger = logging.getLogger(__name__)

class GenerationIntegrationModule:
    """生成集成模块 - 负责LLM集成和回答生成"""
    
    def __init__(self, model_name: str = "kimi-k2-0711-preview", temperature: float = 0.1, max_tokens: int = 2048):
        """
        初始化生成集成模块
        
        Args:
            model_name: 模型名称
            temperature: 生成温度
            max_tokens: 最大token数
        """
        self.model_name = model_name
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.llm = None
        self.setup_llm()
    
    def setup_llm(self):
        """初始化大语言模型"""
        logger.info(f"正在初始化LLM: {self.model_name}")

        api_key = os.getenv("MOONSHOT_API_KEY")
        if not api_key:
            raise ValueError("请设置 MOONSHOT_API_KEY 环境变量")

        self.llm = MoonshotChat(
            model=self.model_name,
            temperature=self.temperature,
            max_tokens=self.max_tokens,
            moonshot_api_key=api_key
        )
        
        logger.info("LLM初始化完成")
    
    def generate_basic_answer(self, query: str, context_docs: List[Document]) -> str:
        """
        生成基础回答

        Args:
            query: 用户查询
            context_docs: 上下文文档列表

        Returns:
            生成的回答
        """
        context = self._build_context(context_docs)

        prompt = ChatPromptTemplate.from_template("""
你是一位专业的知识助手。请根据以下文档信息回答用户的问题。

用户问题: {question}

相关文档信息:
{context}

请提供详细、准确的回答。如果信息不足，请诚实说明。

回答:""")

        # 使用LCEL构建链
        chain = (
            {"question": RunnablePassthrough(), "context": lambda _: context}
            | prompt
            | self.llm
            | StrOutputParser()
        )

        response = chain.invoke(query)
        return response
    
    def generate_step_by_step_answer(self, query: str, context_docs: List[Document]) -> str:
        """
        生成详细分步回答

        Args:
            query: 用户查询
            context_docs: 上下文文档列表

        Returns:
            分步骤的详细回答
        """
        context = self._build_context(context_docs)

        prompt = ChatPromptTemplate.from_template("""
你是一位专业的知识导师。请根据文档信息，为用户提供详细的分步指导。

用户问题: {question}

相关文档信息:
{context}

请提供详细的、结构化的回答：
1. 先给出简明扼要的概述
2. 然后按照逻辑顺序分步骤详细说明
3. 最后提供关键要点或注意事项

请确保回答准确、实用、易于理解。

回答:""")

        chain = (
            {"question": RunnablePassthrough(), "context": lambda _: context}
            | prompt
            | self.llm
            | StrOutputParser()
        )

        response = chain.invoke(query)
        return response
    
    def query_rewrite(self, query: str) -> str:
        """
        智能查询重写 - 让大模型判断是否需要重写查询

        Args:
            query: 原始查询

        Returns:
            重写后的查询或原查询
        """
        prompt = PromptTemplate(
            template="""
你是一个智能查询分析助手。请分析用户的查询，判断是否需要重写以提高搜索效果。

原始查询: {query}

分析规则：
1. **具体明确的查询**（直接返回原查询）：
   - 包含具体概念或术语
   - 明确的问题询问
   - 具体的技术或方法询问

2. **模糊不清的查询**（需要重写）：
   - 过于宽泛或模糊
   - 缺乏具体信息
   - 口语化表达

重写原则：
- 保持原意不变
- 增加相关领域术语
- 保持简洁性

示例：
- "怎么弄" → "操作方法和步骤"
- "有什么" → "相关技术和方法"
- "怎么做X" → "X的具体操作方法"（保持原查询）

请输出最终查询（如果不需要重写就返回原查询）:""",
            input_variables=["query"]
        )

        chain = (
            {"query": RunnablePassthrough()}
            | prompt
            | self.llm
            | StrOutputParser()
        )

        response = chain.invoke(query).strip()

        # 记录重写结果
        if response != query:
            logger.info(f"查询已重写: '{query}' → '{response}'")
        else:
            logger.info(f"查询无需重写: '{query}'")

        return response



    def query_router(self, query: str) -> str:
        """
        查询路由 - 根据查询类型选择不同的处理方式

        Args:
            query: 用户查询

        Returns:
            路由类型 ('list', 'detail', 'general')
        """
        prompt = ChatPromptTemplate.from_template("""
根据用户的问题，将其分类为以下三种类型之一：

1. 'list' - 用户想要获取列表或推荐，只需要名称或标题
   例如：推荐几个、有什么、列出所有

2. 'detail' - 用户想要具体的方法或详细信息
   例如：怎么做、操作步骤、详细说明

3. 'general' - 其他一般性问题
   例如：是什么、原理说明、定义解释

请只返回分类结果：list、detail 或 general

用户问题: {query}

分类结果:""")

        chain = (
            {"query": RunnablePassthrough()}
            | prompt
            | self.llm
            | StrOutputParser()
        )

        result = chain.invoke(query).strip().lower()

        # 确保返回有效的路由类型
        if result in ['list', 'detail', 'general']:
            return result
        else:
            return 'general'  # 默认类型

    def generate_list_answer(self, query: str, context_docs: List[Document]) -> str:
        """
        生成列表式回答 - 适用于推荐类查询

        Args:
            query: 用户查询
            context_docs: 上下文文档列表

        Returns:
            列表式回答
        """
        if not context_docs:
            return "抱歉，没有找到相关的文档信息。"

        # 提取文档标题
        doc_titles = []
        for doc in context_docs:
            title = doc.metadata.get('title', doc.metadata.get('file_name', '未知文档'))
            if title not in doc_titles:
                doc_titles.append(title)

        # 构建简洁的列表回答
        if len(doc_titles) == 1:
            return f"为您推荐：{doc_titles[0]}"
        elif len(doc_titles) <= 3:
            return f"为您推荐以下文档：\n" + "\n".join([f"{i+1}. {title}" for i, title in enumerate(doc_titles)])
        else:
            return f"为您推荐以下文档：\n" + "\n".join([f"{i+1}. {title}" for i, title in enumerate(doc_titles[:3])]) + f"\n\n还有其他 {len(doc_titles)-3} 个文档可供选择。"

    def generate_basic_answer_stream(self, query: str, context_docs: List[Document]):
        """
        生成基础回答 - 流式输出

        Args:
            query: 用户查询
            context_docs: 上下文文档列表

        Yields:
            生成的回答片段
        """
        context = self._build_context(context_docs)

        prompt = ChatPromptTemplate.from_template("""
你是一位专业的知识助手。请根据以下文档信息回答用户的问题。

用户问题: {question}

相关文档信息:
{context}

请提供详细、准确的回答。如果信息不足，请诚实说明。

回答:""")

        chain = (
            {"question": RunnablePassthrough(), "context": lambda _: context}
            | prompt
            | self.llm
            | StrOutputParser()
        )

        for chunk in chain.stream(query):
            yield chunk

    def generate_step_by_step_answer_stream(self, query: str, context_docs: List[Document]):
        """
        生成详细步骤回答 - 流式输出

        Args:
            query: 用户查询
            context_docs: 上下文文档列表

        Yields:
            详细步骤回答片段
        """
        context = self._build_context(context_docs)

        prompt = ChatPromptTemplate.from_template("""
你是一位专业的知识导师。请根据文档信息，为用户提供详细的分步指导。

用户问题: {question}

相关文档信息:
{context}

请提供详细的、结构化的回答：
1. 先给出简明扼要的概述
2. 然后按照逻辑顺序分步骤详细说明
3. 最后提供关键要点或注意事项

请确保回答准确、实用、易于理解。

回答:""")

        chain = (
            {"question": RunnablePassthrough(), "context": lambda _: context}
            | prompt
            | self.llm
            | StrOutputParser()
        )

        for chunk in chain.stream(query):
            yield chunk

    def _build_context(self, docs: List[Document], max_length: int = 2000) -> str:
        """
        构建上下文字符串

        Args:
            docs: 文档列表
            max_length: 最大长度

        Returns:
            格式化的上下文字符串
        """
        if not docs:
            return "暂无相关文档信息。"

        context_parts = []
        current_length = 0

        for i, doc in enumerate(docs, 1):
            # 添加元数据信息
            metadata_info = f"【文档 {i}】"
            if 'title' in doc.metadata:
                metadata_info += f" {doc.metadata['title']}"
            elif 'file_name' in doc.metadata:
                metadata_info += f" {doc.metadata['file_name']}"
            if 'file_extension' in doc.metadata:
                metadata_info += f" | 类型: {doc.metadata['file_extension']}"

            # 构建文档文本
            doc_text = f"{metadata_info}\n{doc.page_content}\n"

            # 检查长度限制
            if current_length + len(doc_text) > max_length:
                break

            context_parts.append(doc_text)
            current_length += len(doc_text)

        return "\n" + "="*50 + "\n".join(context_parts)
