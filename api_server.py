"""
RAG系统API服务器 - 支持SSE流式输出
"""

import os
import json
import logging
from typing import AsyncGenerator, Optional
from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException, status, Request
from fastapi.responses import StreamingResponse
from fastapi.security import APIKeyHeader
from pydantic import BaseModel, Field
from starlette.middleware.cors import CORSMiddleware

from dotenv import load_dotenv
from main import BlueRAGSystem
from config import RAGConfig

# 加载环境变量
load_dotenv()

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# API Key 认证
API_KEY_NAME = "X-API-Key"
api_key_header = APIKeyHeader(name=API_KEY_NAME, auto_error=False)

# 验证API Key
async def verify_api_key(api_key: Optional[str] = None) -> bool:
    """验证API Key"""
    valid_key = os.getenv("SERVER_API_KEY")
    if not valid_key:
        # 如果未设置SERVER_API_KEY，则跳过认证
        return True
    if api_key is None:
        return False
    return api_key == valid_key


# 全局RAG系统实例
rag_system: Optional[BlueRAGSystem] = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    """应用生命周期管理"""
    global rag_system

    # 启动时初始化
    logger.info("正在启动RAG API服务器...")
    try:
        rag_system = BlueRAGSystem()
        rag_system.initialize_system()
        rag_system.build_knowledge_base()
        logger.info("RAG系统初始化完成")
    except Exception as e:
        logger.error(f"RAG系统初始化失败: {e}")
        raise

    yield

    # 关闭时清理
    logger.info("正在关闭服务器...")


# 创建FastAPI应用
app = FastAPI(
    title="RAG知识库API",
    description="网络安全知识库检索增强生成API",
    version="1.0.0",
    lifespan=lifespan
)

# 配置CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # 生产环境请限制具体域名
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# 请求模型
class QuestionRequest(BaseModel):
    """问题请求模型"""
    question: str = Field(..., description="用户问题", min_length=1)
    top_k: Optional[int] = Field(None, description="检索文档数量", ge=1, le=10)


class SSEEvent:
    """SSE事件封装"""

    @staticmethod
    def format_event(event_type: str, data: dict) -> str:
        """格式化SSE事件"""
        data_str = json.dumps(data, ensure_ascii=False)
        return f"event: {event_type}\ndata: {data_str}\n\n"

    @staticmethod
    def done() -> str:
        """完成事件"""
        return SSEEvent.format_event("done", {"status": "completed"})

    @staticmethod
    def error(message: str) -> str:
        """错误事件"""
        return SSEEvent.format_event("error", {"error": message})

    @staticmethod
    def metadata(route_type: str, chunks: list, sources: list) -> str:
        """元数据事件"""
        return SSEEvent.format_event("metadata", {
            "route_type": route_type,
            "chunks_count": len(chunks),
            "sources": sources
        })

    @staticmethod
    def chunk(content: str) -> str:
        """内容块事件"""
        return SSEEvent.format_event("chunk", {"content": content})


async def stream_answer(
    question: str,
    top_k: Optional[int] = None
) -> AsyncGenerator[str, None]:
    """
    流式生成回答

    Args:
        question: 用户问题
        top_k: 检索文档数量

    Yields:
        SSE格式的字符串
    """
    if rag_system is None:
        yield SSEEvent.error("RAG系统未初始化")
        return

    try:
        # 1. 查询路由
        route_type = rag_system.generation_module.query_router(question)
        yield SSEEvent.metadata(route_type, [], [])

        # 2. 列表查询特殊处理
        if route_type == 'list':
            # 列表查询直接返回结果
            rewritten_query = question
        else:
            # 智能查询重写
            rewritten_query = rag_system.generation_module.query_rewrite(question)

        # 3. 检索相关文档
        filters = rag_system._extract_filters_from_query(question)
        if filters:
            relevant_chunks = rag_system.retrieval_module.metadata_filtered_search(
                rewritten_query, filters, top_k=top_k or rag_system.config.top_k
            )
            if not relevant_chunks:
                relevant_chunks = rag_system.retrieval_module.hybrid_search(
                    rewritten_query, top_k=top_k or rag_system.config.top_k
                )
        else:
            relevant_chunks = rag_system.retrieval_module.hybrid_search(
                rewritten_query, top_k=top_k or rag_system.config.top_k
            )

        if not relevant_chunks:
            yield SSEEvent.chunk("抱歉，没有找到相关的安全知识信息。请尝试其他关键词。")
            yield SSEEvent.done()
            return

        # 获取完整文档
        relevant_docs = rag_system.data_module.get_parent_documents(relevant_chunks)

        # 提取来源信息
        sources = []
        for doc in relevant_docs:
            doc_title = doc.metadata.get('title') or doc.metadata.get('file_name', '未知文档')
            if doc_title not in sources:
                sources.append(doc_title)

        # 发送来源信息
        yield SSEEvent.format_event("sources", {
            "sources": sources,
            "count": len(sources)
        })

        # 4. 生成回答
        if route_type == 'list':
            answer = rag_system.generation_module.generate_list_answer(question, relevant_docs)
            yield SSEEvent.chunk(answer)
            yield SSEEvent.done()
        elif route_type == "detail":
            # 详细查询使用分步指导模式
            async for chunk in rag_system.generation_module.generate_step_by_step_answer_stream_async(
                question, relevant_docs
            ):
                yield SSEEvent.chunk(chunk)
            yield SSEEvent.done()
        else:
            # 一般查询使用基础回答模式
            async for chunk in rag_system.generation_module.generate_basic_answer_stream_async(
                question, relevant_docs
            ):
                yield SSEEvent.chunk(chunk)
            yield SSEEvent.done()

    except Exception as e:
        logger.error(f"处理问题时出错: {e}", exc_info=True)
        yield SSEEvent.error(f"处理问题时出错: {str(e)}")


@app.get("/health")
async def health_check():
    """健康检查端点"""
    return {
        "status": "healthy",
        "rag_initialized": rag_system is not None
    }


@app.post("/api/ask")
async def ask_question(request: QuestionRequest, http_request: Request):
    """
    SSE流式问答接口

    请求头:
        X-API-Key: API密钥 (可选，取决于配置)

    请求体:
        {
            "question": "用户问题",
            "top_k": 3  // 可选，检索文档数量
        }

    SSE事件类型:
        - metadata: 查询元数据
        - sources: 来源文档信息
        - chunk: 内容片段
        - done: 完成
        - error: 错误
    """
    # 验证API Key
    api_key = http_request.headers.get(API_KEY_NAME)
    if not await verify_api_key(api_key):
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="无效的API Key"
        )

    return StreamingResponse(
        stream_answer(request.question, request.top_k),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no",  # 禁用Nginx缓冲
        }
    )


def main():
    """启动服务器"""
    import uvicorn

    # 从环境变量读取配置
    host = os.getenv("SERVER_HOST", "0.0.0.0")
    port = int(os.getenv("SERVER_PORT", "8000"))

    # 检查是否配置了API Key
    api_key = os.getenv("SERVER_API_KEY")
    if api_key:
        logger.info(f"API Key认证已启用")
    else:
        logger.warning("未配置SERVER_API_KEY，API将无需认证")

    logger.info(f"启动服务器: http://{host}:{port}")

    uvicorn.run(
        "api_server:app",
        host=host,
        port=port,
        reload=False,
        log_level="info"
    )


if __name__ == "__main__":
    main()
