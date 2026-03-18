"""
Milvus检索优化模块
"""

import logging
from typing import List, Dict, Any

from langchain_community.retrievers import BM25Retriever
from langchain_core.documents import Document
from rag_modules.milvus_index import MilvusIndexModule

logger = logging.getLogger(__name__)


class MilvusRetrievalOptimizationModule:
    """基于Milvus的检索优化模块 - 支持向量检索和BM25混合检索"""

    def __init__(self, milvus_index: MilvusIndexModule, chunks: List[Document]):
        """
        初始化检索优化模块

        Args:
            milvus_index: Milvus索引模块实例
            chunks: 文档块列表（用于BM25）
        """
        self.milvus_index = milvus_index
        self.chunks = chunks
        self.bm25_retriever = None
        self.setup_retrievers()

    def setup_retrievers(self):
        """设置BM25检索器"""
        logger.info("正在设置BM25检索器...")

        if self.chunks:
            self.bm25_retriever = BM25Retriever.from_documents(
                self.chunks,
                k=5
            )
            logger.info("BM25检索器设置完成")
        else:
            logger.warning("文档块为空，跳过BM25检索器设置")

    def vector_search(self, query: str, top_k: int = 5, filter: str = None) -> List[Document]:
        """
        向量检索

        Args:
            query: 查询文本
            top_k: 返回结果数量
            filter: Milvus过滤表达式

        Returns:
            检索到的文档列表
        """
        return self.milvus_index.similarity_search(query, k=top_k, filter=filter)

    def hybrid_search(self, query: str, top_k: int = 3) -> List[Document]:
        """
        混合检索 - 结合向量检索和BM25检索，使用RRF重排

        Args:
            query: 查询文本
            top_k: 返回结果数量

        Returns:
            检索到的文档列表
        """
        # 向量检索
        vector_docs = self.vector_search(query, top_k * 2)

        # BM25检索
        bm25_docs = []
        if self.bm25_retriever:
            bm25_docs = self.bm25_retriever.invoke(query)

        # 使用RRF重排
        reranked_docs = self._rrf_rerank(vector_docs, bm25_docs)
        return reranked_docs[:top_k]

    def metadata_filtered_search(self, query: str, filters: Dict[str, Any], top_k: int = 5) -> List[Document]:
        """
        带元数据过滤的检索（使用Milvus原生过滤）

        Args:
            query: 查询文本
            filters: 元数据过滤条件
            top_k: 返回结果数量

        Returns:
            过滤后的文档列表
        """
        # 构建Milvus过滤表达式
        filter_expr = self._build_filter_expression(filters)

        # 使用Milvus的过滤搜索
        try:
            results = self.vector_search(query, top_k=top_k, filter=filter_expr)
            if results:
                return results
        except Exception as e:
            logger.warning(f"Milvus过滤搜索失败: {e}")

        # 回退：先获取更多结果，然后手动过滤
        docs = self.hybrid_search(query, top_k=top_k * 3)
        return self._apply_metadata_filter(docs, filters, top_k)

    def _build_filter_expression(self, filters: Dict[str, Any]) -> str:
        """
        构建Milvus过滤表达式

        Args:
            filters: 过滤条件字典

        Returns:
            Milvus过滤表达式字符串
        """
        conditions = []
        for key, value in filters.items():
            if key == "domain":
                conditions.append(f'domain == "{value}"')
            elif key == "file_extension":
                conditions.append(f'file_extension == "{value}"')
            elif key == "file_name":
                conditions.append(f'file_name like "%{value}%"')
            elif key == "title":
                conditions.append(f'title like "%{value}%"')

        return " and ".join(conditions) if conditions else ""

    def _apply_metadata_filter(self, docs: List[Document], filters: Dict[str, Any], top_k: int) -> List[Document]:
        """
        应用元数据过滤

        Args:
            docs: 文档列表
            filters: 过滤条件
            top_k: 返回数量

        Returns:
            过滤后的文档列表
        """
        filtered_docs = []
        for doc in docs:
            match = True
            for key, value in filters.items():
                if key in doc.metadata:
                    if isinstance(value, list):
                        if doc.metadata[key] not in value:
                            match = False
                            break
                    else:
                        if doc.metadata[key] != value:
                            match = False
                            break
                else:
                    match = False
                    break

            if match:
                filtered_docs.append(doc)
                if len(filtered_docs) >= top_k:
                    break

        return filtered_docs

    def _rrf_rerank(
        self,
        vector_docs: List[Document],
        bm25_docs: List[Document],
        k: int = 60
    ) -> List[Document]:
        """
        使用RRF (Reciprocal Rank Fusion) 算法重排文档

        Args:
            vector_docs: 向量检索结果
            bm25_docs: BM25检索结果
            k: RRF参数，用于平滑排名

        Returns:
            重排后的文档列表
        """
        doc_scores = {}
        doc_objects = {}

        # 计算向量检索结果的RRF分数
        for rank, doc in enumerate(vector_docs):
            doc_id = hash(doc.page_content)
            doc_objects[doc_id] = doc
            rrf_score = 1.0 / (k + rank + 1)
            doc_scores[doc_id] = doc_scores.get(doc_id, 0) + rrf_score
            logger.debug(f"向量检索 - 文档{rank+1}: RRF分数 = {rrf_score:.4f}")

        # 计算BM25检索结果的RRF分数
        for rank, doc in enumerate(bm25_docs):
            doc_id = hash(doc.page_content)
            doc_objects[doc_id] = doc
            rrf_score = 1.0 / (k + rank + 1)
            doc_scores[doc_id] = doc_scores.get(doc_id, 0) + rrf_score
            logger.debug(f"BM25检索 - 文档{rank+1}: RRF分数 = {rrf_score:.4f}")

        # 按最终RRF分数排序
        sorted_docs = sorted(doc_scores.items(), key=lambda x: x[1], reverse=True)

        # 构建最终结果
        reranked_docs = []
        for doc_id, final_score in sorted_docs:
            if doc_id in doc_objects:
                doc = doc_objects[doc_id]
                doc.metadata['rrf_score'] = final_score
                reranked_docs.append(doc)
                logger.debug(f"最终排序 - 分数: {final_score:.4f}")

        logger.info(
            f"RRF重排完成: 向量{len(vector_docs)}个, "
            f"BM25{len(bm25_docs)}个, 合并{len(reranked_docs)}个"
        )

        return reranked_docs

    def get_stats(self) -> Dict[str, Any]:
        """获取检索模块统计信息"""
        stats = self.milvus_index.get_collection_stats()
        stats["bm25_available"] = self.bm25_retriever is not None
        stats["chunks_count"] = len(self.chunks) if self.chunks else 0
        return stats
