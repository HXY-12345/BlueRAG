"""
Milvus 向量索引模块。

该模块提供基于 Milvus 的向量存储和检索功能，用于构建 RAG 系统的知识库。
主要功能包括：
- 创建和管理 Milvus 集合（Collection）
- 向量索引的创建和加载
- 文档的向量化和存储
- 相似度搜索
"""

import logging
import os
import time
from pathlib import Path
from typing import Any, Dict, List

from langchain_core.documents import Document
from langchain_huggingface import HuggingFaceEmbeddings
from pymilvus import (
    CollectionSchema,
    DataType,
    FieldSchema,
    MilvusClient,
    MilvusException,
)
from pymilvus.client.types import LoadState

logger = logging.getLogger(__name__)

# ==================== 集合配置 ====================
COLLECTION_NAME = "rag_security_kb"  # 集合名称
VECTOR_FIELD_NAME = "embedding"  # 向量字段名
VECTOR_INDEX_NAME = "embedding_idx"  # 向量索引名
VECTOR_INDEX_TYPE = "IVF_FLAT"  # 索引类型：倒排文件索引
VECTOR_METRIC_TYPE = "COSINE"  # 距离度量类型：余弦相似度
VECTOR_INDEX_CONFIG = {"nlist": 128}  # 索引参数：聚类中心数量


def get_embedding_dimension(embeddings_model: HuggingFaceEmbeddings) -> int:
    """探测嵌入模型的真实输出维度。"""
    vector = embeddings_model.embed_query("test")
    dim = len(vector)
    logger.info("检测到嵌入维度: %s", dim)
    return dim


def has_local_model_cache(model_name: str, cache_dir: Path) -> bool:
    """检查 sentence-transformers 模型是否已存在于本地缓存。"""
    model_cache_dir = cache_dir / f"models--{model_name.replace('/', '--')}" / "snapshots"
    return model_cache_dir.exists() and any(model_cache_dir.iterdir())


class MilvusIndexModule:
    """基于 Milvus 的向量存储辅助类。

    提供完整的向量索引管理功能，包括：
    - Milvus 客户端连接
    - 嵌入模型初始化
    - 集合和索引的创建与管理
    - 文档的向量化和检索
    """

    def __init__(
        self,
        model_name: str = "BAAI/bge-small-zh-v1.5",
        milvus_uri: str = "http://localhost:19530",
        database_name: str = "rag_db",
        collection_name: str = COLLECTION_NAME,
    ):
        """初始化 Milvus 索引模块。

        Args:
            model_name: 嵌入模型名称，默认使用 BGE 中文模型
            milvus_uri: Milvus 服务器地址
            database_name: 数据库名称
            collection_name: 集合名称
        """
        self.model_name = model_name
        self.milvus_uri = milvus_uri
        self.database_name = database_name
        self.collection_name = collection_name
        self.embeddings = None
        self.client = None
        self.collection_exists = False
        self.embedding_dim = None

        self.setup_client()
        self.setup_embeddings()
        self.embedding_dim = get_embedding_dimension(self.embeddings)
        self.setup_collection()

    def setup_client(self):
        """初始化 Milvus 客户端并确保目标数据库存在。"""
        logger.info("正在连接 Milvus: %s", self.milvus_uri)

        # 数据库创建只能通过根级别客户端完成
        temp_client = MilvusClient(uri=self.milvus_uri, timeout=30)
        try:
            databases = temp_client.list_databases()
            if self.database_name not in databases:
                logger.info("创建数据库: %s", self.database_name)
                temp_client.create_database(self.database_name)
            else:
                logger.info("数据库已存在: %s", self.database_name)
        finally:
            temp_client.close()

        self.client = MilvusClient(
            uri=self.milvus_uri,
            db_name=self.database_name,
            timeout=30,
        )
        logger.info("Milvus 客户端就绪, 数据库=%s", self.database_name)

    def setup_embeddings(self):
        """初始化嵌入模型。"""
        logger.info("正在初始化嵌入模型: %s", self.model_name)

        # 将模型缓存保留在项目内，使部署不依赖于全局缓存
        cache_dir = Path(__file__).resolve().parent.parent / "models" / "embeddings"
        cache_dir.mkdir(parents=True, exist_ok=True)
        logger.info("嵌入模型缓存目录: %s", cache_dir)

        os.environ["SENTENCE_TRANSFORMERS_HOME"] = str(cache_dir)

        model_kwargs = {"device": "cpu"}
        if has_local_model_cache(self.model_name, cache_dir):
            model_kwargs["local_files_only"] = True
            logger.info("使用本地嵌入模型缓存")

        self.embeddings = HuggingFaceEmbeddings(
            model_name=self.model_name,
            cache_folder=str(cache_dir),
            model_kwargs=model_kwargs,
            encode_kwargs={"normalize_embeddings": True},  # 归一化向量以便使用余弦相似度
        )

        logger.info("嵌入模型初始化完成")

    def setup_collection(self):
        """检查集合是否已存在。"""
        try:
            collections = self.client.list_collections()
            self.collection_exists = self.collection_name in collections

            if self.collection_exists:
                logger.info("集合已存在: %s", self.collection_name)
            else:
                logger.info("集合尚不存在: %s", self.collection_name)
        except MilvusException as exc:
            logger.error("检查集合状态失败: %s", exc)
            raise

    def create_collection(self):
        """创建集合并立即附加向量索引。"""
        if self.collection_exists:
            logger.info("集合已存在，跳过创建: %s", self.collection_name)
            return

        logger.info("正在创建集合: %s", self.collection_name)

        # 定义集合的字段结构
        fields = [
            FieldSchema(name="id", dtype=DataType.INT64, is_primary=True, auto_id=False),  # 主键ID
            FieldSchema(name="content", dtype=DataType.VARCHAR, max_length=65535),  # 文档内容
            FieldSchema(name="title", dtype=DataType.VARCHAR, max_length=512),  # 文档标题
            FieldSchema(name="file_name", dtype=DataType.VARCHAR, max_length=256),  # 文件名
            FieldSchema(name="file_extension", dtype=DataType.VARCHAR, max_length=50),  # 文件扩展名
            FieldSchema(name="domain", dtype=DataType.VARCHAR, max_length=100),  # 域名/来源
            FieldSchema(name="chunk_index", dtype=DataType.INT64),  # 文档块索引
            FieldSchema(name=VECTOR_FIELD_NAME, dtype=DataType.FLOAT_VECTOR, dim=self.embedding_dim),  # 向量字段
            FieldSchema(name="created_at", dtype=DataType.INT64),  # 创建时间戳
        ]

        schema = CollectionSchema(
            fields=fields,
            description="RAG 安全知识库",
            enable_dynamic_field=True,  # 启用动态字段支持
        )

        self.client.create_collection(
            collection_name=self.collection_name,
            schema=schema,
        )

        self.collection_exists = True
        # 没有索引的向量集合无法可靠加载进行搜索
        # 因此新集合在同一工作流中总是创建向量索引
        self.create_vector_index()
        logger.info("集合创建完成: %s", self.collection_name)

    def _build_vector_index_params(self):
        """为向量字段构建 Milvus 索引参数。"""
        index_params = self.client.prepare_index_params()
        index_params.add_index(
            field_name=VECTOR_FIELD_NAME,
            index_name=VECTOR_INDEX_NAME,
            index_type=VECTOR_INDEX_TYPE,
            metric_type=VECTOR_METRIC_TYPE,
            params=VECTOR_INDEX_CONFIG,
        )
        return index_params

    @staticmethod
    def _normalize_index_descriptions(index_info: Any) -> List[Dict[str, Any]]:
        """标准化不同 pymilvus 版本的 describe_index 输出。

        pymilvus 在此返回过单个字典和包装列表两种格式。
        """
        if isinstance(index_info, dict):
            descriptions = index_info.get("index_descriptions")
            if isinstance(descriptions, list):
                return [item for item in descriptions if isinstance(item, dict)]
            return [index_info]
        if isinstance(index_info, list):
            return [item for item in index_info if isinstance(item, dict)]
        return []

    def get_vector_index_names(self) -> List[str]:
        """返回属于向量字段的索引名称列表。"""
        if not self.collection_exists:
            return []

        vector_index_names: List[str] = []
        # list_indexes() 只提供名称；需要 describe_index() 来确认
        # 每个索引属于哪个字段
        index_names = self.client.list_indexes(collection_name=self.collection_name) or []

        for index_name in index_names:
            descriptions = self._normalize_index_descriptions(
                self.client.describe_index(
                    collection_name=self.collection_name,
                    index_name=index_name,
                )
            )
            for description in descriptions:
                field_name = description.get("field_name") or description.get("fieldName")
                description_index_name = (
                    description.get("index_name")
                    or description.get("indexName")
                    or index_name
                )
                if field_name == VECTOR_FIELD_NAME:
                    vector_index_names.append(description_index_name)

        return list(dict.fromkeys(vector_index_names))

    def create_vector_index(self):
        """创建向量索引（如果尚不存在）。"""
        if not self.collection_exists:
            raise ValueError("集合不存在，无法创建索引")

        index_names = self.get_vector_index_names()
        if index_names:
            logger.info("向量索引已存在: %s", index_names)
            return

        logger.info("正在为字段 '%s' 创建向量索引", VECTOR_FIELD_NAME)

        if self.is_collection_loaded():
            # Milvus 要求在重建索引之前释放已加载的集合
            logger.info("集合已加载，在构建索引前先释放")
            self.client.release_collection(collection_name=self.collection_name)

        self.client.create_index(
            collection_name=self.collection_name,
            index_params=self._build_vector_index_params(),
            sync=True,  # 同步等待索引创建完成
        )

        index_names = self.get_vector_index_names()
        if not index_names:
            raise RuntimeError(f"Milvus 未返回字段 '{VECTOR_FIELD_NAME}' 的索引信息")

        logger.info("向量索引创建成功: %s", index_names)

    def index_exists(self) -> bool:
        """当向量字段已有索引时返回 True。"""
        try:
            return len(self.get_vector_index_names()) > 0
        except Exception:
            return False

    def get_load_state(self) -> str:
        """返回集合的当前加载状态。"""
        if not self.collection_exists:
            return LoadState.NotExist.name

        state = self.client.get_load_state(collection_name=self.collection_name).get("state")
        if isinstance(state, LoadState):
            return state.name
        return str(state)

    def is_collection_loaded(self) -> bool:
        """当集合已加载到内存时返回 True。"""
        return self.get_load_state() == LoadState.Loaded.name

    def wait_for_collection_loaded(self, timeout: int = 120, poll_interval: float = 1.0):
        """等待集合完成加载。

        Args:
            timeout: 超时时间（秒）
            poll_interval: 轮询间隔（秒）

        Raises:
            RuntimeError: 当遇到意外的加载状态时
            TimeoutError: 当加载超时时
        """
        deadline = time.time() + timeout
        last_state = None

        while time.time() < deadline:
            load_info = self.client.get_load_state(collection_name=self.collection_name)
            state = load_info.get("state")
            state_name = state.name if isinstance(state, LoadState) else str(state)

            if state_name != last_state:
                logger.info("集合加载状态: %s", state_name)
                last_state = state_name

            if state_name == LoadState.Loaded.name:
                return

            # NotLoad -> Loading -> Loaded 是正常加载周期的预期状态变化
            if state_name in {LoadState.NotExist.name, LoadState.NotLoad.name, LoadState.Loading.name}:
                time.sleep(poll_interval)
                continue

            raise RuntimeError(f"意外的 Milvus 加载状态: {state_name}")

        raise TimeoutError(
            f"等待集合加载超时: {self.collection_name}, last_state={last_state}"
        )

    def ensure_collection_loaded(self, require_index: bool = True):
        """确保集合有向量索引并已加载。

        Args:
            require_index: 是否要求集合必须有向量索引
        """
        if not self.collection_exists:
            raise ValueError("集合不存在，请先构建索引")

        # 旧集合可能已包含数据但缺少向量索引
        if require_index and not self.index_exists():
            logger.info("向量索引缺失，在加载前创建")
            self.create_vector_index()

        current_state = self.get_load_state()
        if current_state == LoadState.Loaded.name:
            return

        if current_state == LoadState.Loading.name:
            self.wait_for_collection_loaded()
            return

        logger.info("正在将集合加载到内存: %s", self.collection_name)
        self.client.load_collection(collection_name=self.collection_name)
        self.wait_for_collection_loaded()

    def build_vector_index(self, chunks: List[Document]) -> "MilvusIndexModule":
        """将文档插入 Milvus 并确保集合可加载。

        Args:
            chunks: 要索引的文档块列表

        Returns:
            self，支持链式调用

        Raises:
            ValueError: 当文档块为空时
        """
        logger.info("正在构建向量索引...")

        if not chunks:
            raise ValueError("文档块不能为空")

        if not self.collection_exists:
            self.create_collection()

        logger.info("正在生成嵌入向量...")
        data = self._prepare_data(chunks)

        # 批量插入避免过大的负载，并在日志中显示进度
        batch_size = 100
        total_inserted = 0
        for i in range(0, len(data), batch_size):
            batch = data[i : i + batch_size]
            self.client.insert(collection_name=self.collection_name, data=batch)
            total_inserted += len(batch)
            logger.info("已插入 %s/%s 行", total_inserted, len(data))

        logger.info("正在将数据刷新到 Milvus...")
        self.client.flush(collection_name=self.collection_name)

        if not self.index_exists():
            self.create_vector_index()

        self.ensure_collection_loaded(require_index=True)

        logger.info("向量索引构建完成, chunks=%s", len(chunks))
        return self

    def _prepare_data(self, chunks: List[Document]) -> List[Dict[str, Any]]:
        """准备 Milvus 插入数据。

        Args:
            chunks: 要准备的文档块列表

        Returns:
            包含所有字段和嵌入向量的数据字典列表
        """
        texts = [chunk.page_content for chunk in chunks]
        embedding_list = self.embeddings.embed_documents(texts)
        current_time = int(time.time())

        data: List[Dict[str, Any]] = []
        for idx, (chunk, embedding) in enumerate(zip(chunks, embedding_list)):
            data.append(
                {
                    "id": idx + 1,
                    "content": chunk.page_content,
                    "title": (chunk.metadata.get("title", "") or "")[:512],
                    "file_name": (chunk.metadata.get("file_name", "") or "")[:256],
                    "file_extension": (chunk.metadata.get("file_extension", "") or "")[:50],
                    "domain": (chunk.metadata.get("domain", "") or "")[:100],
                    "chunk_index": chunk.metadata.get("chunk_index", idx),
                    VECTOR_FIELD_NAME: embedding,
                    "created_at": current_time,
                }
            )

        return data

    def add_documents(self, new_chunks: List[Document]):
        """向现有集合追加文档。

        Args:
            new_chunks: 要添加的新文档块列表

        Raises:
            ValueError: 当集合不存在时
        """
        if not self.collection_exists:
            raise ValueError("集合不存在，请先构建索引")

        logger.info("正在向集合添加 %s 个新文档块", len(new_chunks))

        # row_count 就足够了，因为 ID 是由本模块顺序分配的
        stats = self.client.get_collection_stats(collection_name=self.collection_name)
        max_id = int(stats.get("row_count", 0))

        texts = [chunk.page_content for chunk in new_chunks]
        embedding_list = self.embeddings.embed_documents(texts)
        current_time = int(time.time())

        data: List[Dict[str, Any]] = []
        for idx, (chunk, embedding) in enumerate(zip(new_chunks, embedding_list)):
            data.append(
                {
                    "id": max_id + idx + 1,
                    "content": chunk.page_content,
                    "title": (chunk.metadata.get("title", "") or "")[:512],
                    "file_name": (chunk.metadata.get("file_name", "") or "")[:256],
                    "file_extension": (chunk.metadata.get("file_extension", "") or "")[:50],
                    "domain": (chunk.metadata.get("domain", "") or "")[:100],
                    "chunk_index": chunk.metadata.get("chunk_index", max_id + idx),
                    VECTOR_FIELD_NAME: embedding,
                    "created_at": current_time,
                }
            )

        self.client.insert(collection_name=self.collection_name, data=data)
        self.client.flush(collection_name=self.collection_name)

        if not self.index_exists():
            self.create_vector_index()

        self.ensure_collection_loaded(require_index=True)
        logger.info("新文档添加成功")

    def similarity_search(self, query: str, k: int = 5, filter: str = None) -> List[Document]:
        """从 Milvus 搜索相似文档。

        Args:
            query: 查询文本
            k: 返回的最相似结果数量
            filter: 过滤表达式（可选）

        Returns:
            按相似度排序的文档列表

        Raises:
            ValueError: 当集合不存在时
        """
        if not self.collection_exists:
            raise ValueError("集合不存在，请先构建索引")

        self.ensure_collection_loaded(require_index=True)
        query_embedding = self.embeddings.embed_query(query)

        # 搜索使用与索引创建相同的度量类型，以保持分数一致
        results = self.client.search(
            collection_name=self.collection_name,
            data=[query_embedding],
            limit=k,
            output_fields=["content", "title", "file_name", "file_extension", "domain", "chunk_index"],
            filter=filter,
            anns_field=VECTOR_FIELD_NAME,
            search_params={
                "metric_type": VECTOR_METRIC_TYPE,
                "params": {"nprobe": 10},  # 搜索的聚类中心数量
            },
        )

        documents: List[Document] = []
        # 将 Milvus 实体转换回 LangChain 文档，供管道的其余部分使用
        for result in results[0]:
            documents.append(
                Document(
                    page_content=result["entity"]["content"],
                    metadata={
                        "title": result["entity"].get("title", ""),
                        "file_name": result["entity"].get("file_name", ""),
                        "file_extension": result["entity"].get("file_extension", ""),
                        "domain": result["entity"].get("domain", ""),
                        "chunk_index": result["entity"].get("chunk_index", 0),
                        "score": result["distance"],  # 相似度分数
                    },
                )
            )

        return documents

    def delete_collection(self):
        """从 Milvus 删除集合。"""
        if self.collection_exists:
            logger.info("正在删除集合: %s", self.collection_name)
            self.client.drop_collection(collection_name=self.collection_name)
            self.collection_exists = False
            logger.info("集合已删除")

    def get_collection_stats(self) -> Dict[str, Any]:
        """返回集合统计信息。

        Returns:
            包含集合存在状态、行数、名称、索引和加载状态的字典
        """
        if not self.collection_exists:
            return {"exists": False}

        stats = self.client.get_collection_stats(collection_name=self.collection_name)
        return {
            "exists": True,
            "row_count": stats.get("row_count", 0),
            "name": self.collection_name,
            "indexes": self.get_vector_index_names(),
            "load_state": self.get_load_state(),
        }

    def load_index(self):
        """加载已持久化的 Milvus 集合并确保可搜索。

        Returns:
            self，支持链式调用；如果集合不存在则返回 None
        """
        if not self.collection_exists:
            logger.info("集合不存在: %s", self.collection_name)
            return None

        self.ensure_collection_loaded(require_index=True)

        if self.has_data():
            logger.info("集合已就绪，包含现有数据: %s", self.collection_name)
        else:
            logger.info("集合存在但为空: %s", self.collection_name)

        return self

    def save_index(self):
        """将缓冲写入刷新到 Milvus。"""
        if self.collection_exists:
            self.client.flush(collection_name=self.collection_name)
            logger.info("数据已刷新到 Milvus")

    def has_data(self) -> bool:
        """当集合包含数据行时返回 True。"""
        if not self.collection_exists:
            return False

        stats = self.client.get_collection_stats(collection_name=self.collection_name)
        return int(stats.get("row_count", 0)) > 0

    def close(self):
        """关闭 Milvus 客户端。"""
        if self.client:
            self.client.close()
            logger.info("Milvus 客户端已关闭")
