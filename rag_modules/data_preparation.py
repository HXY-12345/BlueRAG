"""
数据准备模块 - 支持多格式文档加载和智能分块
"""

import hashlib
import logging
import uuid
from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Dict, Any, Tuple, Optional
from abc import ABC, abstractmethod

from langchain_core.documents import Document

logger = logging.getLogger(__name__)


@dataclass
class LoaderConfig:
    """文档加载器配置"""
    supported_formats: List[str] = field(default_factory=lambda: ['.pdf', '.docx', '.md', '.txt'])
    recursive: bool = True
    encoding: str = 'utf-8'
    max_file_size_mb: int = 50
    skip_corrupted: bool = True


@dataclass
class ChunkConfig:
    """文档分块配置"""
    strategy: str = "semantic"  # 'semantic', 'recursive', 'hybrid'

    # 语义分块参数
    semantic_breakpoint_threshold: float = 0.6
    semantic_chunk_size: int = 500

    # 递归分块参数
    recursive_chunk_size: int = 1000
    recursive_chunk_overlap: int = 200
    recursive_separators: List[str] = field(default_factory=lambda: ["\n\n", "\n", "。", ".", " ", ""])

    # Markdown结构分块参数
    markdown_headers: List[Tuple[str, str]] = field(default_factory=lambda: [
        ("#", "Header1"), ("##", "Header2"), ("###", "Header3")
    ])
    markdown_strip_headers: bool = False

    # 通用参数
    min_chunk_size: int = 50
    max_chunk_size: int = 3000
    enable_parent_mapping: bool = True


class DataPreparationModule:
    """数据准备模块 - 负责多格式文档加载、清洗和预处理"""

    def __init__(self, data_path: str, loader_config: LoaderConfig = None, chunk_config: ChunkConfig = None):
        """
        初始化数据准备模块

        Args:
            data_path: 数据文件夹路径
            loader_config: 文档加载器配置
            chunk_config: 文档分块配置
        """
        self.data_path = data_path
        self.loader_config = loader_config or LoaderConfig()
        self.chunk_config = chunk_config or ChunkConfig()
        self.documents: List[Document] = []
        self.chunks: List[Document] = []
        self.parent_child_map: Dict[str, str] = {}
        self.embeddings = None  # 用于语义分块

    def load_documents(self) -> List[Document]:
        """
        加载多格式文档

        Returns:
            加载的文档列表
        """
        logger.info(f"正在从 {self.data_path} 加载文档...")
        documents = []
        data_path_obj = Path(self.data_path)

        # 根据文件扩展名选择加载方式
        if self.loader_config.recursive:
            file_iterator = data_path_obj.rglob("*")
        else:
            file_iterator = data_path_obj.glob("*")

        for file_path in file_iterator:
            if file_path.is_file() and file_path.suffix.lower() in self.loader_config.supported_formats:
                # 检查文件大小
                file_size_mb = file_path.stat().st_size / (1024 * 1024)
                if file_size_mb > self.loader_config.max_file_size_mb:
                    logger.warning(f"文件 {file_path} 超过大小限制 ({file_size_mb:.2f}MB)，跳过")
                    continue

                doc = self._load_single_file(file_path)
                if doc:
                    self._enhance_metadata(doc)
                    documents.append(doc)

        self.documents = documents
        logger.info(f"成功加载 {len(documents)} 个文档")
        return documents

    def _load_single_file(self, file_path: Path) -> Optional[Document]:
        """
        加载单个文件 - 支持多种格式

        Args:
            file_path: 文件路径

        Returns:
            加载的文档对象
        """
        # 获取文件扩展名
        file_ext = file_path.suffix.lower()

        # 生成 parent_id
        try:
            data_root = Path(self.data_path).resolve()
            relative_path = file_path.resolve().relative_to(data_root).as_posix()
        except Exception:
            relative_path = file_path.as_posix()
        parent_id = hashlib.md5(relative_path.encode("utf-8")).hexdigest()

        # 根据文件类型选择加载方式
        content = None

        # Markdown 和 TXT 文件 - 直接读取
        if file_ext in ['.md', '.txt']:
            try:
                with open(file_path, 'r', encoding=self.loader_config.encoding) as f:
                    content = f.read()
            except UnicodeDecodeError:
                # 尝试其他编码
                try:
                    with open(file_path, 'r', encoding='gbk') as f:
                        content = f.read()
                except Exception as e:
                    logger.warning(f"无法读取文件 {file_path}: {e}")
                    return None

        # PDF 文件 - 使用 PyPDFLoader
        elif file_ext == '.pdf':
            try:
                from langchain_community.document_loaders import PyPDFLoader

                loader = PyPDFLoader(str(file_path))
                pages = loader.load()

                if not pages:
                    logger.warning(f"文件 {file_path} 未能解析出任何内容")
                    return None

                content = "\n\n".join([page.page_content for page in pages])

            except ImportError:
                logger.warning(f"未安装 PyPDFLoader，无法加载 {file_path}")
                return None
            except Exception as e:
                if self.loader_config.skip_corrupted:
                    logger.warning(f"读取文件 {file_path} 失败: {e}")
                    return None
                else:
                    raise

        # DOCX 文件 - 使用 Docx2txtLoader
        elif file_ext == '.docx':
            try:
                from langchain_community.document_loaders import Docx2txtLoader

                loader = Docx2txtLoader(str(file_path))
                documents = loader.load()

                if not documents:
                    logger.warning(f"文件 {file_path} 未能解析出任何内容")
                    return None

                content = "\n\n".join([doc.page_content for doc in documents])

            except ImportError:
                logger.warning(f"未安装 Docx2txtLoader，无法加载 {file_path}")
                return None
            except Exception as e:
                if self.loader_config.skip_corrupted:
                    logger.warning(f"读取文件 {file_path} 失败: {e}")
                    return None
                else:
                    raise

        # 不支持的文件类型
        else:
            logger.warning(f"不支持的文件类型: {file_ext}")
            return None

        if not content:
            return None

        return Document(
            page_content=content,
            metadata={
                "source": str(file_path),
                "parent_id": parent_id,
                "doc_type": "parent"
            }
        )

    def _enhance_metadata(self, doc: Document) -> None:
        """
        增强文档元数据 - 通用文件元数据

        Args:
            doc: 需要增强元数据的文档
        """
        file_path = Path(doc.metadata.get('source', ''))

        # 基本文件信息
        doc.metadata['file_name'] = file_path.stem
        doc.metadata['file_extension'] = file_path.suffix.lower()
        doc.metadata['file_size_bytes'] = file_path.stat().st_size if file_path.exists() else 0
        doc.metadata['file_size_kb'] = round(doc.metadata['file_size_bytes'] / 1024, 2)

        # 相对路径和结构信息
        try:
            data_root = Path(self.data_path).resolve()
            relative_path = file_path.resolve().relative_to(data_root).as_posix()
            doc.metadata['relative_path'] = relative_path
            doc.metadata['folder_depth'] = len(relative_path.split('/')) - 1
        except Exception:
            doc.metadata['relative_path'] = file_path.as_posix()
            doc.metadata['folder_depth'] = 0

        # 从内容中提取标题
        doc.metadata['title'] = self._extract_title_from_content(doc.page_content, file_path.stem)

    def _extract_title_from_content(self, content: str, default_title: str) -> str:
        """
        从内容中提取标题

        Args:
            content: 文档内容
            default_title: 默认标题（文件名）

        Returns:
            提取的标题
        """
        # 尝试从第一行获取标题（Markdown格式）
        lines = content.strip().split('\n')
        for line in lines[:5]:  # 只检查前5行
            line = line.strip()
            if line.startswith('#'):
                # 移除 # 符号和空格
                title = line.lstrip('#').strip()
                if title:
                    return title

        # 如果没有找到Markdown标题，使用文件名
        return default_title

    def set_embeddings(self, embeddings):
        """
        设置嵌入模型（用于语义分块）

        Args:
            embeddings: 嵌入模型实例
        """
        self.embeddings = embeddings
        logger.info("已设置嵌入模型，可用于语义分块")

    def chunk_documents(self) -> List[Document]:
        """
        文档分块 - 支持多种策略

        Returns:
            分块后的文档列表
        """
        if not self.documents:
            raise ValueError("请先加载文档")

        strategy = self.chunk_config.strategy
        logger.info(f"使用 {strategy} 分块策略...")

        if strategy == 'semantic':
            chunks = self._semantic_chunking()
        elif strategy == 'recursive':
            chunks = self._recursive_chunking()
        elif strategy == 'hybrid':
            chunks = self._hybrid_chunking()
        elif strategy == 'markdown':
            chunks = self._markdown_chunking()
        else:
            logger.warning(f"未知的分块策略 '{strategy}'，使用递归分块")
            chunks = self._recursive_chunking()

        # 添加基础元数据
        for i, chunk in enumerate(chunks):
            if 'chunk_id' not in chunk.metadata:
                chunk.metadata['chunk_id'] = str(uuid.uuid4())
            chunk.metadata['batch_index'] = i
            chunk.metadata['chunk_size'] = len(chunk.page_content)

        self.chunks = chunks
        logger.info(f"分块完成，共生成 {len(chunks)} 个chunk")
        return chunks

    def _semantic_chunking(self) -> List[Document]:
        """
        语义分块策略

        Returns:
            分块后的文档列表
        """
        if not self.embeddings:
            logger.warning("未设置嵌入模型，回退到递归分块")
            return self._recursive_chunking()

        try:
            from langchain_experimental.text_splitter import SemanticChunker
        except ImportError:
            logger.warning("langchain-experimental 未安装，回退到递归分块")
            return self._recursive_chunking()

        splitter = SemanticChunker(
            embeddings=self.embeddings,
            breakpoint_threshold_type="percentile",
            breakpoint_threshold_amount=self.chunk_config.semantic_breakpoint_threshold
        )

        all_chunks = []
        for doc in self.documents:
            try:
                doc_chunks = splitter.split_text(doc.page_content)
                for i, chunk_text in enumerate(doc_chunks):
                    chunk = Document(page_content=chunk_text, metadata={})
                    chunk.metadata.update(doc.metadata)
                    chunk.metadata.update({
                        "chunk_id": str(uuid.uuid4()),
                        "parent_id": doc.metadata["parent_id"],
                        "doc_type": "child",
                        "chunk_index": i
                    })
                    if self.chunk_config.enable_parent_mapping:
                        self.parent_child_map[chunk.metadata["chunk_id"]] = doc.metadata["parent_id"]
                    all_chunks.append(chunk)
            except Exception as e:
                logger.warning(f"语义分块失败: {e}，使用递归分块")
                # 回退到递归分块
                doc_chunks = self._split_single_document_recursive(doc)
                all_chunks.extend(doc_chunks)

        return all_chunks

    def _recursive_chunking(self) -> List[Document]:
        """
        递归分块策略

        Returns:
            分块后的文档列表
        """
        from langchain_text_splitters import RecursiveCharacterTextSplitter

        splitter = RecursiveCharacterTextSplitter(
            chunk_size=self.chunk_config.recursive_chunk_size,
            chunk_overlap=self.chunk_config.recursive_chunk_overlap,
            separators=self.chunk_config.recursive_separators,
            length_function=len,
        )

        all_chunks = []
        for doc in self.documents:
            doc_chunks = self._split_single_document_recursive(doc, splitter)
            all_chunks.extend(doc_chunks)

        return all_chunks

    def _split_single_document_recursive(self, doc: Document, splitter=None) -> List[Document]:
        """
        对单个文档进行递归分块

        Args:
            doc: 文档对象
            splitter: 可选的分词器

        Returns:
            分块后的文档列表
        """
        if splitter is None:
            from langchain_text_splitters import RecursiveCharacterTextSplitter
            splitter = RecursiveCharacterTextSplitter(
                chunk_size=self.chunk_config.recursive_chunk_size,
                chunk_overlap=self.chunk_config.recursive_chunk_overlap,
                separators=self.chunk_config.recursive_separators,
                length_function=len,
            )

        doc_chunks = splitter.split_documents([doc])
        for i, chunk in enumerate(doc_chunks):
            chunk.metadata["chunk_id"] = str(uuid.uuid4())
            chunk.metadata["parent_id"] = doc.metadata["parent_id"]
            chunk.metadata["doc_type"] = "child"
            chunk.metadata["chunk_index"] = i
            if self.chunk_config.enable_parent_mapping:
                self.parent_child_map[chunk.metadata["chunk_id"]] = doc.metadata["parent_id"]

        return doc_chunks

    def _hybrid_chunking(self) -> List[Document]:
        """
        混合分块策略 - 结合语义和递归分块

        Returns:
            分块后的文档列表
        """
        # 先尝试语义分块，如果生成的块太大则使用递归分块
        all_chunks = []

        for doc in self.documents:
            if self.embeddings:
                try:
                    from langchain_experimental.text_splitter import SemanticChunker
                    splitter = SemanticChunker(
                        embeddings=self.embeddings,
                        breakpoint_threshold_type="percentile",
                        breakpoint_threshold_amount=self.chunk_config.semantic_breakpoint_threshold
                    )
                    doc_chunks = splitter.split_text(doc.page_content)

                    # 检查是否有超过最大大小的块
                    need_refinement = False
                    refined_chunks = []
                    for chunk_text in doc_chunks:
                        if len(chunk_text) > self.chunk_config.max_chunk_size:
                            need_refinement = True
                            # 对大块使用递归分块
                            from langchain_text_splitters import RecursiveCharacterTextSplitter
                            recursive_splitter = RecursiveCharacterTextSplitter(
                                chunk_size=self.chunk_config.recursive_chunk_size,
                                chunk_overlap=self.chunk_config.recursive_chunk_overlap,
                                separators=self.chunk_config.recursive_separators,
                                length_function=len,
                            )
                            sub_chunks = recursive_splitter.split_text(chunk_text)
                            refined_chunks.extend(sub_chunks)
                        else:
                            refined_chunks.append(chunk_text)

                    doc_chunks = refined_chunks

                except Exception as e:
                    logger.warning(f"混合分块中的语义分块失败: {e}，使用递归分块")
                    doc_chunks = self._split_single_document_recursive(doc)
            else:
                # 没有嵌入模型，直接使用递归分块
                doc_chunks = self._split_single_document_recursive(doc)

            # 添加元数据
            for i, chunk_item in enumerate(doc_chunks):
                if isinstance(chunk_item, Document):
                    chunk = chunk_item
                else:
                    chunk = Document(page_content=chunk_item, metadata={})
                    chunk.metadata.update(doc.metadata)
                    chunk.metadata.update({
                        "chunk_id": str(uuid.uuid4()),
                        "parent_id": doc.metadata["parent_id"],
                        "doc_type": "child",
                        "chunk_index": i
                    })
                if self.chunk_config.enable_parent_mapping:
                    self.parent_child_map[chunk.metadata["chunk_id"]] = doc.metadata["parent_id"]
                all_chunks.append(chunk)

        return all_chunks

    def _markdown_chunking(self) -> List[Document]:
        """
        Markdown结构分块策略

        Returns:
            分块后的文档列表
        """
        from langchain_text_splitters import MarkdownHeaderTextSplitter

        markdown_splitter = MarkdownHeaderTextSplitter(
            headers_to_split_on=self.chunk_config.markdown_headers,
            strip_headers=self.chunk_config.markdown_strip_headers
        )

        all_chunks = []

        for doc in self.documents:
            try:
                md_chunks = markdown_splitter.split_text(doc.page_content)

                for i, chunk in enumerate(md_chunks):
                    chunk.metadata.update(doc.metadata)
                    chunk.metadata.update({
                        "chunk_id": str(uuid.uuid4()),
                        "parent_id": doc.metadata["parent_id"],
                        "doc_type": "child",
                        "chunk_index": i
                    })
                    if self.chunk_config.enable_parent_mapping:
                        self.parent_child_map[chunk.metadata["chunk_id"]] = doc.metadata["parent_id"]
                    all_chunks.append(chunk)

            except Exception as e:
                logger.warning(f"Markdown分块失败: {e}，使用递归分块")
                doc_chunks = self._split_single_document_recursive(doc)
                all_chunks.extend(doc_chunks)

        return all_chunks

    def get_statistics(self) -> Dict[str, Any]:
        """
        获取数据统计信息

        Returns:
            统计信息字典
        """
        if not self.documents:
            return {}

        file_types = {}
        total_size = 0
        folder_depths = {}

        for doc in self.documents:
            ext = doc.metadata.get('file_extension', 'unknown')
            file_types[ext] = file_types.get(ext, 0) + 1
            total_size += doc.metadata.get('file_size_bytes', 0)

            depth = doc.metadata.get('folder_depth', 0)
            folder_depths[depth] = folder_depths.get(depth, 0) + 1

        return {
            'total_documents': len(self.documents),
            'total_chunks': len(self.chunks),
            'total_size_mb': round(total_size / 1024 / 1024, 2),
            'file_types': file_types,
            'folder_depths': folder_depths,
            'avg_chunk_size': sum(chunk.metadata.get('chunk_size', 0) for chunk in self.chunks) / len(self.chunks) if self.chunks else 0,
            'chunk_strategy': self.chunk_config.strategy
        }

    def get_supported_file_types(self) -> List[str]:
        """
        获取支持的文件类型列表

        Returns:
            支持的文件扩展名列表
        """
        return self.loader_config.supported_formats

    def export_metadata(self, output_path: str):
        """
        导出元数据到JSON文件

        Args:
            output_path: 输出文件路径
        """
        import json

        metadata_list = []
        for doc in self.documents:
            metadata_list.append({
                'source': doc.metadata.get('source'),
                'file_name': doc.metadata.get('file_name'),
                'title': doc.metadata.get('title'),
                'file_extension': doc.metadata.get('file_extension'),
                'file_size_kb': doc.metadata.get('file_size_kb'),
                'folder_depth': doc.metadata.get('folder_depth'),
                'relative_path': doc.metadata.get('relative_path'),
                'content_length': len(doc.page_content)
            })

        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(metadata_list, f, ensure_ascii=False, indent=2)

        logger.info(f"元数据已导出到: {output_path}")

    def get_parent_documents(self, child_chunks: List[Document]) -> List[Document]:
        """
        根据子块获取对应的父文档（智能去重）

        Args:
            child_chunks: 检索到的子块列表

        Returns:
            对应的父文档列表（去重，按相关性排序）
        """
        # 统计每个父文档被匹配的次数（相关性指标）
        parent_relevance = {}
        parent_docs_map = {}

        # 收集所有相关的父文档ID和相关性分数
        for chunk in child_chunks:
            parent_id = chunk.metadata.get("parent_id")
            if parent_id:
                # 增加相关性计数
                parent_relevance[parent_id] = parent_relevance.get(parent_id, 0) + 1

                # 缓存父文档（避免重复查找）
                if parent_id not in parent_docs_map:
                    for doc in self.documents:
                        if doc.metadata.get("parent_id") == parent_id:
                            parent_docs_map[parent_id] = doc
                            break

        # 按相关性排序（匹配次数多的排在前面）
        sorted_parent_ids = sorted(parent_relevance.keys(),
                                   key=lambda x: parent_relevance[x],
                                   reverse=True)

        # 构建去重后的父文档列表
        parent_docs = []
        for parent_id in sorted_parent_ids:
            if parent_id in parent_docs_map:
                parent_docs.append(parent_docs_map[parent_id])

        # 收集父文档名称和相关性信息用于日志
        parent_info = []
        for doc in parent_docs:
            title = doc.metadata.get('title', doc.metadata.get('file_name', '未知文档'))
            parent_id = doc.metadata.get('parent_id')
            relevance_count = parent_relevance.get(parent_id, 0)
            parent_info.append(f"{title}({relevance_count}块)")

        logger.info(f"从 {len(child_chunks)} 个子块中找到 {len(parent_docs)} 个去重父文档: {', '.join(parent_info)}")
        return parent_docs
