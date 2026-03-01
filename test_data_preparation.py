"""
测试 DataPreparationModule 功能
"""
import os
import sys
from pathlib import Path

# 设置环境变量（减少内存使用）
os.environ['OPENBLAS_NUM_THREADS'] = '1'

sys.path.insert(0, str(Path(__file__).parent))


def test_data_preparation():
    """测试数据准备模块的三个核心功能"""
    print("=" * 60)
    print("DataPreparationModule 功能测试")
    print("=" * 60)

    from rag_modules.data_preparation import DataPreparationModule, LoaderConfig, ChunkConfig

    # 配置
    data_path = "./data"
    print(f"\n数据路径: {data_path}")

    # ========== 测试 1: 初始化 ==========
    print("\n[测试 1] 初始化 DataPreparationModule")
    print("-" * 40)

    try:
        # 创建配置
        loader_config = LoaderConfig(
            supported_formats=['.pdf', '.docx', '.md', '.txt'],
            recursive=True,
            max_file_size_mb=50
        )

        chunk_config = ChunkConfig(
            strategy="semantic",
            semantic_chunk_size=500,
            semantic_breakpoint_threshold=0.6
        )

        # 初始化模块
        data_module = DataPreparationModule(
            data_path=data_path,
            loader_config=loader_config,
            chunk_config=chunk_config
        )

        print("OK: 模块初始化成功")
        print(f"  - 支持格式: {loader_config.supported_formats}")
        print(f"  - 分块策略: {chunk_config.strategy}")
        print(f"  - 块大小: {chunk_config.recursive_chunk_size}")

    except Exception as e:
        print(f"FAIL: 初始化失败 - {e}")
        import traceback
        traceback.print_exc()
        return

    # ========== 测试 2: 加载文档 ==========
    print("\n[测试 2] 加载文档 (load_documents)")
    print("-" * 40)

    try:
        documents = data_module.load_documents()

        print(f"OK: 成功加载 {len(documents)} 个文档")

        # 按类型统计
        type_count = {}
        for doc in documents:
            ext = doc.metadata.get('file_extension', 'unknown')
            type_count[ext] = type_count.get(ext, 0) + 1

        print(f"  文件类型分布:")
        for ext, count in type_count.items():
            print(f"    - {ext}: {count} 个")

        # 显示前3个文档信息
        print(f"\n  前3个文档详情:")
        for i, doc in enumerate(documents[:3], 1):
            print(f"\n    [{i}] {doc.metadata.get('file_name', 'unknown')}")
            print(f"        扩展名: {doc.metadata.get('file_extension', 'unknown')}")
            print(f"        大小: {doc.metadata.get('file_size_kb', 0)} KB")
            print(f"        标题: {doc.metadata.get('title', 'unknown')[:40]}...")
            print(f"        内容长度: {len(doc.page_content)} 字符")

    except Exception as e:
        print(f"FAIL: 加载文档失败 - {e}")
        import traceback
        traceback.print_exc()
        return

    # ========== 测试 3: 文本分块 ==========
    print("\n[测试 3] 文本分块 (chunk_documents)")
    print("-" * 40)

    try:
        chunks = data_module.chunk_documents()

        print(f"OK: 成功生成 {len(chunks)} 个文本块")

        # 统计块大小
        chunk_sizes = [c.metadata.get('chunk_size', 0) for c in chunks]
        if chunk_sizes:
            print(f"  块大小统计:")
            print(f"    - 最小: {min(chunk_sizes)} 字符")
            print(f"    - 最大: {max(chunk_sizes)} 字符")
            print(f"    - 平均: {sum(chunk_sizes) // len(chunk_sizes)} 字符")

        # 显示前3个块
        print(f"\n  前3个文本块详情:")
        for i, chunk in enumerate(chunks[:3], 1):
            print(f"\n    [块 {i}]")
            print(f"        来源: {chunk.metadata.get('file_name', 'unknown')}")
            print(f"        大小: {chunk.metadata.get('chunk_size', 0)} 字符")
            print(f"        父文档ID: {chunk.metadata.get('parent_id', 'unknown')[:16]}...")
            print(f"        内容预览: {chunk.page_content[:80].replace(chr(10), ' ')}...")

    except Exception as e:
        print(f"FAIL: 分块失败 - {e}")
        import traceback
        traceback.print_exc()
        return

    # ========== 测试 4: 获取统计信息 ==========
    print("\n[测试 4] 获取统计信息 (get_statistics)")
    print("-" * 40)

    try:
        stats = data_module.get_statistics()

        print("OK: 统计信息")
        for key, value in stats.items():
            print(f"  - {key}: {value}")

    except Exception as e:
        print(f"FAIL: 获取统计失败 - {e}")

    # ========== 测试 5: 父子文档映射 ==========
    print("\n[测试 5] 父子文档映射 (get_parent_documents)")
    print("-" * 40)

    try:
        # 取前3个chunk，获取对应的父文档
        test_chunks = chunks[:3]
        parent_docs = data_module.get_parent_documents(test_chunks)

        print(f"OK: 从 {len(test_chunks)} 个子块获取到 {len(parent_docs)} 个父文档")

        for doc in parent_docs:
            title = doc.metadata.get('title', doc.metadata.get('file_name', 'unknown'))
            print(f"  - {title}")

    except Exception as e:
        print(f"FAIL: 获取父文档失败 - {e}")

    print("\n" + "=" * 60)
    print("所有测试完成!")
    print("=" * 60)


if __name__ == "__main__":
    test_data_preparation()
