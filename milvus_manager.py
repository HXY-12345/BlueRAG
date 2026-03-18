"""
Milvus管理工具
用于管理和测试Milvus向量数据库
"""

import os
from dotenv import load_dotenv
from pymilvus import MilvusClient

# 加载环境变量
load_dotenv()

MILVUS_URI = os.getenv("MILVUS_URI", "http://localhost:19530")
DATABASE_NAME = os.getenv("MILVUS_DATABASE", "rag_db")
COLLECTION_NAME = "rag_security_kb"


def create_database():
    """创建数据库"""
    client = MilvusClient(uri=MILVUS_URI)

    databases = client.list_databases()
    if DATABASE_NAME not in databases:
        print(f"创建数据库: {DATABASE_NAME}")
        client.create_database(DATABASE_NAME)
        print("✅ 数据库创建成功")
    else:
        print(f"数据库已存在: {DATABASE_NAME}")

    client.close()


def drop_database():
    """删除数据库（危险操作！）"""
    client = MilvusClient(uri=MILVUS_URI)

    confirm = input(f"⚠️ 确认删除数据库 '{DATABASE_NAME}'? (yes/no): ")
    if confirm.lower() == "yes":
        client.drop_database(DATABASE_NAME)
        print("✅ 数据库已删除")
    else:
        print("已取消")

    client.close()


def list_collections():
    """列出所有Collection"""
    client = MilvusClient(uri=MILVUS_URI, db_name=DATABASE_NAME)

    collections = client.list_collections()
    print(f"\n数据库 '{DATABASE_NAME}' 中的Collections:")

    if not collections:
        print("  (无)")
    else:
        for col in collections:
            stats = client.get_collection_stats(col)
            print(f"  - {col}: {stats.get('row_count', 0)} 条记录")

    client.close()


def drop_collection():
    """删除Collection"""
    client = MilvusClient(uri=MILVUS_URI, db_name=DATABASE_NAME)

    collections = client.list_collections()
    if COLLECTION_NAME not in collections:
        print(f"Collection不存在: {COLLECTION_NAME}")
        return

    confirm = input(f"⚠️ 确认删除Collection '{COLLECTION_NAME}'? (yes/no): ")
    if confirm.lower() == "yes":
        client.drop_collection(COLLECTION_NAME)
        print("✅ Collection已删除")
    else:
        print("已取消")

    client.close()


def show_collection_info():
    """显示Collection详细信息"""
    client = MilvusClient(uri=MILVUS_URI, db_name=DATABASE_NAME)

    if COLLECTION_NAME not in client.list_collections():
        print(f"Collection不存在: {COLLECTION_NAME}")
        return

    print(f"\n=== Collection: {COLLECTION_NAME} ===")

    # Schema信息
    schema = client.describe_collection(COLLECTION_NAME)
    print("\n字段:")
    for field in schema['fields']:
        field_type = field['type']
        if field_type == 101:  # FLOAT_VECTOR
            dim = field.get('dim', '?')
            print(f"  - {field['name']}: FLOAT_VECTOR({dim})")
        elif field_type == 5:  # INT64
            print(f"  - {field['name']}: INT64 {'(主键)' if field.get('is_primary') else ''}")
        elif field_type == 23:  # VARCHAR
            max_len = field.get('max_length', '?')
            print(f"  - {field['name']}: VARCHAR({max_len})")
        else:
            print(f"  - {field['name']}: {field_type}")

    # 统计信息
    stats = client.get_collection_stats(COLLECTION_NAME)
    print(f"\n记录数: {stats.get('row_count', 0)}")

    # 索引信息
    print("\n索引:")
    indexes = client.list_indexes(COLLECTION_NAME)
    for idx in indexes:
        idx_info = client.describe_index(COLLECTION_NAME, idx)
        print(f"  - {idx}: {idx_info.get('index_type', 'unknown')}")

    client.close()


def test_connection():
    """测试Milvus连接"""
    try:
        client = MilvusClient(uri=MILVUS_URI)
        databases = client.list_databases()
        print(f"✅ 连接成功!")
        print(f"  地址: {MILVUS_URI}")
        print(f"  数据库: {databases}")
        client.close()
    except Exception as e:
        print(f"❌ 连接失败: {e}")


def main():
    """主菜单"""
    print("=" * 50)
    print("       Milvus 管理工具")
    print("=" * 50)
    print(f"Milvus地址: {MILVUS_URI}")
    print(f"数据库名称: {DATABASE_NAME}")
    print(f"Collection: {COLLECTION_NAME}")
    print("=" * 50)

    while True:
        print("""
1. 测试连接
2. 创建数据库
3. 列出Collections
4. 显示Collection详情
5. 删除Collection
6. 删除数据库 (危险!)
7. 退出
        """)

        choice = input("请选择操作: ").strip()

        if choice == "1":
            test_connection()
        elif choice == "2":
            create_database()
        elif choice == "3":
            list_collections()
        elif choice == "4":
            show_collection_info()
        elif choice == "5":
            drop_collection()
        elif choice == "6":
            drop_database()
        elif choice == "7":
            print("再见!")
            break
        else:
            print("无效选择")


if __name__ == "__main__":
    main()
