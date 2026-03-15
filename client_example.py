"""
SSE API 客户端示例
演示如何调用RAG系统的SSE流式API
"""

import requests
import json
from typing import Generator


class RAGClient:
    """RAG系统客户端"""

    def __init__(self, base_url: str = "http://localhost:8000", api_key: str = None):
        """
        初始化客户端

        Args:
            base_url: API服务器地址
            api_key: API密钥（如果需要）
        """
        self.base_url = base_url.rstrip('/')
        self.api_key = api_key
        self.session = requests.Session()
        if api_key:
            self.session.headers.update({"X-API-Key": api_key})

    def ask_stream(self, question: str, top_k: int = 3) -> Generator[dict, None, None]:
        """
        流式提问

        Args:
            question: 用户问题
            top_k: 检索文档数量

        Yields:
            服务器返回的SSE事件数据
        """
        url = f"{self.base_url}/api/ask"
        payload = {"question": question, "top_k": top_k}

        try:
            with self.session.post(
                url,
                json=payload,
                stream=True,
                timeout=120  # 长超时以支持流式传输
            ) as response:
                response.raise_for_status()

                # 解析SSE流
                for line in response.iter_lines():
                    if line:
                        line = line.decode('utf-8')
                        if line.startswith('event:'):
                            event_type = line[6:].strip()
                        elif line.startswith('data:'):
                            data = json.loads(line[5:].strip())
                            yield {"event": event_type, "data": data}

        except requests.exceptions.RequestException as e:
            yield {"event": "error", "data": {"error": str(e)}}

    def health_check(self) -> dict:
        """
        健康检查

        Returns:
            服务器状态
        """
        url = f"{self.base_url}/health"
        response = self.session.get(url)
        response.raise_for_status()
        return response.json()


def main():
    """示例用法"""

    # 从环境变量读取API密钥
    import os
    api_key = os.getenv("SERVER_API_KEY")

    # 创建客户端
    client = RAGClient(api_key=api_key)

    # 健康检查
    print("=== 健康检查 ===")
    health = client.health_check()
    print(f"服务器状态: {health}\n")

    # 流式提问
    print("=== 流式提问 ===")
    question = "什么是SQL注入攻击？"
    print(f"问题: {question}\n")
    print("回答:")

    full_answer = ""
    for event in client.ask_stream(question):
        event_type = event["event"]
        data = event["data"]

        if event_type == "metadata":
            print(f"\n[查询类型: {data.get('route_type')}]")

        elif event_type == "sources":
            sources = data.get("sources", [])
            print(f"[来源: {', '.join(sources)}]\n")

        elif event_type == "chunk":
            content = data.get("content", "")
            print(content, end="", flush=True)
            full_answer += content

        elif event_type == "done":
            print("\n\n[完成]")

        elif event_type == "error":
            print(f"\n[错误: {data.get('error')}]")

    print(f"\n完整回答长度: {len(full_answer)} 字符")


if __name__ == "__main__":
    main()
