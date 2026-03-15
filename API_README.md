# RAG系统 SSE API 使用指南

## 快速启动

### 1. 安装依赖

```bash
pip install -r requirements.txt
```

### 2. 配置环境变量

复制 `.env.example` 为 `.env` 并配置：

```bash
cp .env.example .env
```

编辑 `.env` 文件，填入必要的配置：

```env
# LLM API密钥（必需）
RAG_API_KEY=your_openrouter_api_key_here

# API服务器密钥（可选，留空则无需认证）
SERVER_API_KEY=your_server_api_key_here

# 服务器端口
SERVER_PORT=8000
```

### 3. 启动服务器

```bash
python api_server.py
```

服务器将在 `http://localhost:8000` 启动。

## API 端点

### 健康检查

```
GET /health
```

**响应示例：**

```json
{
  "status": "healthy",
  "rag_initialized": true
}
```

### SSE 流式问答

```
POST /api/ask
```

**请求头：**

| 名称 | 类型 | 必需 | 说明 |
|------|------|------|------|
| X-API-Key | string | 可选 | API密钥（如果配置了SERVER_API_KEY） |

**请求体：**

```json
{
  "question": "什么是SQL注入攻击？",
  "top_k": 3
}
```

| 参数 | 类型 | 必需 | 说明 |
|------|------|------|------|
| question | string | 是 | 用户问题 |
| top_k | integer | 否 | 检索文档数量（1-10），默认3 |

**SSE 事件类型：**

| 事件 | 说明 | 数据结构 |
|------|------|----------|
| metadata | 查询元数据 | `{"route_type": "detail\|general\|list", ...}` |
| sources | 来源文档信息 | `{"sources": ["文档1", "文档2"], "count": 2}` |
| chunk | 内容片段 | `{"content": "文本片段..."}` |
| done | 完成标志 | `{"status": "completed"}` |
| error | 错误信息 | `{"error": "错误描述"}` |

## 客户端示例

### Python 客户端

```python
from client_example import RAGClient

# 创建客户端
client = RAGClient(api_key="your_api_key")

# 流式提问
for event in client.ask_stream("什么是SQL注入？"):
    if event["event"] == "chunk":
        print(event["data"]["content"], end="", flush=True)
    elif event["event"] == "sources":
        print(f"\n来源: {event['data']['sources']}")
```

### JavaScript 客户端

```javascript
const client = new RAGClient('http://localhost:8000', 'your_api_key');

client.askStream('什么是SQL注入？', {
    onSources: (data) => {
        console.log('来源:', data.sources);
    },
    onChunk: (content) => {
        process.stdout.write(content);
    },
    onDone: (fullAnswer) => {
        console.log('\n完成!');
    }
});
```

### cURL 示例

```bash
curl -N -X POST http://localhost:8000/api/ask \
  -H "Content-Type: application/json" \
  -H "X-API-Key: your_api_key" \
  -d '{"question": "什么是SQL注入？", "top_k": 3}'
```

## 生产部署建议

### 1. 使用 Gunicorn + Uvicorn

```bash
pip install gunicorn

gunicorn api_server:app \
  --workers 4 \
  --worker-class uvicorn.workers.UvicornWorker \
  --bind 0.0.0.0:8000
```

### 2. Nginx 反向代理配置

```nginx
location /api/ask {
    proxy_pass http://127.0.0.1:8000;
    proxy_buffering off;
    proxy_cache off;
    proxy_set_header X-Real-IP $remote_addr;
    proxy_set_header X-Forwarded-For $proxy_add_x_forwarded_for;
    proxy_set_header Connection '';
    proxy_http_version 1.1;
    chunked_transfer_encoding off;
}
```

### 3. Docker 部署

```dockerfile
FROM python:3.10-slim

WORKDIR /app
COPY requirements.txt .
RUN pip install -r requirements.txt

COPY . .
CMD ["python", "api_server.py"]
```

## 错误处理

| HTTP状态码 | 说明 |
|-----------|------|
| 401 | API Key 无效 |
| 500 | 服务器内部错误 |

## 常见问题

**Q: 如何禁用API Key认证？**

A: 在 `.env` 中不设置 `SERVER_API_KEY` 或留空。

**Q: 连接超时怎么办？**

A: 增加客户端超时时间，服务器端处理大模型请求可能需要较长时间。

**Q: 如何限制跨域访问？**

A: 修改 `api_server.py` 中的 `allow_origins` 参数。
