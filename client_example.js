/**
 * SSE API JavaScript客户端示例
 * 演示如何在浏览器中使用RAG系统的SSE流式API
 */

class RAGClient {
    /**
     * @param {string} baseUrl - API服务器地址
     * @param {string} apiKey - API密钥（如果需要）
     */
    constructor(baseUrl = 'http://localhost:8000', apiKey = null) {
        this.baseUrl = baseUrl.replace(/\/$/, '');
        this.apiKey = apiKey;
    }

    /**
     * 流式提问
     * @param {string} question - 用户问题
     * @param {Object} options - 配置选项
     * @param {number} options.topK - 检索文档数量
     * @param {Function} options.onChunk - 接收内容片段的回调
     * @param {Function} options.onSources - 接收来源信息的回调
     * @param {Function} options.onMetadata - 接收元数据的回调
     * @param {Function} options.onDone - 完成的回调
     * @param {Function} options.onError - 错误的回调
     * @returns {EventSource} - EventSource实例，可用于关闭连接
     */
    askStream(question, options = {}) {
        const {
            topK = 3,
            onChunk = null,
            onSources = null,
            onMetadata = null,
            onDone = null,
            onError = null
        } = options;

        // 构建URL和请求体
        // 由于EventSource不支持POST，我们需要使用fetch实现
        return this._streamWithFetch(question, topK, {
            onChunk, onSources, onMetadata, onDone, onError
        });
    }

    /**
     * 使用fetch实现SSE流式请求
     */
    _streamWithFetch(question, topK, callbacks) {
        const url = `${this.baseUrl}/api/ask`;
        const headers = {
            'Content-Type': 'application/json'
        };
        if (this.apiKey) {
            headers['X-API-Key'] = this.apiKey;
        }

        let controller = new AbortController();
        let fullAnswer = '';

        fetch(url, {
            method: 'POST',
            headers: headers,
            body: JSON.stringify({ question, top_k: topK }),
            signal: controller.signal
        }).then(response => {
            if (!response.ok) {
                throw new Error(`HTTP error! status: ${response.status}`);
            }

            const reader = response.body.getReader();
            const decoder = new TextDecoder();
            let buffer = '';
            let currentEventType = '';

            function processText(text) {
                const lines = text.split('\n');
                for (const line of lines) {
                    if (line.startsWith('event: ')) {
                        currentEventType = line.slice(7).trim();
                    } else if (line.startsWith('data: ')) {
                        const data = JSON.parse(line.slice(6));
                        handleEvent(currentEventType, data);
                    }
                }
            }

            function handleEvent(eventType, data) {
                switch (eventType) {
                    case 'metadata':
                        callbacks.onMetadata?.(data);
                        break;
                    case 'sources':
                        callbacks.onSources?.(data);
                        break;
                    case 'chunk':
                        fullAnswer += data.content;
                        callbacks.onChunk?.(data.content, fullAnswer);
                        break;
                    case 'done':
                        callbacks.onDone?.(fullAnswer);
                        break;
                    case 'error':
                        callbacks.onError?.(data.error);
                        break;
                }
            }

            function read() {
                reader.read().then(({ done, value }) => {
                    if (done) {
                        return;
                    }
                    buffer += decoder.decode(value, { stream: true });
                    const lines = buffer.split('\n');
                    buffer = lines.pop() || '';
                    processText(lines.join('\n'));
                    read();
                });
            }

            read();
        }).catch(error => {
            callbacks.onError?.(error.message);
        });

        // 返回控制对象
        return {
            close: () => controller.abort(),
            get fullAnswer() { return fullAnswer; }
        };
    }

    /**
     * 健康检查
     */
    async healthCheck() {
        const response = await fetch(`${this.baseUrl}/health`);
        if (!response.ok) {
            throw new Error(`HTTP error! status: ${response.status}`);
        }
        return await response.json();
    }
}

// ========== 使用示例 ==========

// 创建客户端实例
const client = new RAGClient('http://localhost:8000', 'your-api-key');

// 流式提问示例
async function example() {
    console.log('=== 流式提问示例 ===');

    const question = '什么是SQL注入攻击？';
    console.log('问题:', question);
    console.log('\n回答:');

    const stream = client.askStream(question, {
        topK: 3,
        onMetadata: (data) => {
            console.log(`\n[查询类型: ${data.route_type}]`);
        },
        onSources: (data) => {
            console.log(`[来源: ${data.sources.join(', ')}]\n`);
        },
        onChunk: (content, fullAnswer) => {
            process.stdout.write(content);
        },
        onDone: (fullAnswer) => {
            console.log('\n\n[完成]');
            console.log(`完整回答长度: ${fullAnswer.length} 字符`);
        },
        onError: (error) => {
            console.error('\n[错误:', error, ']');
        }
    });

    // 如需中断连接
    // stream.close();
}

// HTML示例
function htmlExample() {
    return `
<!DOCTYPE html>
<html>
<head>
    <title>RAG知识库助手</title>
    <style>
        body { font-family: Arial, sans-serif; max-width: 800px; margin: 50px auto; padding: 20px; }
        .chat-container { border: 1px solid #ddd; border-radius: 8px; padding: 20px; }
        #output { min-height: 100px; white-space: pre-wrap; line-height: 1.6; }
        input[type="text"] { width: 70%; padding: 10px; }
        button { padding: 10px 20px; cursor: pointer; }
        .sources { font-size: 12px; color: #666; margin-top: 10px; }
    </style>
</head>
<body>
    <div class="chat-container">
        <h2>RAG知识库助手</h2>
        <input type="text" id="question" placeholder="输入问题..." />
        <button onclick="askQuestion()">提问</button>
        <div id="output"></div>
        <div id="sources" class="sources"></div>
    </div>

    <script src="rag-client.js"></script>
    <script>
        const client = new RAGClient('http://localhost:8000');

        function askQuestion() {
            const question = document.getElementById('question').value;
            const output = document.getElementById('output');
            const sources = document.getElementById('sources');

            output.textContent = '';
            sources.textContent = '';

            const stream = client.askStream(question, {
                onSources: (data) => {
                    sources.textContent = '来源: ' + data.sources.join(', ');
                },
                onChunk: (content) => {
                    output.textContent += content;
                },
                onError: (error) => {
                    output.textContent = '错误: ' + error;
                }
            });
        }
    </script>
</body>
</html>
    `;
}

// 导出（Node.js环境）
if (typeof module !== 'undefined' && module.exports) {
    module.exports = { RAGClient, htmlExample };
}
