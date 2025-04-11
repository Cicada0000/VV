import os
import json
import logging
import requests
import urllib.parse
import subprocess
import time
from flask import Flask, request, Response, jsonify
from flask_cors import CORS

try:
    from langchain.chat_models import init_chat_model
    from langchain_core.messages import HumanMessage, SystemMessage
    from langchain_core.tools import tool
    from duckduckgo_search import DDGS
    LANGCHAIN_AVAILABLE = True
except ImportError:
    LANGCHAIN_AVAILABLE = False

logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)

# 从环境变量获取API密钥，如果不存在则使用默认值
API_KEY = os.environ.get('SILICONFLOW_API_KEY', '')
BASE_URL = os.environ.get('SILICONFLOW_BASE_URL', 'https://api.siliconflow.cn/v1')
MODEL_NAME = os.environ.get('SILICONFLOW_MODEL', 'Qwen/Qwen2.5-7B-Instruct')

# 检查API密钥是否配置
if not API_KEY:
    logger.warning("环境变量SILICONFLOW_API_KEY未设置，AI检索功能可能不可用")

app = Flask(__name__)
CORS(app)

# 初始化AI相关资源
if LANGCHAIN_AVAILABLE:
    @tool
    def get_web_data(query: str) -> str:
        """
        使用DuckDuckGo搜索引擎查询网络信息。
        输入应为搜索查询字符串，将返回相关搜索结果。
        
        Args:
            query: 要搜索的查询字符串
            
        Returns:
            搜索结果列表或空列表(如果搜索失败)
        """
        try:
            results = DDGS().text(query, max_results=10)
            return results
        except Exception as e:
            return []

    try:
        if API_KEY:
            model = init_chat_model(MODEL_NAME, model_provider="openai", api_key=API_KEY, base_url=BASE_URL)
            tools = [get_web_data]
            llm_with_tools = model.bind_tools(tools)
        else:
            logger.error("无法初始化AI模型：缺少API密钥")
            LANGCHAIN_AVAILABLE = False
    except Exception as e:
        logger.error(f"初始化AI模型失败: {str(e)}")
        LANGCHAIN_AVAILABLE = False

def create_qwen_payload(messages, temperature=0.4, max_tokens=200):
    return {
        "model": MODEL_NAME,
        "messages": messages,
        "temperature": temperature,
        "max_tokens": max_tokens,
        "stream": False
    }

def query_qwen_model_with_web_data(query):
    """使用API查询AI获取回答和关键词"""
    try:
        if not API_KEY:
            logger.error("无法调用AI模型：未提供API密钥")
            return query, {"answer": "无法生成回答，API密钥未配置", "keywords": query}
            
        if not LANGCHAIN_AVAILABLE:
            # 如果langchain不可用，直接使用API调用
            web_results = []
        else:
            web_results = get_web_data(query)
        
        system_message = """你是一个专业的问答助手和关键词提取器。你有两个任务：
                        1. 回答用户的问题
                        2. 提取问题中的关键词并将英文术语翻译成中文
                        
                        回答问题时：
                        - 提供简洁明了的回答
                        - 基于事实和提供的信息
                        - 如果不确定，请说明
                        
                        提取关键词时：
                        - 从问题中提取5-8个最核心的关键词
                        - 将英文术语翻译成对应的中文术语
                        - 专有名词使用"中文含义"的格式
                        - 英文缩写展开并翻译
                        - 用顿号"、"分隔关键词
                        
                        你的回复格式必须是：
                        1. 先回答问题
                        2. 然后空一行
                        3. 最后一行以"关键词："开头，后跟提取的关键词列表"""

        user_message = f"""请回答这个问题，并提取其中的关键词：问题: {query}"""

        if web_results:
            web_content = str(web_results)
            if len(web_content) > 1000:
                web_content = web_content[:1000] + "..."
            user_message += f"\n\n以下是相关信息，可以参考：\n{web_content}"

        messages = [
            {"role": "system", "content": system_message},
            {"role": "user", "content": user_message}
        ]
        
        payload = create_qwen_payload(messages, temperature=0.4, max_tokens=800)
        
        headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {API_KEY}"
        }
        
        response = requests.post(
            f"{BASE_URL}/chat/completions",
            headers=headers,
            json=payload,
            timeout=30
        )
        
        if response.status_code != 200:
            logger.error(f"AI API返回错误: {response.status_code}, {response.text[:200]}")
            return query, {"answer": f"AI请求失败 (HTTP {response.status_code})", "keywords": query}
            
        result = response.json()
        full_response = result.get("choices", [{}])[0].get("message", {}).get("content", "")
        
        parts = full_response.split("\n\n")
        answer = parts[0]
        
        keywords_part = ""
        for part in parts:
            if "关键词：" in part:
                keywords_part = part.split("关键词：")[1].strip()
                break
        
        if not keywords_part:
            for part in parts:
                if "关键词" in part:
                    keywords_part = part.replace("关键词", "").replace(":", "").replace("：", "").strip()
                    break
        
        if not keywords_part:
            keywords_part = query
        
        # 提取搜索关键词
        search_keywords = " ".join(keywords_part.split("、")[:3])
        
        query_result = {
            "answer": answer,
            "keywords": keywords_part,
            "search_keywords": search_keywords
        }
        
        return search_keywords, query_result
        
    except Exception as e:
        logger.error(f"查询模型失败: {str(e)}")
        return query, {"answer": f"处理查询时出错: {str(e)}", "keywords": query}

def fetch_remote_search_results(search_query, min_ratio=50, min_similarity=0, max_results=20):
    """从远程API获取字幕搜索结果"""
    # 确保max_results不超过20
    if max_results > 20:
        max_results = 20
        
    # 最大尝试次数
    max_attempts = 3
    attempt = 0
    
    while attempt < max_attempts:
        attempt += 1
        try:
            encoded_query = urllib.parse.quote(search_query)
            url = f"https://subtitle-search.cicada000.workers.dev/search?query={encoded_query}&min_ratio={min_ratio}&min_similarity={min_similarity}&max_results={max_results}"
            
            logger.debug(f"请求远程搜索API (尝试 {attempt}/{max_attempts}): {url}")
            
            response = requests.get(url, timeout=30)
            
            if response.status_code != 200:
                logger.error(f"远程搜索API返回错误状态码: {response.status_code}")
                logger.error(f"响应内容: {response.text[:200]}")
                
                # 如果是参数错误，调整参数后重试
                if response.status_code == 400 and "max_results" in response.text and attempt < max_attempts:
                    max_results = 10  # 降低max_results
                    logger.warning(f"降低max_results到{max_results}后重试")
                    continue
                    
                # 其他错误，加大重试间隔
                if attempt < max_attempts:
                    time.sleep(1)  # 等待1秒后重试
                    continue
                    
                return []
            
            # 检查响应内容
            content = response.text.strip()
            logger.debug(f"远程搜索API响应长度: {len(content)}")
            if content:
                logger.debug(f"远程搜索API响应前100字符: {content[:100]}")
            
            # 如果响应为空
            if not content:
                logger.warning("远程搜索API返回空响应")
                if attempt < max_attempts:
                    time.sleep(1)
                    continue
                return []
                
            # 返回每行解析后的结果
            results = []
            for line in content.split('\n'):
                if not line.strip():
                    continue
                    
                # 尝试解析JSON，如果失败则直接保留原始行
                try:
                    json_obj = json.loads(line)
                    # 如果解析成功，检查是否含有必要字段
                    if isinstance(json_obj, dict) and ("filename" in json_obj or "type" in json_obj):
                        results.append(line)
                    else:
                        logger.warning(f"远程API返回的JSON缺少必要字段: {line[:100]}")
                except json.JSONDecodeError:
                    logger.warning(f"无法解析远程API返回的行: {line[:100]}")
                    # 仍然添加这一行，以便于后续处理
                    if "{" in line and "}" in line:
                        results.append(line)
            
            logger.debug(f"成功解析远程结果数量: {len(results)}")
            
            # 如果没有结果，但还有重试机会，尝试调整参数
            if not results and attempt < max_attempts:
                logger.warning(f"没有获取到结果，调整参数后重试")
                min_similarity = max(0, min_similarity - 0.1)  # 降低相似度要求
                continue
                
            return results
            
        except requests.exceptions.Timeout:
            logger.error(f"远程搜索API请求超时 (尝试 {attempt}/{max_attempts})")
            if attempt < max_attempts:
                time.sleep(1)
                continue
            return []
        except requests.exceptions.RequestException as e:
            logger.error(f"远程搜索API请求异常 (尝试 {attempt}/{max_attempts}): {str(e)}")
            if attempt < max_attempts:
                time.sleep(1)
                continue
            return []
        except Exception as e:
            logger.error(f"远程搜索失败 (尝试 {attempt}/{max_attempts}): {str(e)}")
            if attempt < max_attempts:
                time.sleep(1)
                continue
            return []
            
    return []

def run_rust_search(query, min_ratio, min_similarity, max_results):
    """调用Rust二进制文件进行搜索"""
    try:
        query_string = f"query={query}"
        query_string += f"&min_ratio={min_ratio}"
        query_string += f"&min_similarity={min_similarity}"
        query_string += f"&max_results={max_results}"
        
        # 判断是否在Vercel环境
        if 'VERCEL' in os.environ:
            # 在Vercel上，二进制文件在特定目录
            bin_path = os.path.join('/var/task', 'api', 'subtitle_search_api')
        else:
            # 本地环境
            bin_path = './api/subtitle_search_api'
        
        rust_process = subprocess.Popen(
            [bin_path],
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            bufsize=1
        )
        
        rust_process.stdin.write(query_string + '\n')
        rust_process.stdin.flush()
        
        results = []
        while True:
            line = rust_process.stdout.readline()
            if not line:
                break
            results.append(line.strip())
        
        # 检查是否有错误
        stderr_output = rust_process.stderr.read()
        if stderr_output:
            logger.error(f"Rust程序错误: {stderr_output}")
        
        rust_process.terminate()
        return results
        
    except Exception as e:
        logger.error(f"运行Rust搜索失败: {str(e)}")
        return []

@app.route('/search', methods=['GET'])
def search():
    query = request.args.get('query', '')
    
    if not query:
        return Response(json.dumps({
            "status": "error",
            "message": "搜索关键词不能为空"
        }, ensure_ascii=False), status=400, mimetype='application/json')
    
    try:
        # 获取并验证参数
        min_ratio = request.args.get('min_ratio', '50')
        min_similarity = request.args.get('min_similarity', '0.5')
        max_results = request.args.get('max_results', '20')  # 默认改为20
        rag = request.args.get('rag', 'false').lower() == 'true'
        
        try:
            min_ratio = float(min_ratio)
            min_similarity = float(min_similarity)
            max_results = int(max_results)
            
            # 限制max_results不超过20
            if max_results > 20:
                logger.warning(f"请求的max_results={max_results}超过远程API限制，将被限制为20")
                max_results = 20
            
            if min_ratio > 1 and min_ratio <= 100:
                # 如果min_ratio在1-100之间，转换为0-1范围
                min_ratio = min_ratio / 100
                
            if not (0 <= min_ratio <= 1) or not (0 <= min_similarity <= 1) or max_results <= 0:
                return Response(json.dumps({
                    "status": "error",
                    "message": "参数格式错误: min_ratio和min_similarity必须在0-1之间，max_results必须大于0且不超过20"
                }, ensure_ascii=False), status=400, mimetype='application/json')
                
        except ValueError:
            return Response(json.dumps({
                "status": "error",
                "message": "参数必须是有效的数字"
            }, ensure_ascii=False), status=400, mimetype='application/json')
        
        # 根据rag参数决定使用哪种搜索方式
        if rag:
            # 使用AI增强检索
            if not LANGCHAIN_AVAILABLE:
                # 如果在Vercel上，不支持langchain，fallback到远程API
                search_results = fetch_remote_search_results(query, min_ratio, min_similarity, max_results)
                
                def generate_remote_results():
                    for line in search_results:
                        yield line + '\n'
                        
                return app.response_class(
                    generate_remote_results(),
                    mimetype='application/json',
                    headers={
                        'X-Accel-Buffering': 'no',
                        'Cache-Control': 'no-cache'
                    }
                )
            
            # 使用AI生成搜索关键词
            search_query, query_result = query_qwen_model_with_web_data(query)
            
            # 使用远程API搜索结果
            remote_results = fetch_remote_search_results(search_query, min_ratio, min_similarity, max_results)
            
            if not remote_results and search_query != query:
                # 如果没有结果，使用原始查询再试一次
                remote_results = fetch_remote_search_results(query, min_ratio, min_similarity, max_results)
                
            # 记录日志用于调试
            logger.debug(f"AI生成的搜索查询: {search_query}")
            logger.debug(f"远程结果数量: {len(remote_results)}")
            if remote_results:
                logger.debug(f"第一个远程结果: {remote_results[0][:100]}...")
            
            # 格式化输出结果
            keywords = [kw.strip() for kw in query_result["keywords"].split("、")]
            
            keyword_result = {
                "type": "keywords",
                "keywords": keywords,
                "answer": query_result["answer"],
                "search_keywords": query_result["search_keywords"]
            }
            
            # 如果没有远程结果
            if not remote_results:
                def generate_keywords_only():
                    yield json.dumps(keyword_result, ensure_ascii=False) + '\n'
                    
                return app.response_class(
                    generate_keywords_only(),
                    mimetype='application/json',
                    headers={
                        'X-Accel-Buffering': 'no',
                        'Cache-Control': 'no-cache'
                    }
                )
            
            # 有远程结果，流式返回
            def generate_combined_results():
                yield json.dumps(keyword_result, ensure_ascii=False) + '\n'
                for result in remote_results:
                    yield result + '\n'
                    
            return app.response_class(
                generate_combined_results(),
                mimetype='application/json',
                headers={
                    'X-Accel-Buffering': 'no',
                    'Cache-Control': 'no-cache'
                }
            )
            
        else:
            # 使用Rust子进程搜索
            rust_results = run_rust_search(query, min_ratio, min_similarity, max_results)
            
            def generate_rust_results():
                first_item = True
                for line in rust_results:
                    if not first_item:
                        yield '\n'
                    first_item = False
                    yield line
                    
            return app.response_class(
                generate_rust_results(),
                mimetype='application/json',
                headers={
                    'X-Accel-Buffering': 'no',
                    'Cache-Control': 'no-cache'
                }
            )
                
    except Exception as e:
        logger.error(f"搜索处理异常: {str(e)}")
        return Response(json.dumps({
            "status": "error",
            "message": f"服务器内部错误: {str(e)}"
        }, ensure_ascii=False), status=500, mimetype='application/json')

@app.route('/health', methods=['GET'])
def health_check():
    return jsonify({"status": "ok"})

# Vercel serverless function入口点
def handler(event, context):
    return app(event, context)

# 为Vercel添加index路由
@app.route('/')
def index():
    return jsonify({
        "status": "ok",
        "message": "字幕搜索API服务正常运行",
        "endpoints": {
            "/search": "搜索API，参数：query, min_ratio, min_similarity, max_results, rag",
            "/health": "健康检查"
        }
    })

application = app  # 兼容WSGI服务器

if __name__ == '__main__':
    # 本地运行
    app.run(debug=True, host='0.0.0.0', port=8000)
