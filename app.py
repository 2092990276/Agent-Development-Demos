import tiktoken
import os
import streamlit as st
import json
from openai import OpenAI
from langchain_community.document_loaders import TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain_openai import OpenAIEmbeddings
from dotenv import load_dotenv
load_dotenv() # 加载 .env 文件

# ==========================================
# 1. 页面配置 & 初始化（只在页面加载时执行一次）
# ==========================================
st.set_page_config(page_title="企业智能客服 Agent", page_icon="🤖")
st.title("🤖 企业智能客服 Agent")
st.caption("基于 RAG + Function Calling 构建")

API_KEY = os.getenv("ZHIPU_API_KEY")


# 缓存向量数据库，避免每次刷新网页都重新加载文档（极度重要，否则网页会卡死）
@st.cache_resource
def init_vectorstore():
    embeddings = OpenAIEmbeddings(
        model="embedding-2",
        openai_api_key=API_KEY,
        openai_api_base="https://open.bigmodel.cn/api/paas/v4/"
    )
    loader = TextLoader("company_rules.txt", encoding='utf-8')
    documents = loader.load()
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=100, chunk_overlap=10)
    chunks = text_splitter.split_documents(documents)
    vectorstore = Chroma.from_documents(documents=chunks, embedding=embeddings)
    return vectorstore


vectorstore = init_vectorstore()

client = OpenAI(api_key=API_KEY, base_url="https://open.bigmodel.cn/api/paas/v4/")

def trim_messages(messages, max_tokens=2000):
    """
    裁剪历史消息，防止 Token 超限
    """
    encoding = tiktoken.get_encoding("cl100k_base")

    # 计算当前所有消息的总 Token 数
    total_tokens = sum(
        len(encoding.encode(str(m.get("content", "") if isinstance(m, dict) else m.content))) for m in messages)

    # 如果没超过限制，原样返回
    if total_tokens <= max_tokens:
        return messages

    # 如果超限了，强制保留第一条（通常是最重要的 system 提示词）
    trimmed = []
    current_tokens = 0

    # 先把 system 放进去（如果有的话）
    if messages and (isinstance(messages[0], dict) and messages[0].get("role") == "system"):
        trimmed.append(messages[0])
        current_tokens += len(encoding.encode(str(messages[0].get("content", ""))))
        start_index = 1
    else:
        start_index = 0

    # 从最新的消息开始，倒着往回捞
    temp_msgs = []
    for msg in reversed(messages[start_index:]):
        # 兼容字典和对象两种格式
        content_str = str(msg.get("content", "") if isinstance(msg, dict) else msg.content)
        msg_tokens = len(encoding.encode(content_str))

        if current_tokens + msg_tokens <= max_tokens:
            temp_msgs.insert(0, msg)
            current_tokens += msg_tokens
        else:
            break  # 装不下了，停止

    # 把保留下来的历史拼接到 system 后面
    trimmed.extend(temp_msgs)

    print(f"  [系统日志: 上下文过长，已触发裁剪！原 Token 数: {total_tokens}, 裁剪后: {current_tokens}]")
    return trimmed
# ==========================================
# 2. 定义工具和 RAG 检索函数
# ==========================================
def search_company_rules(query: str) -> str:
    """去知识库里检索公司规章制度"""
    docs = vectorstore.similarity_search(query, k=2)
    if not docs:
        return "未找到相关规定。"
    return "\n---\n".join([doc.page_content for doc in docs])


tools = [
    {
        "type": "function",
        "function": {
            "name": "search_company_rules",
            "description": "当用户询问公司内部规章制度（如报销、请假、Wi-Fi密码等）时，使用此工具检索知识库。",
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {"type": "string", "description": "检索关键词，例如：报销邮箱"}
                },
                "required": ["query"]
            }
        }
    }
]
available_functions = {"search_company_rules": search_company_rules}

# ==========================================
# 3. 管理聊天记录（Streamlit 专属逻辑）
# ==========================================
# 初始化 session_state 里的 messages 列表
if "messages" not in st.session_state:
    st.session_state.messages = [
        {"role": "system",
         "content": "你是一个公司智能客服。你需要解答员工关于公司制度的问题。遇到不懂的，请调用工具查询知识库。"}
    ]

# 把历史聊天记录渲染到网页上
for msg in st.session_state.messages:
    # 只展示用户和 AI 的文字对话，不展示内部的 tool 消息
    if not isinstance(msg, dict):
        continue

    if msg["role"] in ["user", "assistant"] and isinstance(msg.get("content"), str):
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])

# ==========================================
# 4. Agent 核心交互逻辑
# ==========================================
if user_input := st.chat_input("请输入您的问题..."):
    # 1. 展示用户输入
    with st.chat_message("user"):
        st.markdown(user_input)
    st.session_state.messages.append({"role": "user", "content": user_input})

    # 2. 启动 Agent 循环
    with st.chat_message("assistant"):
        # 显示一个思考中的加载动画
        with st.spinner("思考中..."):
            step = 0
            max_steps = 5

            while step < max_steps:
                step += 1
                safe_messages = trim_messages(st.session_state.messages, max_tokens=2000)
                response = client.chat.completions.create(
                    model="glm-4-flash",
                    messages=safe_messages,
                    tools=tools,
                    tool_choice="auto"
                )
                msg = response.choices[0].message

                # 如果没有调用工具，直接输出结果
                if not msg.tool_calls:
                    st.markdown(msg.content)
                    st.session_state.messages.append({"role": "assistant", "content": msg.content})
                    break

                # 记录模型决策
                st.session_state.messages.append(msg)

                # 执行工具
                for tool_call in msg.tool_calls:
                    func_name = tool_call.function.name
                    func_args = json.loads(tool_call.function.arguments)

                    try:
                        result = available_functions[func_name](**func_args)
                    except Exception as e:
                        result = f"工具报错: {e}"

                    # 把结果喂回去
                    st.session_state.messages.append({
                        "role": "tool",
                        "tool_call_id": tool_call.id,
                        "name": func_name,
                        "content": str(result)
                    })