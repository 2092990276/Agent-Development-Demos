import json
import tiktoken
import os
import pandas as pd
from openai import OpenAI
from dotenv import load_dotenv
load_dotenv() # 加载 .env 文件
# 1. 初始化 API
client = OpenAI(api_key=os.getenv("ZHIPU_API_KEY"), base_url="https://open.bigmodel.cn/api/paas/v4/")


# 2. 定义真实的工具函数
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
def read_csv_data(file_path: str) -> str:
    """读取本地 CSV 数据"""
    try:
        df = pd.read_csv(file_path)
        # 取前 7 行（上周）和后 7 行（本周）
        last_week = df.iloc[:7]
        this_week = df.iloc[7:]

        last_week_total = last_week['revenue'].sum()
        this_week_total = this_week['revenue'].sum()

        return (f"数据读取成功。上周总销售额为 {last_week_total}。"
                f"本周总销售额为 {this_week_total}。")
    except Exception as e:
        return f"读取失败: {e}"


def calculate_growth_rate(current: float, previous: float) -> str:
    """计算增长率"""
    rate = ((current - previous) / previous) * 100
    return f"增长率为: {rate:.2f}%"


def send_email(to: str, subject: str, content: str) -> str:
    """发送邮件（这里用打印模拟）"""
    print(f"\n[系统日志: 正在发送邮件给 {to}...]")
    print(f"主题: {subject}\n内容: {content}")
    return "邮件发送成功"


# 3. 组装工具清单
tools = [
    # read_csv_data 需要一个 file_path 参数
    # calculate_growth_rate 需要 current 和 previous 参数
    # send_email 需要 to, subject, content 参数
    {
        "type":"function",
        "function":{
            "name":"read_csv_data",
            "description":"读取本地 CSV 数据并返回前 5 行预览",
            "parameters":{
                "type":"object",
                "properties":{
                    "file_path":{"type":"string","description":"文件路径"},
                },
                "required":["file_path"]
            }
        }
    },
    {
        "type":"function",
        "function":{
            "name":"calculate_growth_rate",
            "description":"计算增长率",
            "parameters":{
                "type":"object",
                "properties":{
                    "current":{"type":"number"},
                    "previous":{"type":"number"},
                },
                "required":["current","previous"]
            }
        }
    },
    {
        "type":"function",
    "function":{
        "name":"send_email",
        "description":"发送邮件",
        "parameters":{
            "type":"object",
            "properties":{
                "to":{"type":"string"},
                "subject":{"type":"string"},
                "content":{"type":"string"},
            },
            "required":["to","subject","content"]
        }
    }
    }
]

available_functions = {
    "read_csv_data": read_csv_data,
    "calculate_growth_rate": calculate_growth_rate,
    "send_email": send_email
}


# 4. 核心 Agent 循环（带历史裁剪 + 多步工具调用）
def run_agent(user_query: str):
    print(f"\n--- 用户输入: {user_query} ---")
    messages = [{"role": "user", "content": user_query}]

    max_steps = 8  # 因为是复杂任务，多给点步数
    step = 0

    while step < max_steps:
        step += 1

        # 【关键】这里接入你刚写的 trim_messages 裁剪上下文逻辑
        safe_messages = trim_messages(messages, max_tokens=2000)

        response = client.chat.completions.create(
            model="glm-4-flash",
            messages=safe_messages,
            tools=tools,
            tool_choice="auto"
        )
        msg = response.choices[0].message

        if not msg.tool_calls:
            print(f"\n最终报告:\n{msg.content}")
            break

        messages.append(msg)

        for tool_call in msg.tool_calls:
            func_name = tool_call.function.name
            func_args = json.loads(tool_call.function.arguments)
            print(f"  -> 调用工具: {func_name}, 参数: {func_args}")

            try:
                result = available_functions[func_name](**func_args)
            except Exception as e:
                result = f"工具执行报错: {e}"

            print(f"  <- 工具返回: {result}")

            messages.append({
                "role": "tool",
                "tool_call_id": tool_call.id,
                "name": func_name,
                "content": str(result)
            })


# 5. 测试
if __name__ == "__main__":
    run_agent(
        "帮我读取 data.csv 看看上周和本周的销售额，计算增长率。如果超过 10%，你【必须调用 send_email 工具】把具体数据和增长率发邮件给老板 boss@company.com。不要直接说发送了，必须使用工具！")