import streamlit as st
import io
import time
from openai import OpenAI
from pypdf import PdfReader
import docx
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

st.set_page_config(page_title="电商 AI 智能客服", page_icon="🛍️", layout="wide")
st.title("🛍️ 电商 AI 智能客服 Agent")

# ================= 内置大模型服务商 =================
PROVIDERS = {
    "DeepSeek (深度求索)": "https://api.deepseek.com/v1",
    "阿里通义千问": "https://dashscope.aliyuncs.com/compatible-mode/v1",
    "Kimi (月之暗面)": "https://api.moonshot.cn/v1",
    "智谱 GLM": "https://open.bigmodel.cn/api/paas/v4/",
    "OpenAI (需魔法)": "https://api.openai.com/v1",
    "自定义 / 其他": "custom"
}

# ================= 侧边栏配置 =================
st.sidebar.header("⚙️ 系统配置")
provider = st.sidebar.selectbox("选择大模型服务商", list(PROVIDERS.keys()))

if PROVIDERS[provider] == "custom":
    base_url = st.sidebar.text_input("自定义 API 地址 (Base URL)")
else:
    base_url = PROVIDERS[provider]
    st.sidebar.caption(f"接口地址已自动配置：{base_url}")

api_key = st.sidebar.text_input(f"输入 {provider} 的 API Key", type="password")
if not api_key:
    st.info("👈 请在左侧填入 API Key 开始使用。")
    st.stop()

client = OpenAI(api_key=api_key, base_url=base_url)

# 自动拉取模型列表
model_list = []
try:
    model_list = [m.id for m in client.models.list()]
except Exception:
    st.sidebar.error("获取模型失败，请检查 Key。")

if model_list:
    default_index = model_list.index("deepseek-chat") if "deepseek-chat" in model_list else 0
    model_name = st.sidebar.selectbox("模型名称", model_list, index=default_index)
else:
    model_name = st.sidebar.text_input("手动输入模型名称", value="deepseek-chat")

# ================= 本地知识库构建 =================
if "kb_vectorizer" not in st.session_state:
    # 模拟电商知识库
    kb_texts = [
        "退换货政策：支持7天无理由退货。商品必须保持完好，不影响二次销售。生鲜食品不支持退货。",
        "发货规则：每天下午4点前付款，当天发货。默认发中通快递，偏远地区发邮政。",
        "会员权益：普通会员享受95折，VIP会员享受9折，并赠送运费险。"
    ]
    vectorizer = TfidfVectorizer()
    kb_matrix = vectorizer.fit_transform(kb_texts)
    st.session_state.kb_vectorizer = vectorizer
    st.session_state.kb_matrix = kb_matrix
    st.session_state.kb_texts = kb_texts


# ================= 模拟业务 API =================
def mock_query_order(order_id: str):
    """查询订单状态的模拟函数"""
    if "123" in order_id:
        return "订单 123：已发货，预计明天送达。"
    return f"订单 {order_id}：未找到该订单，请核实。"


def mock_query_logistics(order_id: str):
    """查询物流信息的模拟函数"""
    return f"订单 {order_id} 的物流轨迹：今天上午10点已到达杭州分拨中心，正在派送中。"


# ================= 工具定义 =================
tools = [
    {
        "type": "function",
        "function": {
            "name": "mock_query_order",
            "description": "查询订单状态",
            "parameters": {
                "type": "object",
                "properties": {"order_id": {"type": "string"}},
                "required": ["order_id"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "mock_query_logistics",
            "description": "查询物流轨迹",
            "parameters": {
                "type": "object",
                "properties": {"order_id": {"type": "string"}},
                "required": ["order_id"]
            }
        }
    }
]

# ================= 聊天逻辑 =================
if "messages" not in st.session_state:
    st.session_state.messages = [{"role": "system",
                                  "content": "你是专业电商客服，能回答政策问题，也能调用工具查订单/物流。如果知识库没答案，必须回答'正在为您转接人工客服'。"}]

for msg in st.session_state.messages:
    if msg["role"] != "system" and msg["role"] != "tool":
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])

if prompt := st.chat_input("您好，有什么可以帮您？(例如：查一下订单123 / 怎么退货？)"):
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    with st.chat_message("assistant"):
        # 1. 检索知识库
        query_vec = st.session_state.kb_vectorizer.transform([prompt])
        sims = cosine_similarity(query_vec, st.session_state.kb_matrix).flatten()
        best_idx = sims.argmax()
        best_score = sims[best_idx]
        context = st.session_state.kb_texts[best_idx] if best_score > 0.1 else ""

        # 2. 组装请求
        current_messages = [
                               {"role": "system",
                                "content": f"你是电商客服。请基于知识库回答：\n{context}\n如果是查订单或物流，请调用工具。"}
                           ] + st.session_state.messages

        # 3. 第一次请求
        response = client.chat.completions.create(
            model=model_name,
            messages=current_messages,
            tools=tools
        )
        msg = response.choices[0].message

        # 4. 如果调用了工具
        if msg.tool_calls:
            tool_call = msg.tool_calls[0]
            func_name = tool_call.function.name
            import json

            args = json.loads(tool_call.function.arguments)

            # 执行函数
            if func_name == "mock_query_order":
                result = mock_query_order(args.get("order_id", ""))
            elif func_name == "mock_query_logistics":
                result = mock_query_logistics(args.get("order_id", ""))
            else:
                result = "未知工具"

            st.write(f"🛠️ 调用业务接口: `{func_name}`")

            # === 【核心修复：补回 reasoning_content】 ===
            # 用 getattr 安全获取，防止没有思考模式的模型报错
            reasoning_content = getattr(msg, "reasoning_content", None)

            safe_msg = {
                "role": "assistant",
                "content": msg.content or "",
                # 如果有思考过程，就原样塞回去，确保 API 校验通过
                "reasoning_content": reasoning_content,
                "tool_calls": [
                    {
                        "id": tool_call.id,
                        "type": "function",
                        "function": {
                            "name": tool_call.function.name,
                            "arguments": tool_call.function.arguments
                        }
                    }
                ]
            }

            # 顺便过滤掉 None 值，防止某些模型接口不认
            if safe_msg["reasoning_content"] is None:
                del safe_msg["reasoning_content"]

            current_messages.append(safe_msg)
            current_messages.append({
                "role": "tool",
                "tool_call_id": tool_call.id,
                "content": result
            })

            # 再次请求大模型
            final_resp = client.chat.completions.create(
                model=model_name,
                messages=current_messages
            )
            final_text = final_resp.choices[0].message.content

            # 存入长记忆（同样使用 safe_msg）
            st.session_state.messages.append(safe_msg)
            st.session_state.messages.append({
                "role": "tool",
                "tool_call_id": tool_call.id,
                "content": result
            })

            current_messages.append(safe_msg)
            current_messages.append({
                "role": "tool",
                "tool_call_id": tool_call.id,
                "content": result
            })

            # 再次请求
            final_resp = client.chat.completions.create(
                model=model_name,
                messages=current_messages
            )
            final_text = final_resp.choices[0].message.content

            # 存入长记忆
            st.session_state.messages.append(safe_msg)
            st.session_state.messages.append({
                "role": "tool",
                "tool_call_id": tool_call.id,
                "content": result
            })
        else:
            final_text = msg.content

        # 5. 最终展示
        st.markdown(final_text)
        st.session_state.messages.append({"role": "assistant", "content": final_text})
