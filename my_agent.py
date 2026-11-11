import streamlit as st
import io
import time
import logging
import concurrent.futures
from openai import OpenAI
from pypdf import PdfReader
import docx
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# ================= 0. 企业级日志配置 =================
logging.basicConfig(
    filename='agent.log',
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)

# ================= 1. 页面配置 =================
st.set_page_config(page_title="企业级智能招聘中枢", page_icon="🧠", layout="wide")
st.title("🧠 企业级智能招聘中枢 (Pro 版)")
st.caption("支持并发解析 · 日志追踪 · 多轮对话 RAG · 高可用重试")

PROVIDERS = {
    "DeepSeek": "https://api.deepseek.com/v1",
    "阿里通义千问": "https://dashscope.aliyuncs.com/compatible-mode/v1",
    "Kimi": "https://api.moonshot.cn/v1",
    "自定义": "custom"
}

# ================= 2. 侧边栏配置 =================
st.sidebar.header("⚙️ 系统配置")
provider = st.sidebar.selectbox("选择大模型服务商", list(PROVIDERS.keys()))
base_url = PROVIDERS[provider] if PROVIDERS[provider] != "custom" else st.sidebar.text_input("自定义 API 地址")

api_key = st.sidebar.text_input(f"输入 {provider} 的 API Key", type="password")
if not api_key:
    st.info("👈 请在左侧填入 API Key 开始使用。")
    st.stop()

client = OpenAI(api_key=api_key, base_url=base_url)
model_list = []
try:
    model_list = [m.id for m in client.models.list()]
except Exception:
    pass
model_name = st.sidebar.selectbox("模型名称", model_list) if model_list else "deepseek-chat"


# ================= 3. 高可用 API 调用函数 =================
def robust_chat_completion(messages, max_retries=3):
    """带有指数退避重试的 API 调用封装"""
    for attempt in range(max_retries):
        try:
            response = client.chat.completions.create(
                model=model_name,
                messages=messages
            )
            return response.choices[0].message.content
        except Exception as e:
            logging.error(f"API 请求失败 (尝试 {attempt + 1}/{max_retries}): {str(e)}")
            if attempt == max_retries - 1:
                return f"❌ 系统繁忙，大模型 API 调用失败，请稍后重试。错误信息：{str(e)}"
            time.sleep(2 ** attempt)  # 指数退避：等 1秒, 2秒, 4秒


# ================= 4. 并发文件解析 =================
def parse_single_file(file):
    """解析单个文件的辅助函数，供多线程调用"""
    text = ""
    try:
        file.seek(0)
        if file.name.endswith(".txt"):
            text = file.read().decode("utf-8")
        elif file.name.endswith(".pdf"):
            pdf = PdfReader(file)
            text = "".join([p.extract_text() or "" for p in pdf.pages])
        elif file.name.endswith(".docx"):
            doc = docx.Document(io.BytesIO(file.read()))
            text = "".join([para.text for para in doc.paragraphs])
    except Exception as e:
        logging.error(f"文件解析失败 {file.name}: {str(e)}")
        return file.name, ""
    return file.name, text


# ================= 5. 知识库管理 =================
st.sidebar.divider()
st.sidebar.header("📂 简历知识库")
uploaded_files = st.sidebar.file_uploader("上传简历", type=["txt", "pdf", "docx"], accept_multiple_files=True)

if "corpus" not in st.session_state:
    st.session_state.corpus = []
    st.session_state.filenames = []

if st.sidebar.button("📥 并发解析并构建索引"):
    if not uploaded_files:
        st.sidebar.warning("请先上传文件")
    else:
        with st.spinner("多线程并发解析中..."):
            st.session_state.corpus = []
            st.session_state.filenames = []

            # 引入多线程
            with concurrent.futures.ThreadPoolExecutor(max_workers=5) as executor:
                results = executor.map(parse_single_file, uploaded_files)

                for filename, text in results:
                    if text.strip():
                        chunks = [text[i:i + 300] for i in range(0, len(text), 250)]
                        st.session_state.corpus.extend(chunks)
                        st.session_state.filenames.extend([filename] * len(chunks))

            if st.session_state.corpus:
                vectorizer = TfidfVectorizer()
                st.session_state.tfidf_matrix = vectorizer.fit_transform(st.session_state.corpus)
                st.session_state.vectorizer = vectorizer
                logging.info(f"成功构建索引，包含 {len(st.session_state.corpus)} 个块")
                st.sidebar.success(f"✅ 并发解析完成！共 {len(st.session_state.corpus)} 个文本块。")

# ================= 6. 多轮对话区域 =================
if "messages" not in st.session_state:
    st.session_state.messages = []

for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

if prompt := st.chat_input("输入岗位要求，或针对刚才的候选人追问..."):
    if not st.session_state.get("corpus"):
        st.warning("知识库为空，请先构建索引！")
        st.stop()

    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    with st.spinner("向量检索中..."):
        query_vec = st.session_state.vectorizer.transform([prompt])
        similarities = cosine_similarity(query_vec, st.session_state.tfidf_matrix).flatten()
        top_indices = similarities.argsort()[-10:][::-1]

        retrieved_text = ""
        for idx in top_indices:
            if similarities[idx] > 0.01:
                retrieved_text += st.session_state.corpus[idx] + "\n---\n"

    context_info = f"相关简历片段：\n{retrieved_text}" if retrieved_text else "未检索到片段。"
    system_prompt = f"你是HR助手。请基于以下检索结果和聊天历史回答。\n{context_info}"

    api_messages = [{"role": "system", "content": system_prompt}]
    for m in st.session_state.messages[-6:]:
        api_messages.append(m)

    with st.chat_message("assistant"):
        with st.spinner("思考中..."):
            # 使用高可用重试函数
            final_text = robust_chat_completion(api_messages)

        st.markdown(final_text)
        st.session_state.messages.append({"role": "assistant", "content": final_text})