import streamlit as st
from google.cloud import bigquery
import vertexai
from vertexai.language_models import TextEmbeddingModel
from vertexai.generative_models import GenerativeModel
import time
import pandas as pd
from datetime import date
import numpy as np
import plotly.express as px
import matplotlib.pyplot as plt
from wordcloud import WordCloud
from sklearn.decomposition import PCA
from sklearn.cluster import MiniBatchKMeans
import json
import os
from dotenv import load_dotenv

# ================= 1. Global Config & Styles =================
# 1. Load the .env file into the environment
# This looks for a .env file in the current directory
load_dotenv()
project_id = os.getenv("GCP_PROJECT_ID")
dataset_id = os.getenv("DATASET_ID")
location = os.getenv("LOCATION")

project_id = os.getenv("GCP_PROJECT_ID")
dataset_id = os.getenv("DATASET_ID")
location = os.getenv("LOCATION")

st.set_page_config(
    page_title="Olist AI Smart Assistant",
    page_icon="🛒",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.markdown("""
<style>
    .block-container {padding-top: 2rem; padding-bottom: 2rem;}
    .stChatMessage {border-radius: 10px; border: 1px solid #e0e0e0;}
    .stCode {font-family: 'Fira Code', monospace;}
    /* 修复 Streamlit 原生加载时的闪烁感 */
    .stAppViewBlockContainer {transition: opacity 0.3s ease-in-out;}
</style>
""", unsafe_allow_html=True)

# --- ⚠️ 项目配置 ---
PROJECT_ID = project_id
DATASET_ID = dataset_id
LOCATION = location

SQL_TABLE = f"{PROJECT_ID}.{DATASET_ID}.init_search_unioned"
VECTOR_TABLE = f"{PROJECT_ID}.{DATASET_ID}.dim_embedded_vectors"
EMBEDDING_MODEL_NAME = "text-embedding-005"

# 💲 Pricing Configuration
PRICING_RATES = {
    "gemini-2.0-flash-001": {"input": 0.10, "output": 0.40}, 
    "gemini-1.5-flash-001": {"input": 0.075, "output": 0.30},
    "default": {"input": 0.10, "output": 0.40}
}

def safe_error(e):
    return str(e).replace(PROJECT_ID, "********")

def calculate_cost(model_name, input_tok, output_tok):
    rates = PRICING_RATES.get(model_name, PRICING_RATES["default"])
    return ((input_tok / 1e6) * rates["input"]) + ((output_tok / 1e6) * rates["output"])

# ================= 2. Resource Initialization =================
@st.cache_resource(show_spinner="Initializing AI Core...")
def init_resources():
    try:
        bq_client = bigquery.Client(project=PROJECT_ID)
        vertexai.init(project=PROJECT_ID, location=LOCATION)
        embed_model = TextEmbeddingModel.from_pretrained(EMBEDDING_MODEL_NAME)
        
        try:
            model_name = "gemini-2.0-flash-001"
            gen_model = GenerativeModel(model_name)
            status_msg = "🟢 Online (Gemini 2.0 Flash)"
        except:
            model_name = "gemini-1.5-flash-001"
            gen_model = GenerativeModel(model_name)
            status_msg = "🟡 Online (Gemini 1.5 Flash)"
            
        return bq_client, embed_model, gen_model, status_msg, model_name
    except Exception as e:
        return None, None, None, f"🔴 Error: {safe_error(e)}", "default"

client, embedding_model, generative_model, model_status, current_model_name = init_resources()

if not client:
    st.error(f"System initialization failed: {model_status}")
    st.stop()

# ================= 3. Core Logic: SQL Generation =================
def generate_and_run_sql(user_query):
    full_table_ref = f"`{SQL_TABLE}`"
    
    schema_prompt = f"""
    You are a BigQuery SQL Expert. Write a SQL query to answer the user's question.
    
    TARGET TABLE: {full_table_ref}
    COLUMNS: order_id, customer_city, order_status, price, freight_value, payment_type, product_category_name, product_weight_g, review_score, order_purchase_timestamp
    
    RULES:
    1. Return ONLY raw SQL. No Markdown.
    2. Limit results to 100.
    
    USER QUESTION: "{user_query}"
    """
    
    try:
        response = generative_model.generate_content(schema_prompt)
        generated_sql = response.text.replace("```sql", "").replace("```", "").strip()
        
        usage = {"input": 0, "output": 0}
        if response.usage_metadata:
            usage["input"] = response.usage_metadata.prompt_token_count
            usage["output"] = response.usage_metadata.candidates_token_count

        df = client.query(generated_sql).to_dataframe()
        return generated_sql, df, usage
    except Exception as e:
        return str(e), None, {"input": 0, "output": 0}

# ================= 4. Core Logic: Vector Search =================
@st.cache_data(ttl=3600, show_spinner=False)
def get_query_vector(text):
    for _ in range(3):
        try:
            embeddings = embedding_model.get_embeddings([text])
            return embeddings[0].values
        except:
            time.sleep(1)
    return None

def search_vectors_hybrid(query_vector, user_text, filters, top_k=20):
    query_params = [
        bigquery.ArrayQueryParameter("query_vector", "FLOAT64", query_vector),
        bigquery.ScalarQueryParameter("keyword", "STRING", f"%{user_text}%")
    ]
    
    where_clauses = []
    if filters.get('min_score'):
        where_clauses.append(f"metadata.review_score >= {filters['min_score']}")
        
    where_stmt = "WHERE " + " AND ".join(where_clauses) if where_clauses else ""

    sql = f"""
        SELECT
            knowledge_id, page_content, metadata.review_score, metadata.product_category_name, metadata.price,
            metadata.customer_city,
            (ML.DISTANCE(@query_vector, ml_generate_embedding_result, 'COSINE') - 
             (CASE WHEN LOWER(page_content) LIKE LOWER(@keyword) THEN 0.3 ELSE 0.0 END)) 
            AS hybrid_score
        FROM `{VECTOR_TABLE}`
        {where_stmt}
        ORDER BY hybrid_score ASC
        LIMIT {top_k}
    """
    job_config = bigquery.QueryJobConfig(query_parameters=query_params)
    return client.query(sql, job_config=job_config).to_dataframe()

def ask_gemini_stream(user_query, context_text, chat_history):
    history_block = "\n".join([f"{msg['role']}: {msg['content'][:200]}" for msg in chat_history[-3:]])
    prompt = f"""
    SYSTEM: You are a Senior Data Analyst for Olist. The context data is in Portuguese.
    INSTRUCTION: 
    1. Answer the user's question based strictly on Context.
    2. ALWAYS translate the insights and evidence into ENGLISH.
    3. If quoting a Portuguese review, provide the English translation in parentheses.
    
    CONTEXT: {context_text}
    HISTORY: {history_block}
    QUESTION: {user_query}
    """
    return generative_model.generate_content(prompt, stream=True)

def decide_route(user_query):
    try:
        resp = generative_model.generate_content(f"Classify query as STATS (aggregation/math) or SEARCH (text/reviews). Query: {user_query}")
        return resp.text.strip().upper()
    except:
        return "SEARCH"

# ================= 5. ⚡ Optimized Semantic Galaxy Logic =================

@st.cache_data(ttl=1800, show_spinner=False)
def fetch_data_and_pca(user_query=None, limit_global=3000, limit_focused=800):
    """Level 1 Cache: I/O & PCA"""
    if not user_query:
        query = f"""
        SELECT 
            metadata.review_score as review_score,
            metadata.product_category_name as category,
            metadata.review_comment_message as comment,
            metadata.customer_city as city,
            metadata.price as price,
            ml_generate_embedding_result
        FROM `{VECTOR_TABLE}`
        WHERE ml_generate_embedding_result IS NOT NULL
        AND metadata.review_comment_message IS NOT NULL
        LIMIT {limit_global}
        """
        query_params = []
    else:
        q_vec = get_query_vector(user_query)
        if q_vec is None: return pd.DataFrame()
        query = f"""
        SELECT 
            metadata.review_score as review_score,
            metadata.product_category_name as category,
            metadata.review_comment_message as comment,
            metadata.customer_city as city,
            metadata.price as price,
            ml_generate_embedding_result
        FROM `{VECTOR_TABLE}`
        WHERE ml_generate_embedding_result IS NOT NULL
        AND metadata.review_comment_message IS NOT NULL
        ORDER BY ML.DISTANCE(@q_vec, ml_generate_embedding_result, 'COSINE') ASC
        LIMIT {limit_focused}
        """
        query_params = [bigquery.ArrayQueryParameter("q_vec", "FLOAT64", q_vec)]
    
    job_config = bigquery.QueryJobConfig(query_parameters=query_params)
    df = client.query(query, job_config=job_config).to_dataframe()
    
    if not df.empty:
        matrix = np.stack(df['ml_generate_embedding_result'].values)
        pca = PCA(n_components=3)
        components = pca.fit_transform(matrix)
        df['x'] = components[:, 0]
        df['y'] = components[:, 1]
        df['z'] = components[:, 2]
        df = df.drop(columns=['ml_generate_embedding_result'])
        df['category'] = df['category'].fillna('Unknown')
    return df

@st.cache_data(ttl=3600, show_spinner=False)
def get_labeled_clusters(df_coords, n_clusters):
    """Level 2 Cache: Clustering & Labeling"""
    if df_coords.empty: return df_coords
    df = df_coords.copy()
    
    kmeans = MiniBatchKMeans(n_clusters=n_clusters, random_state=42, batch_size=256, n_init=3)
    features = df[['x', 'y', 'z']].values
    df['cluster_id'] = kmeans.fit_predict(features)
    
    samples = {}
    for cid in range(n_clusters):
        sample_reviews = df[df['cluster_id'] == cid]['comment'].sample(min(3, len(df))).tolist()
        samples[cid] = sample_reviews
    
    label_prompt = f"""
    Generate a short topic title (2-4 words) in ENGLISH for EACH cluster ID based on the samples.
    Data: {json.dumps(samples, ensure_ascii=False)}
    Output strictly valid JSON format: {{ "0": "Topic Name", "1": "Topic Name" }}
    """
    try:
        resp = generative_model.generate_content(label_prompt)
        text_resp = resp.text.replace("```json", "").replace("```", "").strip()
        labels_map = json.loads(text_resp)
        int_map = {int(k): v for k, v in labels_map.items()}
        df['cluster_label'] = df['cluster_id'].map(int_map).fillna(df['cluster_id'].astype(str))
    except Exception:
        df['cluster_label'] = "Cluster " + df['cluster_id'].astype(str)
    return df

def plot_wordcloud(text):
    if not text or len(text) < 5: return None
    wc = WordCloud(background_color="white", colormap="Reds", width=800, height=400, max_words=80).generate(text)
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.imshow(wc, interpolation='bilinear'); ax.axis("off")
    return fig

# ================= 6. UI Main Interface =================

if "messages" not in st.session_state: st.session_state.messages = []
if "total_cost" not in st.session_state: st.session_state.total_cost = 0.0
if "total_tokens" not in st.session_state: st.session_state.total_tokens = 0

# --- Sidebar (Using Placeholders to avoid Rerun) ---
with st.sidebar:
    st.header("⚙️ Control Panel")
    with st.expander("🔎 Chat Filters", expanded=True):
        score_range = st.slider("Min Review Score", 1, 5, 1)
    
    if st.button("🗑️ Reset Session", use_container_width=True):
        st.session_state.messages = []
        st.session_state.total_cost = 0.0
        st.session_state.total_tokens = 0
        st.rerun()
        
    st.divider()
    token_placeholder = st.empty()
    cost_placeholder = st.empty()
    
    token_placeholder.metric("Tokens", f"{st.session_state.total_tokens:,}")
    cost_placeholder.metric("Cost", f"${st.session_state.total_cost:.4f}")

# --- Header ---
st.title("🛒 Olist Smart Data Assistant")
col_h1, col_h2, col_h3 = st.columns([2, 1, 1])
with col_h1: st.markdown(f"**Model:** `{current_model_name}`")
with col_h2: st.markdown(f"**Status:** {model_status}")
with col_h3: st.markdown(f"**Region:** `{LOCATION}`")
st.divider()

# ================= 7. Main Tabs =================
tab1, tab2 = st.tabs(["💬 RAG Chat Assistant", "🌌 Dynamic Semantic Galaxy"])

# --- TAB 1: Chat Interface ---
with tab1:
    # 🔴 FIX START: 修复历史记录渲染
    # Render History
    for m in st.session_state.messages:
        with st.chat_message(m["role"]):
            st.markdown(m["content"])
            # 1. 显示 SQL 代码 (如果存在)
            if "sql_query" in m:
                with st.expander("🛠️ Generated SQL", expanded=False):
                    st.code(m["sql_query"], language="sql")
            # 2. 显示 Data Table (SQL 结果)
            if "data_table" in m: 
                with st.expander("📊 Data Result", expanded=False):
                    st.dataframe(m["data_table"])
            # 3. 显示 Source Preview (向量搜索结果)
            if "source_preview" in m: 
                with st.expander("📚 Source Documents", expanded=False):
                    st.dataframe(m["source_preview"])
    # 🔴 FIX END

    # Chat Input
    if prompt := st.chat_input("Ask a question (e.g., Why do people complain about delivery?)"):
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"): st.markdown(prompt)
        
        with st.chat_message("assistant"):
            cont = st.empty()
            
            with st.status("🧠 Thinking...", expanded=True) as status:
                
                status.write("🤔 Analyzing intent...")
                route = decide_route(prompt)
                
                current_usage = {"input": 0, "output": 0}
                
                if "STATS" in route:
                    status.write("📊 Generating SQL...")
                    sql, df, usage = generate_and_run_sql(prompt)
                    
                    if df is not None:
                        # 🔴 FIX: 立即显示生成的 SQL
                        status.markdown("**Executed SQL:**")
                        status.code(sql, language='sql')
                        
                        status.write("📝 Formatting answer...")
                        ans = generative_model.generate_content(f"Q:{prompt}\nData:{df.head().to_csv()}\nINSTRUCTION: Answer in ENGLISH.").text
                        
                        current_usage["input"] += usage["input"]
                        current_usage["output"] += usage["output"]
                        
                        cont.markdown(ans)
                        
                        # 🔴 FIX: 立即显示 DataFrame
                        with st.expander("📊 Data Result", expanded=True):
                            st.dataframe(df)

                        # 🔴 FIX: 将 SQL 和 DataFrame 都保存到历史记录
                        st.session_state.messages.append({
                            "role": "assistant", 
                            "content": ans, 
                            "sql_query": sql,  # Save SQL
                            "data_table": df   # Save DF
                        })
                        status.update(label="✅ SQL Stats Complete", state="complete", expanded=False)
                    else:
                        status.update(label="❌ SQL Error", state="error")
                        cont.error("SQL Execution Failed")
                        
                else: # SEARCH Mode
                    status.write("🔎 Vector Search...")
                    q_vec = get_query_vector(prompt)
                    df = search_vectors_hybrid(q_vec, prompt, {"min_score": score_range})
                    
                    if df is not None:
                        status.write("🤖 Generating insights...")
                        
                        # 🔴 FIX: 立即显示引用来源 (Source Preview)
                        status.markdown("**Found References:**")
                        status.dataframe(df[['review_score', 'product_category_name', 'page_content']].head(3))

                        ctx = "\n".join(df['page_content'].tolist())
                        resp = ""
                        
                        for chunk in ask_gemini_stream(prompt, ctx, st.session_state.messages):
                            if chunk.text: resp += chunk.text; cont.markdown(resp + "▌")
                        
                        cont.markdown(resp) 
                        
                        # 🔴 FIX: 立即显示完整 Source Preview (放在回答下方)
                        with st.expander("📚 Source Documents", expanded=False):
                            st.dataframe(df.head())

                        # 🔴 FIX: 保存 Source Preview 到历史记录
                        st.session_state.messages.append({
                            "role": "assistant", 
                            "content": resp, 
                            "source_preview": df.head() # Save Source
                        })
                        status.update(label="✅ Analysis Complete", state="complete", expanded=False)
                    else:
                        status.update(label="⚠️ No Data", state="error")
                        cont.warning("No data found.")

            turn_cost = calculate_cost(current_model_name, current_usage["input"], current_usage["output"])
            st.session_state.total_cost += turn_cost
            st.session_state.total_tokens += (current_usage["input"] + current_usage["output"])
            
            token_placeholder.metric("Tokens", f"{st.session_state.total_tokens:,}")
            cost_placeholder.metric("Cost", f"${st.session_state.total_cost:.4f}")

# --- TAB 2: Dynamic Semantic Galaxy ---
with tab2:
    st.header("🌌 Semantic Galaxy & VoC Analytics")
    st.caption("AI-powered visualization. Filter the map by typing a context below.")
    
    col_input, col_ctrl = st.columns([2, 1])
    with col_input:
        map_query = st.text_input("🔍 Filter Map by Context", placeholder="Leave empty for Global View...")
    with col_ctrl:
        n_clusters = st.slider("Topics", 2, 8, 5)
        view_mode = st.radio("View", ["2D", "3D"], horizontal=True)
        color_by = st.selectbox("Color", ["cluster_label", "review_score", "category"])

    title_suffix = f"for '{map_query}'" if map_query else "(Global View)"
    
    with st.spinner("Mapping semantic space..."):
        raw_df_coords = fetch_data_and_pca(user_query=map_query, limit_global=3000, limit_focused=800)

    if not raw_df_coords.empty:
        with st.spinner("AI is organizing topics..."):
            final_df = get_labeled_clusters(raw_df_coords, n_clusters)

        if view_mode == "3D":
            fig = px.scatter_3d(final_df, x='x', y='y', z='z', color=color_by,
                                hover_data=['comment', 'city', 'price'], height=600,
                                color_continuous_scale="RdYlGn" if color_by == "review_score" else None,
                                title=f"3D Semantic Space {title_suffix}")
        else:
            fig = px.scatter(final_df, x='x', y='y', color=color_by,
                             hover_data=['comment', 'city', 'price'], height=500,
                             color_continuous_scale="RdYlGn" if color_by == "review_score" else None,
                             title=f"2D Semantic Map {title_suffix}")
            fig.update_xaxes(visible=False); fig.update_yaxes(visible=False)
        
        st.plotly_chart(fig, use_container_width=True)
        
        st.divider()
        c1, c2 = st.columns(2)
        with c1:
            clusters = sorted(final_df['cluster_label'].unique())
            sel_cluster = st.selectbox("Drill-down Topic", clusters)
            sub_df = final_df[final_df['cluster_label'] == sel_cluster]
            st.dataframe(sub_df[['review_score', 'comment']].head(5), use_container_width=True)
        with c2:
            st.caption(f"Word Cloud: {sel_cluster}")
            text = " ".join(sub_df['comment'].dropna().astype(str).tolist())
            fig_wc = plot_wordcloud(text)
            if fig_wc: st.pyplot(fig_wc)
    else:
        st.warning("No data found.")