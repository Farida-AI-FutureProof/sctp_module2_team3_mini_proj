import streamlit as st
from google.cloud import bigquery
import vertexai
from vertexai.language_models import TextEmbeddingModel
from vertexai.generative_models import GenerativeModel
import time
import pandas as pd
from datetime import date

# ================= 1. Global Config & Styles =================
st.set_page_config(
    page_title="Olist AI Smart Assistant",
    page_icon="🛒",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for visual optimization
st.markdown("""
<style>
    .block-container {padding-top: 2rem; padding-bottom: 2rem;}
    .stChatMessage {border-radius: 10px; border: 1px solid #e0e0e0;}
    .stCode {font-family: 'Fira Code', monospace;}
</style>
""", unsafe_allow_html=True)

PROJECT_ID = "my-project-sctp-module-2" 
DATASET_ID = "olist_dbt_dataset"
LOCATION = "us-central1"

SQL_TABLE = f"{PROJECT_ID}.{DATASET_ID}.init_search_unioned"
VECTOR_TABLE = f"{PROJECT_ID}.{DATASET_ID}.dim_embedded_vectors"
EMBEDDING_MODEL_NAME = "text-embedding-005"

# 💲 Pricing Configuration (USD per 1M Tokens)
PRICING_RATES = {
    "gemini-2.0-flash-001": {"input": 0.10, "output": 0.40}, 
    "gemini-1.5-flash-001": {"input": 0.075, "output": 0.30},
    "default": {"input": 0.10, "output": 0.40}
}

def safe_error(e):
    return str(e).replace(PROJECT_ID, "********")

def calculate_cost(model_name, input_tok, output_tok):
    rates = PRICING_RATES.get(model_name, PRICING_RATES["default"])
    cost_input = (input_tok / 1_000_000) * rates["input"]
    cost_output = (output_tok / 1_000_000) * rates["output"]
    return cost_input + cost_output

# ================= 2. Resource Initialization =================
@st.cache_resource
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
            status_msg = "🟡 Online (Gemini 1.5 Flash - Degraded Mode)"
            
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
    where_clauses = []
    query_params = [
        bigquery.ArrayQueryParameter("query_vector", "FLOAT64", query_vector),
        bigquery.ScalarQueryParameter("top_k", "INT64", top_k),
        bigquery.ScalarQueryParameter("keyword", "STRING", f"%{user_text}%")
    ]

    if filters.get('min_score'):
        where_clauses.append("metadata.review_score >= @min_score")
        query_params.append(bigquery.ScalarQueryParameter("min_score", "INT64", filters['min_score']))
    if filters.get('start_date'):
        where_clauses.append("CAST(metadata.order_purchase_timestamp AS DATE) >= @start_date")
        query_params.append(bigquery.ScalarQueryParameter("start_date", "DATE", filters['start_date']))
    if filters.get('end_date'):
        where_clauses.append("CAST(metadata.order_purchase_timestamp AS DATE) <= @end_date")
        query_params.append(bigquery.ScalarQueryParameter("end_date", "DATE", filters['end_date']))

    where_stmt = "WHERE " + " AND ".join(where_clauses) if where_clauses else ""

    sql = f"""
        SELECT
            knowledge_id, page_content, metadata.order_id, metadata.customer_city,
            metadata.review_score, metadata.product_category_name, metadata.price,
            ML.DISTANCE(@query_vector, ml_generate_embedding_result, 'COSINE') AS vec_dist,
            (ML.DISTANCE(@query_vector, ml_generate_embedding_result, 'COSINE') - 
             (CASE WHEN LOWER(page_content) LIKE LOWER(@keyword) THEN 0.3 ELSE 0.0 END)) 
            AS hybrid_score
        FROM `{VECTOR_TABLE}`
        {where_stmt}
        ORDER BY hybrid_score ASC
        LIMIT @top_k
    """
    job_config = bigquery.QueryJobConfig(query_parameters=query_params)
    return client.query(sql, job_config=job_config).to_dataframe()

def ask_gemini_stream(user_query, context_text, chat_history):
    history_block = "\n".join([f"{msg['role']}: {msg['content'][:200]}" for msg in chat_history[-3:]])
    prompt = f"""
    SYSTEM: You are a Senior Data Analyst for Olist.
    CONTEXT: {context_text}
    HISTORY: {history_block}
    QUESTION: {user_query}
    
    INSTRUCTIONS: Answer based strictly on Context. Render lists as Markdown tables.
    """
    return generative_model.generate_content(prompt, stream=True)

def decide_route(user_query):
    try:
        resp = generative_model.generate_content(f"Classify query as STATS (aggregation/math) or SEARCH (text/reviews). Query: {user_query}")
        return resp.text.strip().upper()
    except:
        return "SEARCH"

# ================= 5. UI Main Interface =================

# --- Initialize Session State ---
if "messages" not in st.session_state:
    st.session_state.messages = []
if "total_cost" not in st.session_state:
    st.session_state.total_cost = 0.0
if "total_tokens" not in st.session_state:
    st.session_state.total_tokens = 0

# --- Sidebar ---
with st.sidebar:
    st.header("⚙️ Control Panel")
    
    with st.expander("🔎 Data Filters", expanded=True):
        score_range = st.slider("Score Range (Review Score)", 1, 5, (1, 5))
        start_date = st.date_input("Start Date", value=date(2017, 1, 1))
        end_date = st.date_input("End Date", value=date(2018, 8, 31))
    
    st.divider()
    
    st.subheader("📊 Session Statistics")
    col_metric1, col_metric2 = st.columns(2)
    col_metric1.metric("Total Tokens", f"{st.session_state.total_tokens:,}")
    col_metric2.metric("Total Cost", f"${st.session_state.total_cost:.4f}")
    
    if st.button("🗑️ Clear Conversation", use_container_width=True, type="primary"):
        st.session_state.messages = []
        st.session_state.total_cost = 0.0
        st.session_state.total_tokens = 0
        st.rerun()

# --- Header & Status ---
st.title("🛒 Olist Smart Data Assistant")

# Use column layout to optimize header info
col_h1, col_h2, col_h3 = st.columns([2, 1, 1])
with col_h1:
    st.markdown(f"**Model:** `{current_model_name}`")
with col_h2:
    st.markdown(f"**Status:** {model_status}")
with col_h3:
    st.markdown(f"**Region:** `{LOCATION}`")

st.divider()

# --- Render Chat History ---
for msg in st.session_state.messages:
    avatar = "🧑‍💻" if msg["role"] == "user" else "🤖"
    with st.chat_message(msg["role"], avatar=avatar):
        st.markdown(msg["content"])
        
        # Render additional info (SQL, Data, Cost)
        if "sql_code" in msg:
            with st.expander("🛠️ View Generated SQL"):
                st.code(msg["sql_code"], language="sql")
        if "data_table" in msg:
            with st.expander("📊 View Source Data"):
                st.dataframe(msg["data_table"], use_container_width=True)
        if "source_preview" in msg:
            with st.expander(f"📚 Reference Documents ({len(msg['source_preview'])} items)"):
                st.dataframe(msg["source_preview"], use_container_width=True)
        if "usage_stats" in msg:
            stats = msg["usage_stats"]
            st.caption(f"⚡ Turn Cost: ${stats['cost']:.5f} ({stats['input']} in / {stats['output']} out)")

# --- Chat Input Processing ---
if prompt := st.chat_input("Enter your question (e.g., What was the average freight value in Rio de Janeiro in 2017?)"):
    
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user", avatar="🧑‍💻"):
        st.markdown(prompt)

    with st.chat_message("assistant", avatar="🤖"):
        container = st.empty()
        current_usage = {"input": 0, "output": 0}
        
        try:
            # Use st.status for better interactive experience
            with st.status("Thinking...", expanded=True) as status:
                
                # 1. Route Decision
                status.write("🤔 Analyzing user intent...")
                route = decide_route(prompt)
                status.write(f"⚙️ Mode Match: **{route}**")
                
                # === Path A: SQL Stats Mode ===
                if "STATS" in route:
                    status.write("📝 Generating SQL query...")
                    sql_query, result_df, sql_usage = generate_and_run_sql(prompt)
                    
                    # Accumulate Tokens
                    current_usage["input"] += sql_usage["input"]
                    current_usage["output"] += sql_usage["output"]
                    
                    if result_df is not None:
                        status.write("🔍 BigQuery search complete, organizing answer...")
                        data_str = result_df.head(100).to_csv(index=False)
                        
                        ans_prompt = f"""
                        User Question: '{prompt}'
                        Data (CSV): {data_str}
                        INSTRUCTION: Answer completely based on data. Render tables if needed.
                        """
                        resp = generative_model.generate_content(ans_prompt)
                        
                        if resp.usage_metadata:
                            current_usage["input"] += resp.usage_metadata.prompt_token_count
                            current_usage["output"] += resp.usage_metadata.candidates_token_count
                        
                        full_response = resp.text
                        status.update(label="✅ Done!", state="complete", expanded=False)
                        
                        # Display Results
                        container.markdown(full_response)
                        
                        # Calculate Cost
                        turn_cost = calculate_cost(current_model_name, current_usage["input"], current_usage["output"])
                        stats_dict = {"input": current_usage["input"], "output": current_usage["output"], "cost": turn_cost}
                        
                        # Update Session Stats
                        st.session_state.total_cost += turn_cost
                        st.session_state.total_tokens += (current_usage["input"] + current_usage["output"])
                        
                        # Save Message
                        st.session_state.messages.append({
                            "role": "assistant",
                            "content": full_response,
                            "sql_code": sql_query,
                            "data_table": result_df,
                            "usage_stats": stats_dict
                        })
                        st.rerun() # Force rerun to update sidebar stats
                        
                    else:
                        status.update(label="❌ SQL Execution Failed", state="error")
                        container.error(f"Unable to execute query: {sql_query}")

                # === Path B: Vector Search Mode ===
                else:
                    status.write("🔎 Performing hybrid vector search...")
                    q_vector = get_query_vector(prompt)
                    filters = {"min_score": score_range[0], "start_date": start_date, "end_date": end_date}
                    
                    if q_vector:
                        df = search_vectors_hybrid(q_vector, prompt, filters, 20)
                    else:
                        df = None

                    if df is not None and not df.empty:
                        status.write("📚 Relevant documents found, generating answer...")
                        context_lines = [f"{row['page_content']}" for _, row in df.iterrows()]
                        full_context = "\n".join(context_lines)
                        
                        status.update(label="🤖 Generating...", state="running", expanded=False)
                        
                        stream = ask_gemini_stream(prompt, full_context, st.session_state.messages)
                        full_response = ""
                        
                        for chunk in stream:
                            if chunk.usage_metadata:
                                current_usage["input"] = chunk.usage_metadata.prompt_token_count
                                current_usage["output"] = chunk.usage_metadata.candidates_token_count
                            if chunk.text:
                                full_response += chunk.text
                                container.markdown(full_response + "▌")
                        
                        container.markdown(full_response)
                        
                        # Calculate Cost
                        turn_cost = calculate_cost(current_model_name, current_usage["input"], current_usage["output"])
                        stats_dict = {"input": current_usage["input"], "output": current_usage["output"], "cost": turn_cost}

                        # Update Session Stats
                        st.session_state.total_cost += turn_cost
                        st.session_state.total_tokens += (current_usage["input"] + current_usage["output"])
                        
                        # Prepare Preview Data
                        preview_cols = ['product_category_name', 'price', 'customer_city', 'review_score']
                        valid_cols = [c for c in preview_cols if c in df.columns]
                        
                        st.session_state.messages.append({
                            "role": "assistant",
                            "content": full_response,
                            "source_preview": df[valid_cols].head(5),
                            "usage_stats": stats_dict
                        })
                        st.rerun()

                    else:
                        status.update(label="⚠️ No Data Found", state="error")
                        container.warning("No relevant data found, please try adjusting the filters.")

        except Exception as e:
            st.error(f"System Error: {safe_error(e)}")