import streamlit as st
from google.cloud import bigquery
import plotly.express as px
import pandas as pd
import vertexai
from vertexai.language_models import TextEmbeddingModel
from vertexai.generative_models import GenerativeModel
import time
from datetime import date

# ==============================================================================
# 0. Project Configuration & Styles
# ==============================================================================
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

# ==============================================================================
# 1. Page Configuration
# ==============================================================================
st.set_page_config(
    page_title="Olist Integrated Operations Dashboard", 
    page_icon="🛒",
    layout="wide", 
    initial_sidebar_state="expanded"
)

# Custom CSS for visual optimization (Merged from both files)
st.markdown("""
<style>
    .block-container {padding-top: 2rem; padding-bottom: 2rem;}
    .stChatMessage {border-radius: 10px; border: 1px solid #e0e0e0;}
    .stCode {font-family: 'Fira Code', monospace;}
</style>
""", unsafe_allow_html=True)

st.title("📊 Olist E-commerce Interactive Dashboard")

# ==============================================================================
# 2. Shared Utilities & Resource Initialization
# ==============================================================================
def safe_error(e):
    return str(e).replace(PROJECT_ID, "********")

def calculate_cost(model_name, input_tok, output_tok):
    rates = PRICING_RATES.get(model_name, PRICING_RATES["default"])
    cost_input = (input_tok / 1_000_000) * rates["input"]
    cost_output = (output_tok / 1_000_000) * rates["output"]
    return cost_input + cost_output

@st.cache_data(ttl=600)
def get_main_data():
    client = bigquery.Client(project=PROJECT_ID)
    
    query = f"""
    SELECT 
        -- Fact Table
        f.order_purchase_date,
        f.price,
        f.freight_value,
        f.order_status,
        
        -- Dim Products
        p.category_name,
        p.product_weight_g,
        
        -- Dim Customers
        c.customer_state,
        c.customer_city,
        c.geolocation_lat,
        c.geolocation_lng
        
    FROM `{PROJECT_ID}.{DATASET_ID}.fct_order_items` f
    LEFT JOIN `{PROJECT_ID}.{DATASET_ID}.dim_products` p 
        ON f.product_id = p.product_id
    LEFT JOIN `{PROJECT_ID}.{DATASET_ID}.dim_customers` c 
        ON f.customer_id = c.customer_id
    WHERE f.order_purchase_date >= '2017-01-01'
    """
    df = client.query(query).to_dataframe()
    
    # 🔧 [Core Fix]: Convert Decimal type to float to avoid quantile calculation errors
    if not df.empty:
        df['price'] = df['price'].astype(float)
        df['freight_value'] = df['freight_value'].astype(float)
        df['product_weight_g'] = df['product_weight_g'].astype(float)
        
    return df

@st.cache_resource
def init_ai_resources():
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

# ==============================================================================
# 3. Sidebar: Global Filters (Dashboard) & AI Control Panel
# ==============================================================================
st.sidebar.header("🔍 Dashboard Filters")

# -- Load Main Data for Dashboard Filters --
try:
    with st.spinner('Loading dashboard data...'):
        df_raw = get_main_data()
        df_raw['order_purchase_date'] = pd.to_datetime(df_raw['order_purchase_date'])
except Exception as e:
    st.error(f"Failed to connect to BigQuery: {e}")
    st.stop()

# 3.1 Dashboard Specific Filters
min_date = df_raw['order_purchase_date'].min()
max_date = df_raw['order_purchase_date'].max()
date_range = st.sidebar.date_input(
    "Select Date Range (Dashboard)",
    value=(min_date, max_date),
    min_value=min_date,
    max_value=max_date
)

all_states = sorted(df_raw['customer_state'].dropna().unique())
selected_states = st.sidebar.multiselect(
    "Select Customer State",
    options=all_states,
    default=all_states 
)

all_categories = sorted(df_raw['category_name'].dropna().astype(str).unique())
selected_categories = st.sidebar.multiselect(
    "Select Product Category",
    options=all_categories,
    default=[]
)

st.sidebar.divider()

# ==============================================================================
# 4. Data Filtering Logic (Dashboard)
# ==============================================================================
mask = (
    (df_raw['order_purchase_date'].dt.date >= date_range[0]) &
    (df_raw['order_purchase_date'].dt.date <= date_range[1]) &
    (df_raw['customer_state'].isin(selected_states))
)

if selected_categories:
    mask = mask & (df_raw['category_name'].isin(selected_categories))

df_filtered = df_raw[mask]

# ==============================================================================
# 5. Tab Layout
# ==============================================================================
tab1, tab2, tab3, tab4 = st.tabs([
    "📊 Sales Trends", 
    "📦 Products", 
    "🚚 Logistics", 
    "🤖 RAG"
])

# ==============================================================================
# TAB 1, 2, 3: Original Dashboard Content
# ==============================================================================
if df_filtered.empty:
    with tab1: st.warning("No data available under current filters.")
    with tab2: st.warning("No data available under current filters.")
    with tab3: st.warning("No data available under current filters.")
else:
    # --- TAB 1: Sales Trends ---
    with tab1:
        st.markdown("### 📈 Core Operational Performance")
        kpi1, kpi2, kpi3, kpi4 = st.columns(4)
        total_sales = df_filtered['price'].sum()
        total_freight = df_filtered['freight_value'].sum()
        order_count = df_filtered.shape[0]
        avg_ticket = total_sales / order_count if order_count > 0 else 0

        kpi1.metric("Total Sales (GMV)", f"${total_sales:,.0f}", delta_color="normal")
        kpi2.metric("Total Freight Revenue", f"${total_freight:,.0f}")
        kpi3.metric("Order Items Count", f"{order_count:,}")
        kpi4.metric("Average Order Value (AOV)", f"${avg_ticket:.2f}")
        st.divider()

        st.subheader("Sales Trends Over Time")
        freq_option = st.radio("Select Time Granularity", ["Monthly (M)", "Weekly (W)"], horizontal=True)
        freq = 'M' if freq_option == "Monthly (M)" else 'W-MON'
        
        trend_data = df_filtered.set_index('order_purchase_date').resample(freq)['price'].sum().reset_index()
        
        fig_trend = px.line(
            trend_data,
            x='order_purchase_date',
            y='price',
            markers=True,
            title=f"GMV Trend ({freq_option})",
            labels={'price': 'Sales ($)', 'order_purchase_date': 'Date'}
        )
        st.plotly_chart(fig_trend, use_container_width=True)

    # --- TAB 2: Products & Categories ---
    with tab2:
        col_p1, col_p2 = st.columns([1, 1])
        with col_p1:
            st.subheader("Top 10 Best-Selling Categories")
            cat_sales = df_filtered.groupby('category_name')['price'].sum().reset_index()
            top_cats = cat_sales.sort_values('price', ascending=False).head(10)
            
            fig_bar = px.bar(
                top_cats,
                x='price',
                y='category_name',
                orientation='h',
                text_auto='.2s',
                color='price',
                title="Category Sales Contribution Ranking"
            )
            fig_bar.update_layout(yaxis=dict(autorange="reversed"))
            st.plotly_chart(fig_bar, use_container_width=True)
            
        with col_p2:
            st.subheader("Category Price Distribution")
            if not top_cats.empty:
                top_cat_names = top_cats['category_name'].tolist()
                df_top_cats = df_filtered[df_filtered['category_name'].isin(top_cat_names)]
                
                fig_box = px.box(
                    df_top_cats,
                    x='category_name',
                    y='price',
                    title="Unit Price Distribution (Top Categories)",
                    points=False 
                )
                y_limit = df_top_cats['price'].quantile(0.95) 
                fig_box.update_yaxes(range=[0, y_limit * 1.2])
                st.plotly_chart(fig_box, use_container_width=True)
            else:
                st.info("Insufficient data.")

    # --- TAB 3: Logistics & Geography ---
    with tab3:
        col_g1, col_g2 = st.columns([1, 1])
        with col_g1:
            st.subheader("Logistics Analysis: Weight vs Freight")
            if len(df_filtered) > 5000:
                scatter_data = df_filtered.sample(5000)
                st.caption("Note: Data randomly sampled (5000 rows)")
            else:
                scatter_data = df_filtered
                
            fig_scatter = px.scatter(
                scatter_data,
                x='product_weight_g',
                y='freight_value',
                color='category_name', 
                size='price',           
                title="Correlation between Product Weight and Freight",
                labels={'product_weight_g': 'Weight (g)', 'freight_value': 'Freight ($)'},
                hover_data=['order_status']
            )
            st.plotly_chart(fig_scatter, use_container_width=True)

        with col_g2:
            st.subheader("Customer Geographic Heatmap")
            map_df = df_filtered.dropna(subset=['geolocation_lat', 'geolocation_lng'])
            if not map_df.empty:
                sample_size = min(5000, len(map_df))
                fig_map = px.scatter_mapbox(
                    map_df.sample(sample_size),
                    lat="geolocation_lat",
                    lon="geolocation_lng",
                    hover_name="customer_city",
                    color="customer_state",
                    zoom=3,
                    height=500,
                    title=f"Customer Distribution ({sample_size} samples)"
                )
                fig_map.update_layout(mapbox_style="open-street-map", margin={"r":0,"t":40,"l":0,"b":0})
                st.plotly_chart(fig_map, use_container_width=True)
            else:
                st.info("No geolocation data available")

# ==============================================================================
# TAB 4: RAG / AI Assistant (Logic from demo.py)
# ==============================================================================
with tab4:
    # --- RAG Specific Session State ---
    if "messages" not in st.session_state:
        st.session_state.messages = []
    if "total_cost" not in st.session_state:
        st.session_state.total_cost = 0.0
    if "total_tokens" not in st.session_state:
        st.session_state.total_tokens = 0

    # --- Initialize AI Resources ---
    client_ai, embedding_model, generative_model, model_status, current_model_name = init_ai_resources()

    if not client_ai:
        st.error(f"AI System initialization failed: {model_status}")
    else:
        # --- RAG Sub-Sidebar (Displayed in Main Sidebar when this tab is active or just appended) ---
        st.sidebar.markdown("---")
        st.sidebar.header("🤖 AI Assistant Controls")
        
        with st.sidebar.expander("🔎 AI Filters", expanded=False):
            rag_score_range = st.slider("Score Range", 1, 5, (1, 5), key="rag_score")
            rag_start_date = st.date_input("Start Date (Search)", value=date(2017, 1, 1), key="rag_start")
            rag_end_date = st.date_input("End Date (Search)", value=date(2018, 8, 31), key="rag_end")

        col_metric1, col_metric2 = st.sidebar.columns(2)
        col_metric1.metric("AI Tokens", f"{st.session_state.total_tokens:,}")
        col_metric2.metric("AI Cost", f"${st.session_state.total_cost:.4f}")

        if st.sidebar.button("🗑️ Clear Chat", type="primary"):
            st.session_state.messages = []
            st.session_state.total_cost = 0.0
            st.session_state.total_tokens = 0
            st.rerun()

        # --- AI Header ---
        col_h1, col_h2, col_h3 = st.columns([2, 1, 1])
        with col_h1: st.markdown(f"**Model:** `{current_model_name}`")
        with col_h2: st.markdown(f"**Status:** {model_status}")
        with col_h3: st.markdown(f"**Region:** `{LOCATION}`")
        st.divider()

        # --- Helper Functions for AI ---
        def generate_and_run_sql(user_query):
            full_table_ref = f"`{SQL_TABLE}`"
            schema_prompt = f"""
            You are a BigQuery SQL Expert. Write a SQL query to answer the user's question.
            TARGET TABLE: {full_table_ref}
            COLUMNS: order_id, customer_city, order_status, price, freight_value, payment_type, product_category_name, product_weight_g, review_score, order_purchase_timestamp
            RULES: 1. Return ONLY raw SQL. No Markdown. 2. Limit results to 100.
            USER QUESTION: "{user_query}"
            """
            try:
                response = generative_model.generate_content(schema_prompt)
                generated_sql = response.text.replace("```sql", "").replace("```", "").strip()
                usage = {"input": 0, "output": 0}
                if response.usage_metadata:
                    usage["input"] = response.usage_metadata.prompt_token_count
                    usage["output"] = response.usage_metadata.candidates_token_count
                df = client_ai.query(generated_sql).to_dataframe()
                return generated_sql, df, usage
            except Exception as e:
                return str(e), None, {"input": 0, "output": 0}

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
            return client_ai.query(sql, job_config=job_config).to_dataframe()

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

        # --- Chat UI ---
        for msg in st.session_state.messages:
            avatar = "🧑‍💻" if msg["role"] == "user" else "🤖"
            with st.chat_message(msg["role"], avatar=avatar):
                st.markdown(msg["content"])
                if "sql_code" in msg:
                    with st.expander("🛠️ View Generated SQL"): st.code(msg["sql_code"], language="sql")
                if "data_table" in msg:
                    with st.expander("📊 View Source Data"): st.dataframe(msg["data_table"], use_container_width=True)
                if "source_preview" in msg:
                    with st.expander(f"📚 Reference Documents ({len(msg['source_preview'])} items)"): st.dataframe(msg["source_preview"], use_container_width=True)
                if "usage_stats" in msg:
                    stats = msg["usage_stats"]
                    st.caption(f"⚡ Turn Cost: ${stats['cost']:.5f} ({stats['input']} in / {stats['output']} out)")

        if prompt := st.chat_input("Ask Olist AI (e.g., 'Avg freight in Rio 2017?' or 'Reviews about delivery delays')"):
            st.session_state.messages.append({"role": "user", "content": prompt})
            with st.chat_message("user", avatar="🧑‍💻"):
                st.markdown(prompt)

            with st.chat_message("assistant", avatar="🤖"):
                container = st.empty()
                current_usage = {"input": 0, "output": 0}
                
                try:
                    with st.status("Thinking...", expanded=True) as status:
                        status.write("🤔 Analyzing user intent...")
                        route = decide_route(prompt)
                        status.write(f"⚙️ Mode Match: **{route}**")
                        
                        # === Path A: SQL Stats Mode ===
                        if "STATS" in route:
                            status.write("📝 Generating SQL query...")
                            sql_query, result_df, sql_usage = generate_and_run_sql(prompt)
                            current_usage["input"] += sql_usage["input"]
                            current_usage["output"] += sql_usage["output"]
                            
                            if result_df is not None:
                                status.write("🔍 Search complete, organizing answer...")
                                data_str = result_df.head(100).to_csv(index=False)
                                ans_prompt = f"User Question: '{prompt}'\nData (CSV): {data_str}\nINSTRUCTION: Answer completely based on data."
                                resp = generative_model.generate_content(ans_prompt)
                                
                                if resp.usage_metadata:
                                    current_usage["input"] += resp.usage_metadata.prompt_token_count
                                    current_usage["output"] += resp.usage_metadata.candidates_token_count
                                
                                full_response = resp.text
                                status.update(label="✅ Done!", state="complete", expanded=False)
                                container.markdown(full_response)
                                
                                turn_cost = calculate_cost(current_model_name, current_usage["input"], current_usage["output"])
                                stats_dict = {"input": current_usage["input"], "output": current_usage["output"], "cost": turn_cost}
                                st.session_state.total_cost += turn_cost
                                st.session_state.total_tokens += (current_usage["input"] + current_usage["output"])
                                
                                st.session_state.messages.append({
                                    "role": "assistant", "content": full_response,
                                    "sql_code": sql_query, "data_table": result_df, "usage_stats": stats_dict
                                })
                                st.rerun()
                            else:
                                status.update(label="❌ SQL Execution Failed", state="error")
                                container.error(f"Unable to execute query: {sql_query}")

                        # === Path B: Vector Search Mode ===
                        else:
                            status.write("🔎 Performing hybrid vector search...")
                            q_vector = get_query_vector(prompt)
                            filters = {"min_score": rag_score_range[0], "start_date": rag_start_date, "end_date": rag_end_date}
                            
                            if q_vector:
                                df_rag = search_vectors_hybrid(q_vector, prompt, filters, 20)
                            else:
                                df_rag = None

                            if df_rag is not None and not df_rag.empty:
                                status.write("📚 Documents found, generating answer...")
                                context_lines = [f"{row['page_content']}" for _, row in df_rag.iterrows()]
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
                                turn_cost = calculate_cost(current_model_name, current_usage["input"], current_usage["output"])
                                stats_dict = {"input": current_usage["input"], "output": current_usage["output"], "cost": turn_cost}
                                st.session_state.total_cost += turn_cost
                                st.session_state.total_tokens += (current_usage["input"] + current_usage["output"])
                                
                                preview_cols = ['product_category_name', 'price', 'customer_city', 'review_score']
                                valid_cols = [c for c in preview_cols if c in df_rag.columns]
                                
                                st.session_state.messages.append({
                                    "role": "assistant", "content": full_response,
                                    "source_preview": df_rag[valid_cols].head(5), "usage_stats": stats_dict
                                })
                                st.rerun()
                            else:
                                status.update(label="⚠️ No Data Found", state="error")
                                container.warning("No relevant data found for these filters.")

                except Exception as e:
                    st.error(f"System Error: {safe_error(e)}")

# ==============================================================================
# 6. Global Data Download (Dashboard Side)
# ==============================================================================
st.sidebar.divider()
st.sidebar.download_button(
    label="📥 Download Filtered Dashboard Data",
    data=df_filtered.to_csv(index=False).encode('utf-8'),
    file_name='filtered_olist_data.csv',
    mime='text/csv',
)