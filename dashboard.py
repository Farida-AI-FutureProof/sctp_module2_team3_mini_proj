import streamlit as st
from google.cloud import bigquery
import plotly.express as px
import pandas as pd

# ==============================================================================
# 0. Project Configuration
# ==============================================================================
PROJECT_ID = "my-project-sctp-module-2"
DATASET_ID = "olist_dbt_dataset"
LOCATION = "us-central1"

# ==============================================================================
# 1. Page Configuration
# ==============================================================================
st.set_page_config(page_title="Olist Interactive Operations Dashboard", layout="wide", initial_sidebar_state="expanded")
st.title("📊 Olist E-commerce Interactive Dashboard")

# ==============================================================================
# 2. Data Retrieval (Core Logic)
# ==============================================================================
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
    # BigQuery's NUMERIC type becomes Python's Decimal, which causes many math operations to fail
    if not df.empty:
        df['price'] = df['price'].astype(float)
        df['freight_value'] = df['freight_value'].astype(float)
        # It is also recommended to convert product_weight_g to prevent accidents
        df['product_weight_g'] = df['product_weight_g'].astype(float)
        
    return df

try:
    with st.spinner('Loading full data and building cache...'):
        df_raw = get_main_data()
        df_raw['order_purchase_date'] = pd.to_datetime(df_raw['order_purchase_date'])
except Exception as e:
    st.error(f"Failed to connect to BigQuery: {e}")
    st.stop()

# ==============================================================================
# 3. Sidebar: Global Filters (Interaction & Selection)
# ==============================================================================
st.sidebar.header("🔍 Global Filters")

# 3.1 Date Filter
min_date = df_raw['order_purchase_date'].min()
max_date = df_raw['order_purchase_date'].max()
date_range = st.sidebar.date_input(
    "Select Date Range",
    value=(min_date, max_date),
    min_value=min_date,
    max_value=max_date
)

# 3.2 State Filter (Multiselect)
all_states = sorted(df_raw['customer_state'].dropna().unique())
selected_states = st.sidebar.multiselect(
    "Select Customer State",
    options=all_states,
    default=all_states # Select all by default
)

# 3.3 Category Filter (Multiselect)
all_categories = sorted(df_raw['category_name'].dropna().astype(str).unique()) # Ensure conversion to str to avoid sorting errors
selected_categories = st.sidebar.multiselect(
    "Select Product Category",
    options=all_categories,
    default=[] # Default empty selection represents "All"
)

# ==============================================================================
# 4. Data Filtering Logic
# ==============================================================================
mask = (
    (df_raw['order_purchase_date'].dt.date >= date_range[0]) &
    (df_raw['order_purchase_date'].dt.date <= date_range[1]) &
    (df_raw['customer_state'].isin(selected_states))
)

if selected_categories:
    mask = mask & (df_raw['category_name'].isin(selected_categories))

df_filtered = df_raw[mask]

# Check if data exists after filtering
if df_filtered.empty:
    st.warning("No data available under current filters. Please adjust the sidebar filters.")
    st.stop()

# ==============================================================================
# 5. Key Performance Indicators (KPIs)
# ==============================================================================
st.markdown("### 📈 Core Operational Performance (Based on Filtered Results)")
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

# ==============================================================================
# 6. Detailed Analysis (Tabs)
# ==============================================================================
tab1, tab2, tab3 = st.tabs(["📊 Sales Trends & Overview", "📦 Product & Category Analysis", "🚚 Logistics & Geography"])

# --- TAB 1: Sales Trends ---
with tab1:
    st.subheader("Sales Trends Over Time")
    
    # Aggregate by month or week
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
        st.subheader("Top 10 Best-Selling Categories (by Revenue)")
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
        st.subheader("Category Price Distribution (Boxplot)")
        if not top_cats.empty:
            top_cat_names = top_cats['category_name'].tolist()
            df_top_cats = df_filtered[df_filtered['category_name'].isin(top_cat_names)]
            
            fig_box = px.box(
                df_top_cats,
                x='category_name',
                y='price',
                title="Unit Price Distribution of Major Categories (Excluding Outliers)",
                points=False 
            )
            # Since price is now a float, the quantile calculation will not fail
            y_limit = df_top_cats['price'].quantile(0.95) 
            fig_box.update_yaxes(range=[0, y_limit * 1.2])
            st.plotly_chart(fig_box, use_container_width=True)
        else:
            st.info("Insufficient data to show price distribution")

# --- TAB 3: Logistics & Geography ---
with tab3:
    st.caption("In-depth analysis based on dim_customers and product weight data")
    
    col_g1, col_g2 = st.columns([1, 1])
    
    with col_g1:
        st.subheader("Logistics Analysis: Weight vs Freight")
        if len(df_filtered) > 5000:
            scatter_data = df_filtered.sample(5000)
            st.caption("Note: Data randomly sampled (5000 rows) to optimize display speed")
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
                title=f"Customer Distribution (Showing {sample_size} samples)"
            )
            fig_map.update_layout(mapbox_style="open-street-map", margin={"r":0,"t":40,"l":0,"b":0})
            st.plotly_chart(fig_map, use_container_width=True)
        else:
            st.info("No geolocation data available")

# ==============================================================================
# 7. Data Download Function
# ==============================================================================
st.sidebar.divider()
st.sidebar.download_button(
    label="📥 Download Filtered Data (CSV)",
    data=df_filtered.to_csv(index=False).encode('utf-8'),
    file_name='filtered_olist_data.csv',
    mime='text/csv',
)