{{ config(
    materialized='table',
    cluster_by = ['category'],
    pre_hook="
        CREATE OR REPLACE MODEL `{{ target.project }}.{{ target.dataset }}.bert_embedding_model`
        REMOTE WITH CONNECTION `{{ target.project }}.US.vertex_ai_conn`
        OPTIONS (ENDPOINT = 'text-embedding-005');
    ",
    post_hook="
        CREATE VECTOR INDEX IF NOT EXISTS my_vector_index
        ON {{ this }}(ml_generate_embedding_result)
        OPTIONS(distance_type='COSINE', index_type='IVF');
    "
) }}

WITH source_knowledge AS (
    SELECT * FROM {{ ref('dim_embedded_knowledge') }}
),

embedded_data AS (
    SELECT
        *
    FROM
        ML.GENERATE_EMBEDDING(
            MODEL `{{ target.project }}.{{ target.dataset }}.bert_embedding_model`,
            (
                SELECT
                    knowledge_id,
                    page_content AS content,
                    
                    -- ✅ You must select all fields here that you want to use in metadata
                    order_id,
                    customer_id,
                    customer_city,
                    order_status,
                    order_purchase_timestamp,
                    order_approved_at,
                    order_delivered_carrier_date,
                    order_delivered_customer_date,
                    review_score,
                    review_comment_message,
                    product_category_name,
                    product_weight_g,
                    price,
                    freight_value,
                    seller_city,
                    payment_type
                FROM source_knowledge
            ),
            STRUCT(TRUE AS flatten_json_output)
        )
)

SELECT
    knowledge_id,
    content AS page_content,
    ml_generate_embedding_result,
    ml_generate_embedding_status,
    
    -- Standalone columns for easier BigQuery preview
    product_category_name AS category,
    review_score,
    
    -- ✅ FULL METADATA STRUCT (fully matches your field list)
    STRUCT(
        order_id AS order_id,
        customer_id AS customer_id,
        customer_city AS customer_city,
        order_status AS order_status,
        order_purchase_timestamp AS order_purchase_timestamp,
        order_approved_at AS order_approved_at,
        order_delivered_carrier_date AS order_delivered_carrier_date,
        order_delivered_customer_date AS order_delivered_customer_date,
        review_score AS review_score,
        review_comment_message AS review_comment_message,
        product_category_name AS product_category_name,
        product_weight_g AS product_weight_g,
        price AS price,
        freight_value AS freight_value,
        seller_city AS seller_city,
        payment_type AS payment_type
    ) AS metadata

FROM embedded_data
