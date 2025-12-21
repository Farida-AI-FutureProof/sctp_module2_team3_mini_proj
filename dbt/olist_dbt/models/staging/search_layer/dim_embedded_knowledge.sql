{{ config(
    materialized='table'
) }}

WITH source AS (
    SELECT * FROM {{ ref('init_search_unioned') }}
)

SELECT
    -- --- 1. Primary Key ---
    review_id AS knowledge_id,
    
    -- --- 2. Pass-through Columns (must be updated: pass through all new fields to the vector table) ---
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
    payment_type,

    -- --- 3. Page Content Construction (must be updated: add new fields into semantic text) ---
    CONCAT(
        'Category: ', COALESCE(product_category_name, 'Unknown'), '. ',
        'Total Price: $', CAST(price AS STRING), '. ',
        'Freight: $', CAST(freight_value AS STRING), '. ',
        'Weight: ', CAST(product_weight_g AS STRING), 'g. ',
        'Payment: ', COALESCE(payment_type, 'Unknown'), '. ',
        'Status: ', COALESCE(order_status, 'Unknown'), '. ',
        'Location: Customer in ', COALESCE(customer_city, 'Unknown'), 
        ' bought from ', COALESCE(seller_city, 'Unknown'), '. ',
        'Review Score: ', CAST(review_score AS STRING), '/5. ',
        'Review Content: ', COALESCE(review_comment_message, 'No content')
    ) AS page_content

FROM source
WHERE review_comment_message IS NOT NULL 
   OR review_score IS NOT NULL
