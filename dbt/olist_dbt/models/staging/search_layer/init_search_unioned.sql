{{ config(
    materialized='table'
) }}

WITH orders AS (
    SELECT 
        order_id,
        customer_id,
        order_status,
        order_purchase_at AS order_purchase_timestamp,
        order_approved_at,
        order_delivered_carrier_at AS order_delivered_carrier_date,
        order_delivered_customer_at AS order_delivered_customer_date
    FROM {{ ref('stg_search_orders') }}
),

payments_agg AS (
    SELECT
        order_id,
        STRING_AGG(DISTINCT payment_type, ', ') as payment_type
    FROM {{ ref('stg_search_payments') }}
    GROUP BY order_id
),

-- Aggregate Items (resolve one-to-many issue)
order_items_agg AS (
    SELECT 
        i.order_id,
        -- Metrics
        COUNT(i.order_item_id) as total_items_count,
        SUM(i.price) as total_order_value,
        -- ✅ New: aggregate freight cost
        SUM(i.freight_value) as total_freight_value,
        
        -- ✅ New: get the maximum item weight in the order (for logistics reference)
        MAX(p.product_weight_g) as product_weight_g,
        
        -- ✅ New: get the primary product category
        ANY_VALUE(p.product_category_name) as main_product_category,

        -- LLM text summary
        STRING_AGG(
            CONCAT(
                COALESCE(p.product_category_name, 'Unknown'), 
                ' ($', CAST(i.price AS STRING), ')'
            ), 
            ', '
        ) as item_summary_list,
        
        -- Seller Info
        ANY_VALUE(s.seller_city) as main_seller_city

    FROM {{ ref('stg_search_order_items') }} i
    LEFT JOIN {{ ref('stg_search_products') }} p ON i.product_id = p.product_id
    LEFT JOIN {{ ref('stg_search_sellers') }} s ON i.seller_id = s.seller_id
    GROUP BY i.order_id
),

reviews AS (
    SELECT * FROM {{ ref('stg_search_reviews') }}
),

customers AS (
    SELECT * FROM {{ ref('stg_search_customers') }}
)

SELECT
    -- IDs
    r.review_id,
    o.order_id,
    o.customer_id,
    
    -- Review Data
    r.review_score,
    r.review_comment_message,
    -- (Optional: if you need title or creation date, you can add them here)

    -- Order Data (using corrected column names)
    o.order_status,
    o.order_purchase_timestamp,
    o.order_approved_at,
    o.order_delivered_carrier_date,
    o.order_delivered_customer_date,

    -- Customer Data
    c.customer_state,
    c.customer_city,

    -- ✅ Aggregated Item & Payment Data
    agg.main_product_category as product_category_name,
    COALESCE(agg.product_weight_g, 0) as product_weight_g,
    COALESCE(agg.total_order_value, 0) as price,
    COALESCE(agg.total_freight_value, 0) as freight_value,
    
    pay.payment_type, -- from payments_agg
    
    agg.main_seller_city as seller_city,
    agg.item_summary_list

FROM reviews r
-- Join Reviews to Orders
INNER JOIN orders o ON r.order_id = o.order_id
-- Join Orders to Items
LEFT JOIN order_items_agg agg ON o.order_id = agg.order_id
-- Join Orders to Payments
LEFT JOIN payments_agg pay ON o.order_id = pay.order_id
-- Join Customers
LEFT JOIN customers c ON o.customer_id = c.customer_id