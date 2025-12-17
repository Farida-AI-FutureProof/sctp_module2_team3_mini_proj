{{ config(
    materialized='table'
) }}

WITH order_base AS (
    SELECT
        order_id,
        customer_id,
        order_status,
        order_purchase_timestamp,
        order_approved_at,
        order_delivered_carrier_date,
        order_delivered_customer_date
    FROM
        {{ source('olist_raw', 'orders') }}
),

customer_info AS (
    SELECT
        customer_id,
        customer_city
    FROM
        {{ source('olist_raw', 'customer') }}
),

reviews_info AS (
    SELECT
        order_id,
        review_score,
        review_comment_message
    FROM
        {{ source('olist_raw', 'order_reviews') }}
),

order_items_info AS (
    SELECT
        order_id,
        product_id,
        price,
        freight_value,
        seller_id
    FROM
        {{ source('olist_raw', 'order_items') }}
),

products_info AS (
    SELECT
        product_id,
        product_category_name,
        product_weight_g,
        product_length_cm,
        product_height_cm,
        product_width_cm
    FROM
        {{ source('olist_raw', 'products') }}
),

sellers_info AS (
    SELECT
        seller_id,
        seller_city
    FROM
        {{ source('olist_raw', 'sellers') }}
),

payments_info AS (
    SELECT
        order_id,
        payment_type
    FROM
        {{ source('olist_raw', 'order_payments') }}
)

SELECT
    o.order_id,
    c.customer_id,
    c.customer_city,
    o.order_status,
    o.order_purchase_timestamp,
    o.order_approved_at,
    o.order_delivered_carrier_date,
    o.order_delivered_customer_date,
    r.review_score,
    r.review_comment_message,
    p.product_category_name,
    p.product_weight_g,
    p.product_length_cm,
    p.product_height_cm,
    p.product_width_cm,
    oi.price,
    oi.freight_value,
    s.seller_city,
    pay.payment_type
FROM
    order_base o
LEFT JOIN customer_info c
    ON o.customer_id = c.customer_id
LEFT JOIN reviews_info r
    ON o.order_id = r.order_id
LEFT JOIN order_items_info oi
    ON o.order_id = oi.order_id
LEFT JOIN products_info p
    ON oi.product_id = p.product_id
LEFT JOIN sellers_info s
    ON oi.seller_id = s.seller_id
LEFT JOIN payments_info pay
    ON o.order_id = pay.order_id
