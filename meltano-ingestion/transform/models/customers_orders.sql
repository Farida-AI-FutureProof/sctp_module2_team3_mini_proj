-- customers_orders.sql

{{ config(materialized='table') }}

with orders as (
    select * from {{ source('olist_raw', 'orders') }}
),
customers as (
    select * from {{ source('olist_raw', 'customer') }}
)
select
    c.customer_id,
    c.customer_unique_id,
    o.order_id,
    o.order_status,
    o.order_purchase_timestamp
from customers c
left join orders o
    on c.customer_id = o.customer_id
