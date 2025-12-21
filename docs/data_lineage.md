olist_orders
    │
    ├── order_estimated_delivery_date
    ├── order_delivered_customer_date
    │
    ▼
delivery_delay_days
    │
    ▼
delay_bucket ────────────────┐
                              │
olist_order_reviews            │
    │                           │
    └── review_score ───────────┤
                                ▼
                    avg_review_score (Viz 1)

olist_customers
    │
    └── customer_id
            │
            ▼
     repeat buyer flag
            │
            ▼
    repeat purchase rate (Viz 2)

delivery_delay_days
    │
    ├── pct_late
    ├── avg_late_days_if_late
    │
    ▼
late_risk_score
    │
    ▼
state risk map (Viz 3)
