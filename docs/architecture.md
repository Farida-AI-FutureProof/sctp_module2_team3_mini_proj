┌───────────────────────┐
│   Olist CSV Datasets  │
│  (Orders, Reviews,    │
│   Customers, Geo)     │
└───────────┬───────────┘
            │
            ▼
┌───────────────────────┐
│      Meltano          │
│  tap-csv (Extract)    │
│  target-bigquery      │
│  (Load)               │
└───────────┬───────────┘
            │
            ▼
┌───────────────────────┐
│   Data Warehouse /    │
│   Staging Layer       │
│  (Raw Tables)         │
└───────────┬───────────┘
            │
            ▼
┌───────────────────────┐
│ Transformation Layer  │
│  (Pandas / dbt)       │
│ - delivery_delay_days │
│ - delay_bucket        │
│ - repeat buyer flag   │
│ - late risk score     │
└───────────┬───────────┘
            │
            ▼
┌───────────────────────┐
│ Analytics & Viz Layer │
│  Python (Pandas)      │
│  Matplotlib / Plotly  │
└───────────────────────┘
