# Technical Decisions & Tooling Rationale

This project was designed to balance **analytical depth**, **engineering clarity**, and **time efficiency**, while following modern data-engineering best practices.

---

## 1. Data Ingestion — Meltano (tap-csv → target-bigquery)

**Why Meltano was chosen:**
- Provides a **structured ELT framework** rather than ad-hoc scripts
- Clear separation of concerns: extract, load, transform
- Reproducible ingestion via declarative configuration (`meltano.yml`)
- Industry-relevant tool aligned with modern analytics engineering workflows

**Alternative considered:**  
Custom Python ingestion scripts  
**Why not used:**  
Higher maintenance cost, weaker lineage visibility, less standardised.

---

## 2. Data Storage — Staging / Raw Tables

**Design choice:**
- Ingest data into raw, untransformed tables
- Preserve source fidelity before transformation

**Why this matters:**
- Supports traceability and debugging
- Enables schema evolution
- Mirrors real-world data warehouse practices

---

## 3. Transformation Layer — Pandas (with dbt-ready logic)

**Why Pandas was used:**
- Flexible and expressive for feature engineering
- Suitable for exploratory and analytical transformations
- Fast iteration for complex metrics such as:
  - Delivery delay buckets
  - Repeat buyer identification
  - Late delivery risk score

**Design consideration:**
- Transformations were written in a way that can be **easily ported to dbt/SQL** in future iterations.

---

## 4. Feature Engineering Decisions

Key engineered features include:
- `delivery_delay_days`
- `delay_bucket` (on-time, 1–3 days late, etc.)
- `is_repeat_buyer`
- `late_risk_score` (frequency × severity)

**Why engineered features were used:**
- Raw timestamps are not business-interpretable
- Features align directly to business questions:
  - Customer satisfaction
  - Loyalty behaviour
  - Regional operational risk

---

## 5. Visualisation — Matplotlib & Plotly

**Why Matplotlib:**
- Full control over chart formatting
- Publication-quality static visuals for executive slides
- Lightweight and reproducible

**Why Plotly (for map):**
- Interactive hover data for multi-metric inspection
- Better geographic storytelling
- Clear visual prioritisation of high-risk regions

---

## 6. Overall Architecture Rationale

The chosen stack prioritises:
- **Clarity over complexity**
- **Explainability over black-box tooling**
- **Business interpretability over raw technical optimisation**

This approach ensures insights are not only technically sound, but also **actionable and executive-ready**.
