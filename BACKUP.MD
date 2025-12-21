# BACKUP — Farida’s Analysis + Recommendations (NTU SCTP Module 2)

This file is a **single source of truth** for:
- Final business analysis (insights)
- Recommendations
- Speaker notes (plain English)
- What we built (technical narrative)
- Challenges + resolutions

Use this as:
- A fallback if slides get duplicated or formatting breaks
- A reference during team alignment
- A source for copy/paste into Google Slides

---

## 1. Executive Summary (for instructors)

We built an end-to-end ELT + analytics workflow using Olist e-commerce data:
- Ingested raw CSV data into BigQuery (raw layer)
- Modeled clean staging + marts using dbt
- Built a validated fact table at order-item grain (`fct_order_items`)
- Produced business-facing outputs (monthly revenue by category 2016–2018) used for reporting and recommendations

Outcome: an analytics-ready star schema plus actionable insights on growth, category concentration, and peak planning.

---

## 2. Project Flow (Plain English)

1) We started with raw marketplace data (orders, items, products, customers, payments, etc.)  
2) Loaded it into a warehouse (BigQuery) so all analysis uses a consistent data source  
3) Used dbt to clean and transform into reliable tables (facts + dimensions)  
4) Validated the key fact model (`fct_order_items`)  
5) Answered business questions using aggregated reporting tables (monthly revenue by category)  
6) Created insights + recommendations and prepared slides for stakeholders

---

## 3. Project Flow (Technical)

### Tools
- Meltano: ingestion
- BigQuery: warehouse
- dbt: transformations + tests
- Google Sheets: reporting export / charting
- Google Slides: presentation
- GitHub: version control & collaboration

### Data modeling approach
- Star schema (dimensional model)
- Central fact: `fct_order_items` (order line item grain)
- Dimensions include product/category, customer, seller (depending on team scope)

### Why order-item grain matters
It ensures revenue and order metrics aggregate correctly without double counting, and allows slicing by product/category accurately.

---

## 4. Business Questions → Insights → Recommendations

### BQ1: How is revenue evolving over time (2016–2018)?
**Insight**
- Revenue shows sustained growth from 2016 through 2017.
- 2018 (partial year) tracks strongly and can exceed the prior year trend.
- Month-to-month variation indicates seasonal patterns.

**Recommendations**
- Use monthly trend for planning: align campaigns, inventory, and logistics capacity.
- Monitor MoM trend per category to detect early decline or emerging categories.
- Pair revenue trend with operational KPIs (delivery delays, review scores) to avoid growth-at-any-cost.

**Speaker Notes**
“This gives leadership a pulse on growth and seasonality. Even without causal claims, we can see patterns useful for planning.”

---

### BQ2: Which categories drive the most revenue? Is revenue concentrated?
**Insight**
- Revenue is concentrated: a small number of categories contribute a disproportionately large share.
- Top categories are the safest investment targets, but concentration is also a business risk.

**Recommendations**
- Double down on top categories with high ROI campaigns and cross-sell bundles.
- Build a diversification strategy: nurture mid-tier categories with consistent demand.
- Create a quarterly ‘category watchlist’ to track rising categories and reduce dependency risk.

**Speaker Notes**
“This is the classic Pareto effect in marketplaces. Great for profit, risky for resilience.”

---

### BQ3: How do we operationalize seasonality?
**Insight**
- Peak months imply demand surges that stress supply chain and customer experience.

**Recommendations**
- Create a ‘Peak Month Readiness’ checklist: supplier buffers, warehouse capacity, delivery SLA monitoring.
- Coordinate marketing calendar with operations capacity.
- Post-peak review: track whether review scores or delivery delays degrade after peaks.

**Speaker Notes**
“Growth is only good if customer experience does not collapse during peaks.”

---

## 5. Challenges Faced + Resolution

### Challenge 1: Data consistency across raw sources
- Multiple datasets with different formats required careful staging and casting.
**Resolution:** Standardized staging models in dbt.

### Challenge 2: Building `fct_order_items` reliably
- Ensuring correct joins and grain was essential to avoid duplicate revenue.
**Resolution:** Fixed dependencies and validated output.

### Challenge 3: Git synchronization issues (push rejected)
- Remote had commits not in local.
**Resolution:** Used stash → pull/rebase → push.

### Challenge 4: Apps Script chart automation differences
- Some methods (e.g., chart insertion helpers) behave differently than expected.
**Resolution:** Use simplified slide generation and/or manual chart insertion for final deck.

---

## 6. Final Deliverables
- Validated warehouse models (dbt marts)
- `fct_order_items` populated and validated
- Monthly revenue by product category (2016–2018) exported to Google Sheets
- Slides: business questions + insights + recommendations
- Notebooks:
  - `notebooks/NTU_Module2_Olist_Analysis.ipynb`
  - `notebooks/01_data_validation_fct_order_items.ipynb`

---

## 7. Next Steps (If more time)
- Add delivery performance analysis (delivery delay vs review score)
- Add cohort analysis (repeat buyers over time)
- Add customer segmentation by category purchase behavior
- Automate slide generation fully using a stable Apps Script approach

