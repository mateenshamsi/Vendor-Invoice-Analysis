# Vendor Invoice Intelligence System

**Freight Cost Prediction & Invoice Risk Flagging**

## 📌 Table of Contents
- <a href="#project-overview">Project Overview</a>
- <a href="#business-objectives">Business Objectives</a>
- <a href="#data-sources">Data Sources</a>
- <a href="#eda">Exploratory Data Analysis</a>
- <a href="#models-used">Models Used</a>
- <a href="#metrics">Evaluation Metrics</a>
- <a href="#application">Application</a>
- <a href="#project-structure">Project Structure</a>
- <a href="#how-to-run-this-project">How to Run This Project</a>
- <a href="#author-contact">Author & Contact</a>

---

<h2><a class="anchor" id="project-overview"></a>📌 Project Overview</h2>

This project implements an **end-to-end machine learning system** designed to support finance teams by:

1. **Predicting expected freight cost** for vendor invoices.
2. **Flagging high-risk invoices** that require manual review due to abnormal cost, freight, or operational patterns.

---

<h2><a class="anchor" id="business-objectives"></a>🎯 Business Objectives</h2>

- Forecast freight costs to support budgeting and vendor negotiations.
- Identify invoices that may need manual review.
- Help finance teams prioritize invoice checks.
- Reduce repetitive manual work in invoice review.

---

<h2><a class="anchor" id="data-sources"></a>🗃️ Data Sources</h2>

The invoice flagging model uses data from a SQLite database containing vendor invoice and purchase records.

The training query combines the `vendor_invoice` and `purchases` tables using `PONumber`. It uses invoice details, purchase totals, and receiving dates to prepare model features and labels.

The database path is configured in `invoice_flagging/train.py`. Update it to match the location of your database.

---

<h2><a class="anchor" id="eda"></a>🔎 Exploratory Data Analysis</h2>

Exploratory data analysis can be used to understand the invoice and purchase data before training. Recommended areas include:

- Missing values and duplicate records
- Invoice dollars compared with purchase dollars
- Freight cost distribution
- Purchase receiving delays
- Flagged versus unflagged invoice counts

Add charts and findings from your analysis here.

---

<h2><a class="anchor" id="models-used"></a>🤖 Models Used</h2>

### Freight Cost Prediction

The freight prediction module loads a saved model from:

```text
models/predict_freight_model.pkl