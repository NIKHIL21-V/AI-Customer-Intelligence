# AI Customer Intelligence & Predictive Analytics Platform

## Project Overview

The AI Customer Intelligence & Predictive Analytics Platform is a data analytics project designed to analyze retail sales, understand customer purchasing behavior, segment customers, and identify potential customer churn risks.

The project uses Python, Pandas, Scikit-learn, and Streamlit to transform retail transaction data into meaningful business insights through an interactive dashboard.

## Objectives

* Analyze sales revenue and order performance.
* Identify top-selling products.
* Understand customer purchasing behavior.
* Segment customers using RFM analysis.
* Estimate customer churn risk using machine learning.
* Present insights through an interactive dashboard.

## Key Features

### 1. Sales Analytics

* Total revenue and order analysis.
* Customer-level sales analysis.
* Monthly revenue trends.
* Top-selling product identification.

### 2. Customer Segmentation

* Recency, Frequency, and Monetary (RFM) analysis.
* Customer segment classification.
* Customer revenue comparisons.
* Customer segment summaries.

### 3. Customer Churn Prediction

* Customer inactivity analysis.
* Machine learning using Random Forest.
* Churn risk distribution.
* Identification of customers with higher estimated churn risk.
* Downloadable churn prediction results.

### 4. Interactive Dashboard

* Built using Streamlit.
* Visualizations created with Plotly.
* Interactive date filters.
* Customer search and data tables.
* Downloadable CSV reports.

## Technologies Used

* Python
* Pandas
* NumPy
* Scikit-learn
* Streamlit
* Plotly
* Matplotlib
* Seaborn
* Joblib
* Jupyter Notebook

## Dataset

This project uses the **Online Retail dataset from the UCI Machine Learning Repository**.

The dataset contains retail transaction information, including invoice numbers, product descriptions, quantities, invoice dates, unit prices, and customer identifiers.

## Project Structure

```text
AI_Customer_Intelligence/
├── data/
│   ├── Online Retail.xlsx
│   └── cleaned_sales.csv
├── notebooks/
│   └── 01_data_exploration.ipynb
├── models/
│   └── churn_model.joblib
├── reports/
│   ├── rfm_customer_segments.csv
│   ├── customer_revenue.csv
│   ├── customer_segment_summary.csv
│   ├── customer_churn_predictions.csv
│   └── churn_model_metrics.txt
├── app.py
├── churn_model.py
├── requirements.txt
├── .gitignore
└── README.md
```

*Note: The listed files represent the intended project structure. Confirm that each file exists in your local project before publishing.*

## Installation and Setup

### Step 1: Clone the repository

```bash
git clone YOUR_GITHUB_REPOSITORY_URL
```

### Step 2: Navigate to the project folder

```bash
cd AI_Customer_Intelligence
```

### Step 3: Create a virtual environment

```bash
python -m venv venv
```

### Step 4: Activate the virtual environment

On Windows:

```bash
venv\Scripts\activate
```

### Step 5: Install dependencies

```bash
python -m pip install -r requirements.txt
```

### Step 6: Run the churn prediction model

```bash
python churn_model.py
```

### Step 7: Launch the dashboard

```bash
python -m streamlit run app.py
```

Open the local URL displayed in your terminal to access the dashboard.

## Analytical Methods

### RFM Analysis

RFM analysis evaluates customers based on:

* **Recency:** How recently a customer made a purchase.
* **Frequency:** How often a customer made purchases.
* **Monetary:** How much a customer spent.

These measures help classify customers into groups such as Champions, Loyal Customers, Recent Customers, At-Risk Customers, and Regular Customers.

### Churn Prediction

A Random Forest classifier estimates customer churn risk using historical purchasing behavior.

The current demonstration defines churn using purchase inactivity during a 90-day period. Its predictions and evaluation results should be interpreted in light of this definition and the current validation approach.

## Business Value

This project demonstrates how retail businesses can use data analytics to:

* Monitor sales performance.
* Discover important products and customers.
* Understand customer purchasing patterns.
* Identify customers who may need retention campaigns.
* Support data-driven business decisions.

## Future Improvements

* Add sales forecasting.
* Improve churn prediction with time-based model validation.
* Connect the dashboard to a SQL database.
* Add automated data ingestion and refresh.
* Deploy the application online.
* Introduce a natural-language AI analytics assistant.

## Author

**Nikhil Sai Subhash**

Aspiring Data Analyst | Customer Analytics | Business Intelligence | Predictive Analytics

## Disclaimer

This project is intended for educational and portfolio purposes. Churn predictions are estimates and should not be treated as confirmed customer behavior.
