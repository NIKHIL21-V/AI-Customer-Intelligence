# ============================================================
# AI CUSTOMER INTELLIGENCE & PREDICTIVE ANALYTICS PLATFORM
# Complete Streamlit Dashboard
# ============================================================

from pathlib import Path

import pandas as pd
import plotly.express as px
import streamlit as st


# ============================================================
# 1. PAGE CONFIGURATION
# ============================================================

st.set_page_config(
    page_title="AI Customer Intelligence Platform",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded",
)
# ============================================================
# PROFESSIONAL DASHBOARD STYLING - CONSISTENT THEME
# ============================================================

st.markdown("""
<style>

/* Main application background */
.stApp {
    background-color: #f5f7fb;
    color: #334155;
}

/* Main page title */
.stApp h1 {
    color: #172554 !important;
    font-weight: 800 !important;
    letter-spacing: -0.8px;
}

/* All section headings */
.stApp h2,
.stApp h3,
.stApp h4,
.stApp h5,
.stApp h6 {
    color: #1e3a5f !important;
    font-weight: 700 !important;
}

/* Markdown headings */
.stApp [data-testid="stMarkdownContainer"] h1,
.stApp [data-testid="stMarkdownContainer"] h2,
.stApp [data-testid="stMarkdownContainer"] h3,
.stApp [data-testid="stMarkdownContainer"] h4 {
    color: #1e3a5f !important;
}

/* Paragraphs and normal text */
.stApp p,
.stApp li,
.stApp label {
    color: #334155;
}

/* KPI cards */
div[data-testid="stMetric"] {
    background-color: #ffffff;
    border: 1px solid #dbe4f0;
    padding: 20px 18px;
    border-radius: 14px;
    box-shadow: 0 3px 12px rgba(15, 23, 42, 0.05);
}

/* KPI labels */
div[data-testid="stMetricLabel"] {
    color: #475569 !important;
    font-weight: 600 !important;
}

/* KPI values */
div[data-testid="stMetricValue"] {
    color: #1d4ed8 !important;
    font-weight: 800 !important;
}

/* KPI delta text */
div[data-testid="stMetricDelta"] {
    color: #475569;
}

/* Sidebar */
section[data-testid="stSidebar"] {
    background-color: #eef2ff;
    border-right: 1px solid #dbeafe;
}

/* Sidebar headings */
section[data-testid="stSidebar"] h1,
section[data-testid="stSidebar"] h2,
section[data-testid="stSidebar"] h3 {
    color: #172554 !important;
    font-weight: 700 !important;
}

/* Sidebar text */
section[data-testid="stSidebar"] p,
section[data-testid="stSidebar"] label {
    color: #334155 !important;
}

/* Data tables */
div[data-testid="stDataFrame"] {
    border: 1px solid #dbe4f0;
    border-radius: 12px;
    overflow: hidden;
}

/* Download buttons */
.stDownloadButton button {
    background-color: #1d4ed8 !important;
    color: #ffffff !important;
    border: none !important;
    border-radius: 9px;
    padding: 10px 16px;
    font-weight: 600;
}

/* Button hover */
.stDownloadButton button:hover {
    background-color: #1e40af !important;
    color: #ffffff !important;
}

/* Standard buttons */
.stButton button {
    border-radius: 9px;
    font-weight: 600;
}

/* Dividers */
hr {
    border-color: #dbe4f0 !important;
}

/* Alerts */
div[data-testid="stAlert"] {
    border-radius: 10px;
}

/* Expanders */
div[data-testid="stExpander"] {
    background-color: #ffffff;
    border: 1px solid #dbe4f0;
    border-radius: 10px;
}

/* Links */
.stApp a {
    color: #1d4ed8 !important;
}

/* Page spacing */
.block-container {
    padding-top: 2rem;
    padding-bottom: 2rem;
}

</style>
""", unsafe_allow_html=True)

BASE_DIR = Path(__file__).resolve().parent

DATA_PATH = BASE_DIR / "data" / "cleaned_sales.csv"
RFM_PATH = BASE_DIR / "reports" / "rfm_customer_segments.csv"
CHURN_PATH = BASE_DIR / "reports" / "customer_churn_predictions.csv"


# ============================================================
# 2. DASHBOARD HEADER
# ============================================================

st.title("📊 AI Customer Intelligence Platform")

st.markdown(
    """
    ### Sales Analytics | Customer Intelligence | Machine Learning

    Explore sales performance, understand customer behaviour,
    analyse customer segments, and review estimated churn risk.
    """
)

st.caption(
    "An end-to-end analytics project built with Python, "
    "Pandas, Streamlit, Plotly, and Scikit-learn."
)

st.divider()


# ============================================================
# 3. LOAD DATA
# ============================================================

@st.cache_data
def load_sales_data(path, modified_time):
    data = pd.read_csv(path)

    data["InvoiceDate"] = pd.to_datetime(
        data["InvoiceDate"],
        errors="coerce",
    )

    numeric_columns = ["Quantity", "UnitPrice"]

    for column in numeric_columns:
        data[column] = pd.to_numeric(
            data[column],
            errors="coerce",
        )

    data["CustomerID"] = pd.to_numeric(
        data["CustomerID"],
        errors="coerce",
    )

    data["Revenue"] = data["Quantity"] * data["UnitPrice"]

    data = data.dropna(
        subset=[
            "InvoiceNo",
            "InvoiceDate",
            "CustomerID",
            "Quantity",
            "UnitPrice",
        ]
    ).copy()

    # Retain completed positive-quantity sales.
    data = data[
        (
            ~data["InvoiceNo"]
            .astype(str)
            .str.startswith("C")
        )
        & (data["Quantity"] > 0)
        & (data["UnitPrice"] > 0)
    ].copy()

    return data


@st.cache_data
def load_rfm_data(path, modified_time):
    return pd.read_csv(path)


@st.cache_data
def load_churn_data(path, modified_time):
    return pd.read_csv(path)


# Check required files.

if not DATA_PATH.exists():
    st.error(
        f"Sales dataset not found:\n\n{DATA_PATH}"
    )
    st.info(
        "Make sure cleaned_sales.csv exists inside the data folder."
    )
    st.stop()

if not RFM_PATH.exists():
    st.error(
        f"Customer segmentation dataset not found:\n\n{RFM_PATH}"
    )
    st.info(
        "Make sure rfm_customer_segments.csv exists inside reports."
    )
    st.stop()


try:
    df = load_sales_data(
        DATA_PATH,
        DATA_PATH.stat().st_mtime,
    )

    rfm = load_rfm_data(
        RFM_PATH,
        RFM_PATH.stat().st_mtime,
    )

except Exception as error:
    st.error(f"Unable to load project data: {error}")
    st.stop()


required_sales_columns = [
    "InvoiceNo",
    "InvoiceDate",
    "CustomerID",
    "Quantity",
    "UnitPrice",
    "Revenue",
]

missing_sales_columns = [
    column
    for column in required_sales_columns
    if column not in df.columns
]

if missing_sales_columns:
    st.error(
        f"Missing sales columns: {missing_sales_columns}"
    )
    st.stop()

required_rfm_columns = [
    "CustomerID",
    "Recency",
    "Frequency",
    "Monetary",
    "Customer_Segment",
]

missing_rfm_columns = [
    column
    for column in required_rfm_columns
    if column not in rfm.columns
]

if missing_rfm_columns:
    st.error(
        f"Missing customer segmentation columns: "
        f"{missing_rfm_columns}"
    )
    st.stop()

if df.empty:
    st.error("No valid sales transactions are available.")
    st.stop()


# ============================================================
# 4. SIDEBAR FILTERS
# ============================================================

st.sidebar.title("🎛️ Dashboard Controls")

st.sidebar.markdown("### 📅 Sales Date Range")

min_date = df["InvoiceDate"].min().date()
max_date = df["InvoiceDate"].max().date()

selected_dates = st.sidebar.date_input(
    "Select sales period",
    value=(min_date, max_date),
    min_value=min_date,
    max_value=max_date,
    format="YYYY-MM-DD",
)

if isinstance(selected_dates, (tuple, list)) and len(selected_dates) == 2:

    start_date, end_date = selected_dates

    if start_date > end_date:
        st.sidebar.error(
            "Start date must be before end date."
        )
        st.stop()

    filtered_df = df[
        df["InvoiceDate"].dt.date.between(
            start_date,
            end_date,
        )
    ].copy()

elif isinstance(selected_dates, (tuple, list)) and len(selected_dates) == 1:

    selected_date = selected_dates[0]

    filtered_df = df[
        df["InvoiceDate"].dt.date == selected_date
    ].copy()

else:
    filtered_df = df.copy()

st.sidebar.caption(
    f"Transactions available: {len(filtered_df):,}"
)

if filtered_df.empty:
    st.warning(
        "No transactions were found for this date range. "
        "Please select another period."
    )
    st.stop()


# ============================================================
# 5. BUSINESS OVERVIEW AND KPIs
# ============================================================

st.subheader("📌 Business Overview")

total_revenue = filtered_df["Revenue"].sum()

total_orders = filtered_df["InvoiceNo"].nunique()

total_customers = filtered_df["CustomerID"].nunique()

if "StockCode" in filtered_df.columns:
    total_products = filtered_df["StockCode"].nunique()
else:
    total_products = filtered_df["Description"].nunique()

average_order_value = (
    total_revenue / total_orders
    if total_orders > 0
    else 0
)

col1, col2, col3, col4 = st.columns(4)

col1.metric(
    "💰 Total Revenue",
    f"£{total_revenue:,.2f}",
)

col2.metric(
    "🛒 Total Orders",
    f"{total_orders:,}",
)

col3.metric(
    "👥 Unique Customers",
    f"{total_customers:,}",
)

col4.metric(
    "📦 Unique Products",
    f"{total_products:,}",
)

col5, col6 = st.columns(2)

col5.metric(
    "🧾 Average Order Value",
    f"£{average_order_value:,.2f}",
)

col6.metric(
    "📋 Transaction Lines",
    f"{len(filtered_df):,}",
)

st.caption(
    "Revenue is calculated from positive-quantity purchase lines. "
    "Returns are excluded rather than matched back to original sales."
)

st.divider()


# ============================================================
# 6. MONTHLY REVENUE TRENDS
# ============================================================

st.subheader("📈 Monthly Revenue Trends")

monthly_sales = filtered_df.copy()

monthly_sales["Month"] = (
    monthly_sales["InvoiceDate"]
    .dt.to_period("M")
    .astype(str)
)

monthly_sales = (
    monthly_sales.groupby("Month", as_index=False)
    .agg(Revenue=("Revenue", "sum"))
    .sort_values("Month")
)

if not monthly_sales.empty:

    monthly_fig = px.line(
        monthly_sales,
        x="Month",
        y="Revenue",
        markers=True,
        title="Monthly Revenue",
        labels={
            "Month": "Month",
            "Revenue": "Revenue (£)",
        },
        template="plotly_white",
    )

    monthly_fig.update_layout(
        xaxis_title="Month",
        yaxis_title="Revenue (£)",
        hovermode="x unified",
    )

    st.plotly_chart(
        monthly_fig,
        use_container_width=True,
    )

    best_month = monthly_sales.loc[
        monthly_sales["Revenue"].idxmax()
    ]

    st.info(
        f"Highest-revenue month in the selected period: "
        f"**{best_month['Month']}** "
        f"(£{best_month['Revenue']:,.2f})."
    )

else:
    st.info("Monthly revenue information is unavailable.")

st.divider()


# ============================================================
# 7. TOP-SELLING PRODUCTS
# ============================================================

st.subheader("🏆 Top 10 Best-Selling Products")

if "Description" in filtered_df.columns:

    top_products = (
        filtered_df.groupby("Description", as_index=False)
        .agg(
            Total_Quantity=("Quantity", "sum"),
            Total_Revenue=("Revenue", "sum"),
        )
        .sort_values(
            "Total_Quantity",
            ascending=False,
        )
        .head(10)
    )

    if not top_products.empty:

        product_fig = px.bar(
            top_products.sort_values("Total_Quantity"),
            x="Total_Quantity",
            y="Description",
            orientation="h",
            title="Top Products by Quantity Sold",
            labels={
                "Total_Quantity": "Quantity Sold",
                "Description": "Product",
            },
            template="plotly_white",
        )

        product_fig.update_layout(
            yaxis_title="Product",
            xaxis_title="Quantity Sold",
        )

        st.plotly_chart(
            product_fig,
            use_container_width=True,
        )

        with st.expander("View Top Products Table"):
            st.dataframe(
                top_products,
                use_container_width=True,
                hide_index=True,
            )

    else:
        st.info("No product sales are available.")

else:
    st.info("Product descriptions are not available.")

st.divider()


# ============================================================
# 8. CUSTOMER SEGMENTATION
# ============================================================

st.subheader("🎯 Customer Segmentation")

rfm["Customer_Segment"] = (
    rfm["Customer_Segment"]
    .fillna("Unclassified")
)

segment_counts = (
    rfm["Customer_Segment"]
    .value_counts()
    .rename_axis("Customer Segment")
    .reset_index(name="Customer Count")
)

if not segment_counts.empty:

    segment_col1, segment_col2 = st.columns(2)

    with segment_col1:

        segment_pie = px.pie(
            segment_counts,
            names="Customer Segment",
            values="Customer Count",
            hole=0.45,
            title="Customer Distribution by Segment",
            template="plotly_white",
        )

        st.plotly_chart(
            segment_pie,
            use_container_width=True,
        )

    with segment_col2:

        segment_bar = px.bar(
            segment_counts.sort_values("Customer Count"),
            x="Customer Count",
            y="Customer Segment",
            orientation="h",
            title="Customers in Each Segment",
            labels={
                "Customer Count": "Number of Customers",
                "Customer Segment": "Segment",
            },
            template="plotly_white",
        )

        st.plotly_chart(
            segment_bar,
            use_container_width=True,
        )

    st.dataframe(
        segment_counts,
        use_container_width=True,
        hide_index=True,
    )

else:
    st.info("Customer segment information is unavailable.")

st.divider()


# ============================================================
# 9. CUSTOMER INTELLIGENCE
# ============================================================

st.subheader("👥 Customer Intelligence")

rfm_display = rfm.copy()

rfm_display["CustomerID"] = (
    rfm_display["CustomerID"].astype(str)
)

rfm_display["Monetary"] = pd.to_numeric(
    rfm_display["Monetary"],
    errors="coerce",
)

rfm_display = rfm_display.sort_values(
    "Monetary",
    ascending=False,
)

st.markdown(
    "Customers ranked by their historical monetary value."
)

st.dataframe(
    rfm_display.head(20),
    use_container_width=True,
    hide_index=True,
)

st.divider()


# ============================================================
# 10. CUSTOMER CHURN PREDICTION
# ============================================================

st.subheader("🔮 Customer Churn Prediction")

st.markdown(
    """
    This section displays customer churn risk scores generated
    by the separate Random Forest machine learning model.
    """
)

if CHURN_PATH.exists():

    try:

        churn_df = load_churn_data(
            CHURN_PATH,
            CHURN_PATH.stat().st_mtime,
        )

        required_churn_columns = [
            "CustomerID",
            "Recency",
            "Frequency",
            "Monetary",
            "Churn_Probability",
            "Churn_Risk",
        ]

        missing_churn_columns = [
            column
            for column in required_churn_columns
            if column not in churn_df.columns
        ]

        if missing_churn_columns:

            st.error(
                "The churn report is missing these columns: "
                f"{missing_churn_columns}"
            )

        else:

            churn_df["Churn_Probability"] = pd.to_numeric(
                churn_df["Churn_Probability"],
                errors="coerce",
            )

            churn_df["Monetary"] = pd.to_numeric(
                churn_df["Monetary"],
                errors="coerce",
            )

            churn_df = churn_df.dropna(
                subset=[
                    "Churn_Probability",
                    "Churn_Risk",
                ]
            ).copy()

            churn_df["Churn_Risk"] = (
                churn_df["Churn_Risk"]
                .astype(str)
                .str.title()
            )

            churn_df["Churn_Probability_Percent"] = (
                churn_df["Churn_Probability"] * 100
            )

            # ----------------------------------------------
            # Churn KPIs
            # ----------------------------------------------

            st.markdown("### 📊 Churn Risk Overview")

            high_risk_count = (
                churn_df["Churn_Risk"] == "High"
            ).sum()

            medium_risk_count = (
                churn_df["Churn_Risk"] == "Medium"
            ).sum()

            low_risk_count = (
                churn_df["Churn_Risk"] == "Low"
            ).sum()

            risk_col1, risk_col2, risk_col3 = st.columns(3)

            risk_col1.metric(
                "🔴 High-Risk Customers",
                f"{high_risk_count:,}",
            )

            risk_col2.metric(
                "🟠 Medium-Risk Customers",
                f"{medium_risk_count:,}",
            )

            risk_col3.metric(
                "🟢 Low-Risk Customers",
                f"{low_risk_count:,}",
            )

            # ----------------------------------------------
            # Risk Distribution
            # ----------------------------------------------

            risk_counts = (
                churn_df["Churn_Risk"]
                .value_counts()
                .reindex(
                    ["High", "Medium", "Low"],
                    fill_value=0,
                )
                .rename_axis("Risk Level")
                .reset_index(name="Customer Count")
            )

            chart_col1, chart_col2 = st.columns(2)

            with chart_col1:

                risk_pie = px.pie(
                    risk_counts,
                    names="Risk Level",
                    values="Customer Count",
                    hole=0.45,
                    title="Customer Churn Risk Distribution",
                    template="plotly_white",
                    category_orders={
                        "Risk Level": [
                            "High",
                            "Medium",
                            "Low",
                        ]
                    },
                )

                st.plotly_chart(
                    risk_pie,
                    use_container_width=True,
                )

            with chart_col2:

                risk_bar = px.bar(
                    risk_counts,
                    x="Risk Level",
                    y="Customer Count",
                    title="Customers by Risk Level",
                    labels={
                        "Risk Level": "Risk Category",
                        "Customer Count": "Customers",
                    },
                    template="plotly_white",
                    category_orders={
                        "Risk Level": [
                            "High",
                            "Medium",
                            "Low",
                        ]
                    },
                )

                st.plotly_chart(
                    risk_bar,
                    use_container_width=True,
                )

            # ----------------------------------------------
            # Risk Probability Distribution
            # ----------------------------------------------

            st.markdown("### 📈 Churn Probability Distribution")

            probability_fig = px.histogram(
                churn_df,
                x="Churn_Probability_Percent",
                nbins=20,
                title="Distribution of Estimated Churn Probabilities",
                labels={
                    "Churn_Probability_Percent":
                        "Estimated Churn Probability (%)"
                },
                template="plotly_white",
            )

            st.plotly_chart(
                probability_fig,
                use_container_width=True,
            )

            # ----------------------------------------------
            # High-Risk Customers
            # ----------------------------------------------

            st.markdown(
                "### 🚨 Customers Requiring Attention"
            )

            high_risk = churn_df[
                churn_df["Churn_Risk"] == "High"
            ].sort_values(
                "Churn_Probability",
                ascending=False,
            )

            display_columns = [
                "CustomerID",
                "Recency",
                "Frequency",
                "Monetary",
                "Churn_Probability_Percent",
                "Churn_Risk",
            ]

            available_columns = [
                column
                for column in display_columns
                if column in high_risk.columns
            ]

            if not high_risk.empty:

                st.dataframe(
                    high_risk[available_columns].head(20),
                    use_container_width=True,
                    hide_index=True,
                    column_config={
                        "Churn_Probability_Percent": st.column_config.NumberColumn(
                            "Estimated Churn Probability (%)",
                            format="%.2f%%",
                        ),
                        "Monetary": st.column_config.NumberColumn(
                            "Historical Spending (£)",
                            format="£%.2f",
                        ),
                    },
                )

                st.caption(
                    "These customers have the highest estimated "
                    "churn probabilities according to the model."
                )

            else:

                st.success(
                    "No customers are classified as high risk "
                    "using the current thresholds."
                )

            # ----------------------------------------------
            # Search Customers
            # ----------------------------------------------

            st.markdown("### 🔎 Search Customer Risk")

            search_id = st.text_input(
                "Enter a Customer ID",
                placeholder="Example: 17850",
            )

            if search_id.strip():

                customer_ids = (
                    churn_df["CustomerID"]
                    .astype(str)
                    .str.replace(r"\.0$", "", regex=True)
                )

                matching_customers = churn_df[
                    customer_ids == search_id.strip()
                ]

                if not matching_customers.empty:

                    st.dataframe(
                        matching_customers[
                            [
                                column
                                for column in [
                                    "CustomerID",
                                    "Recency",
                                    "Frequency",
                                    "Monetary",
                                    "Churn_Probability_Percent",
                                    "Churn_Risk",
                                    "Actual_Churn_Label",
                                ]
                                if column in matching_customers.columns
                            ]
                        ],
                        use_container_width=True,
                        hide_index=True,
                    )

                else:

                    st.info(
                        "Customer ID not found in the churn report."
                    )

            # ----------------------------------------------
            # Download Churn Predictions
            # ----------------------------------------------

            st.markdown(
                "### 📥 Download Customer Churn Report"
            )

            st.download_button(
                label="Download Churn Predictions CSV",
                data=churn_df.to_csv(
                    index=False
                ).encode("utf-8"),
                file_name="customer_churn_predictions.csv",
                mime="text/csv",
                key="download_churn_predictions",
            )

            st.info(
                "Interpretation: churn risk is a model-generated "
                "estimate. In this project, the churn label is "
                "defined by purchase inactivity during a specified "
                "historical period. These scores are not verified "
                "predictions of future customer behaviour."
            )

    except Exception as error:

        st.error(
            f"Unable to display churn predictions: {error}"
        )

        st.info(
            "Try running churn_model.py again to regenerate "
            "the prediction report."
        )

else:

    st.warning(
        "The churn prediction report was not found."
    )

    st.code(
        "python churn_model.py",
        language="bash",
    )

    st.caption(
        "Run the churn model from your project directory, "
        "then refresh this dashboard."
    )

st.divider()


# ============================================================
# 11. DOWNLOAD SALES AND CUSTOMER REPORTS
# ============================================================

st.subheader("📥 Download Analytics Reports")

download_col1, download_col2 = st.columns(2)

with download_col1:

    st.download_button(
        label="Download Filtered Sales CSV",
        data=filtered_df.to_csv(
            index=False
        ).encode("utf-8"),
        file_name="filtered_sales.csv",
        mime="text/csv",
        key="download_filtered_sales",
    )

with download_col2:

    st.download_button(
        label="Download Customer Segmentation CSV",
        data=rfm.to_csv(
            index=False
        ).encode("utf-8"),
        file_name="rfm_customer_segments.csv",
        mime="text/csv",
        key="download_rfm",
    )


# ============================================================
# 12. FOOTER
# ============================================================

st.divider()

st.markdown(
    """
    <div style="text-align: center;">
        <h4>AI Customer Intelligence & Predictive Analytics Platform</h4>
        <p>
            Built with Python, Pandas, Streamlit, Plotly,
            and Scikit-learn.
        </p>
        <p>
            Sales Analytics • Customer Segmentation • Churn Analysis
        </p>
    </div>
    """,
    unsafe_allow_html=True,
)