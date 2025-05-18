import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.subplots as sp
from plotly.graph_objs import Bar
from sqlalchemy import create_engine
from datetime import datetime

# --- Page Setup ---
st.set_page_config(page_title="📊 Supply Chain KPI Dashboard", layout="wide")

# --- DB Connection ---
engine = create_engine("mysql+pymysql://root:20051030@localhost/IFB107")

# --- Sidebar Navigation ---
st.sidebar.title("📁 IFB107TC")
section = st.sidebar.radio("Go to", [
    "KPI Dashboard", "Advanced Insights", "Outlier Detection", "Interactive Explorer", "AI Anomaly Detection"
])

# --- Time Filter ---
st.sidebar.markdown("---")
st.sidebar.markdown("### 📅 Date Filter (Optional)")
start_date = st.sidebar.date_input("Start Date", datetime(2023, 1, 1))
end_date = st.sidebar.date_input("End Date", datetime(2025, 12, 31))

## 原始查询
df = pd.read_sql("""
    SELECT o.OrderID, o.Supplier, o.ProductCategory, o.OrderQuantity,
           f.CustomerRating, f.IsDamaged, f.WeatherCondition,
           s.ShippingTime, s.ShippingCost
    FROM `Order` o
    JOIN OrderFeedback f ON o.OrderID = f.OrderID
    JOIN ShippingInfo s ON o.OrderID = s.OrderID
""", engine)

df = df.drop_duplicates(subset="OrderID")

if section == "KPI Dashboard":
    st.title("📦 E-Commerce Supply Chain KPI Dashboard")
    st.markdown("Analyze key performance metrics across suppliers, customers, and orders.")

    try:
        df = pd.read_sql("""
            SELECT o.OrderID, o.Supplier, o.ProductCategory, o.OrderQuantity,
                   f.CustomerRating, f.IsDamaged,
                   s.ShippingTime, s.ShippingCost
            FROM `Order` o
            JOIN OrderFeedback f ON o.OrderID = f.OrderID
            JOIN ShippingInfo s ON o.OrderID = s.OrderID
        """, engine)

        # 去除重复订单（确保每个订单只被统计一次）
        df = df.drop_duplicates(subset='OrderID')

        # 核心指标计算
        avg_rating = round(df["CustomerRating"].mean(), 2)
        total_orders = df.shape[0]
        damage_rate = round(df["IsDamaged"].sum() / total_orders * 100, 2)
        avg_ship_time = round(df["ShippingTime"].mean(), 1)
        avg_ship_cost = round(df["ShippingCost"].mean(), 2)
        cost_per_unit = round((df["ShippingCost"] / df["OrderQuantity"]).mean(), 2)
        rating_std = round(df["CustomerRating"].std(), 2)

        # 可视化展示
        col1, col2, col3 = st.columns(3)
        col1.metric("⭐ Avg Rating", avg_rating)
        col2.metric("📦 Total Orders", total_orders)
        col3.metric("💥 Damage Rate", f"{damage_rate} %")

        col4, col5, col6 = st.columns(3)
        col4.metric("⏱️ Avg Shipping Time", f"{avg_ship_time} days")
        col5.metric("💰 Avg Shipping Cost", f"${avg_ship_cost}")
        col6.metric("📦 Cost per Unit", f"${cost_per_unit}")

        st.metric("📉 Rating Std Dev", rating_std)

    except Exception as e:
        st.error(f"❌ Unable to load KPI data: {e}")


# --- Advanced Insights ---
elif section == "Advanced Insights":
    st.title("📈 Strategic Insights from Query Results")

    # 🚚 折叠：供应商运输性能
    with st.expander("🚚 Supplier Shipping Performance"):
        supplier_df = pd.read_sql("""
            SELECT Supplier, 
                   ROUND(AVG(ShippingTime), 2) AS AvgShippingTime,
                   ROUND(AVG(ShippingCost), 2) AS AvgShippingCost
            FROM ShippingInfo GROUP BY Supplier;
        """, engine)

        fig1 = px.bar(supplier_df, x="Supplier", y=["AvgShippingTime", "AvgShippingCost"],
                      barmode="group", title="Average Shipping Time and Cost by Supplier",
                      color_discrete_sequence=px.colors.qualitative.Pastel)
        st.plotly_chart(fig1, use_container_width=True)

    # 📦 折叠：产品类别评分 vs 损坏率
    with st.expander("📦 Product Category: Rating vs Damage Rate"):
        rating_damage_df = pd.read_sql("""
            SELECT o.ProductCategory,
                   ROUND(AVG(f.CustomerRating), 2) AS AvgRating,
                   ROUND(SUM(f.IsDamaged)/COUNT(*) * 100, 2) AS DamageRate
            FROM `Order` o
            JOIN OrderFeedback f ON o.OrderID = f.OrderID
            GROUP BY o.ProductCategory;
        """, engine)

        fig2 = px.line_polar(rating_damage_df, r='DamageRate', theta='ProductCategory', line_close=True,
                             title="Damage Rate by Product Category",
                             color_discrete_sequence=px.colors.qualitative.Pastel)
        fig2.add_trace(px.line_polar(rating_damage_df, r='AvgRating', theta='ProductCategory',
                                     line_close=True,
                                     color_discrete_sequence=px.colors.qualitative.Pastel).data[0])
        st.plotly_chart(fig2, use_container_width=True)

    # 🚨 折叠：高运输成本订单（排名 + 图）
    with st.expander("🚨 Top Shipping Cost Orders (Per Supplier)"):
        rank_df = pd.read_sql("""
            SELECT OrderID, Supplier, ShippingCost,
                   RANK() OVER (PARTITION BY Supplier ORDER BY ShippingCost DESC) AS CostRank
            FROM ShippingInfo;
        """, engine)
        top_orders = rank_df[rank_df["CostRank"] <= 2]

        st.dataframe(top_orders.style.highlight_max(axis=0, color='orange'))

        fig3 = px.bar(top_orders, x="OrderID", y="ShippingCost", color="Supplier",
                      title="Top 2 Shipping Cost Orders by Supplier",
                      color_discrete_sequence=px.colors.qualitative.Pastel)
        st.plotly_chart(fig3, use_container_width=True)


#elif section == "AI Anomaly Detection":
elif section == "AI Anomaly Detection":
    from sklearn.ensemble import IsolationForest
    import numpy as np

    st.title("🧠 AI-Based Outlier Detection using Isolation Forest")

    # 👉 设置 contamination 滑块
    contamination = st.sidebar.slider(
        "Anomaly Detection Sensitivity (Contamination Ratio)",
        min_value=0.01, max_value=0.20, value=0.03, step=0.01,
        help="Defines the percentage of points to detect as outliers. Higher values detect more outliers."
    )

    # 读取数据
    df = pd.read_sql("SELECT OrderID, ShippingTime, ShippingCost FROM ShippingInfo", engine)
    df = df.dropna()

    # 模型训练
    model = IsolationForest(contamination=contamination, random_state=42)
    df["Anomaly"] = model.fit_predict(df[["ShippingTime", "ShippingCost"]])
    df["AnomalyScore"] = model.decision_function(df[["ShippingTime", "ShippingCost"]])

    # 区分正常/异常
    normal = df[df["Anomaly"] == 1]
    outliers = df[df["Anomaly"] == -1]

    st.markdown(f"🔍 **Detected `{len(outliers)}` outliers** out of `{len(df)}` records.")
    st.markdown(f"🧪 Model sensitivity set to **{int(contamination * 100)}%** anomaly threshold.")

    # --- 可视化图表 ---
    st.subheader("📊 Anomaly Scatter Plot")
    fig = px.scatter(df, x="ShippingTime", y="ShippingCost",
                     color=df["Anomaly"].map({1: "Normal", -1: "Outlier"}),
                     color_discrete_map={"Normal": "green", "Outlier": "red"},
                     hover_data=["OrderID", "AnomalyScore"],
                     title="Shipping Time vs Cost with AI Anomaly Detection")
    st.plotly_chart(fig, use_container_width=True)

    # --- 异常表格 ---
    with st.expander("📋 Outlier Table (Detected Abnormal Orders)"):
        st.dataframe(outliers)
        st.download_button("⬇️ Download Anomalies CSV", outliers.to_csv(index=False),
                           file_name="shipping_anomalies.csv", mime="text/csv")

elif section == "Outlier Detection":
    st.title("🚨 Outlier Detection: High Shipping Costs")
    df = pd.read_sql("SELECT * FROM ShippingInfo", engine)
    Q1 = df["ShippingCost"].quantile(0.25)
    Q3 = df["ShippingCost"].quantile(0.75)
    IQR = Q3 - Q1
    outliers = df[df["ShippingCost"] > (Q3 + 1.5 * IQR)]

    st.write(f"Threshold for outliers: > {round(Q3 + 1.5 * IQR, 2)}")
    if not outliers.empty:
        st.dataframe(outliers)
        st.download_button("⬇️ Download Outliers CSV", outliers.to_csv(index=False),
                           file_name="shipping_outliers.csv", mime="text/csv")
    else:
        st.success("✅ No significant shipping cost outliers detected.")

# --- Interactive Explorer ---
elif section == "Interactive Explorer":
    st.title("📊 Custom Query Explorer")

    queries = {
        "Average Rating by Supplier": """
            SELECT o.Supplier, ROUND(AVG(f.CustomerRating), 2) AS AvgRating
            FROM `Order` o JOIN OrderFeedback f ON o.OrderID = f.OrderID
            GROUP BY o.Supplier ORDER BY AvgRating DESC;
        """,
        "Damage Rate by Weather Condition": """
            SELECT f.WeatherCondition,
                   COUNT(*) AS TotalOrders,
                   SUM(CASE WHEN f.IsDamaged = 1 THEN 1 ELSE 0 END) AS DamagedOrders,
                   ROUND(SUM(CASE WHEN f.IsDamaged = 1 THEN 1 ELSE 0 END) / COUNT(*) * 100, 2) AS DamageRatePercent
            FROM OrderFeedback f GROUP BY f.WeatherCondition ORDER BY DamageRatePercent DESC;
        """,
        "Shipping Cost by Product Category": """
            SELECT o.ProductCategory, ROUND(AVG(s.ShippingCost), 2) AS AvgShippingCost
            FROM `Order` o JOIN ShippingInfo s ON o.OrderID = s.OrderID
            GROUP BY o.ProductCategory ORDER BY AvgShippingCost DESC;
        """,
        "Order Count by Supplier": """
            SELECT Supplier, COUNT(*) AS OrderCount FROM `Order`
            GROUP BY Supplier ORDER BY OrderCount DESC;
        """,
        "Shipping Time vs Cost": """
            SELECT ShippingTime, ShippingCost FROM ShippingInfo;
        """
    }

    st.sidebar.header("🔍 Explorer Settings")
    selected_query = st.sidebar.selectbox("Select Analysis", list(queries.keys()))
    chart_type = st.sidebar.radio("Chart Type", ["Bar", "Line", "Scatter", "Multi-Compare"])

    df = pd.read_sql(queries[selected_query], engine)
    if df.empty:
        st.warning("⚠️ No data returned.")
        st.stop()

    # Supplier Filter（增强交互灵活性）
    if 'Supplier' in df.columns:
        options = df['Supplier'].unique().tolist()
        selected = st.sidebar.multiselect("Filter Supplier", options, default=options)
        df = df[df['Supplier'].isin(selected)]

    # --- 表格展示模块 ---
    with st.expander("📊 Query Results Table"):
        page_size = 10
        page = st.number_input("Page", 1, max(1, (len(df) - 1) // page_size + 1), 1)
        start = (page - 1) * page_size
        st.dataframe(df.iloc[start:start + page_size])
        st.download_button("⬇️ Download CSV", df.to_csv(index=False), file_name="output.csv", mime="text/csv")

    # --- 数据统计模块 ---
    with st.expander("📈 Statistical Summary"):
        st.dataframe(df.describe())

    # --- 可视化模块 ---
    with st.expander("📉 Interactive Visualization"):
        categoricals = df.select_dtypes(include="object").columns.tolist()
        numerics = df.select_dtypes(include="number").columns.tolist()

        if "ShippingTime" in df.columns and "ShippingCost" in df.columns and chart_type != "Multi-Compare":
            fig = px.scatter(df, x="ShippingTime", y="ShippingCost",
                             title="Shipping Time vs Cost",
                             color_discrete_sequence=px.colors.qualitative.Pastel,
                             hover_data=df.columns)
            st.plotly_chart(fig, use_container_width=True)
            st.stop()

        x_axis = st.selectbox("X-axis", options=categoricals + numerics)
        y_axis = st.selectbox("Y-axis", options=[col for col in numerics if col != x_axis])

        if chart_type == "Multi-Compare":
            st.markdown("### 🔍 Multi-Compare Subplots")
            features = [col for col in numerics if col != x_axis]
            fig = sp.make_subplots(rows=len(features), cols=1, shared_xaxes=True,
                                   subplot_titles=[f"{col} by {x_axis}" for col in features])
            for i, col in enumerate(features):
                fig.add_trace(Bar(x=df[x_axis], y=df[col], name=col, hovertext=df[col]), row=i+1, col=1)
            fig.update_layout(height=300 * len(features), showlegend=False)
            st.plotly_chart(fig, use_container_width=True)
            st.stop()

        if chart_type == "Bar":
            fig = px.bar(df, x=x_axis, y=y_axis,
                         text_auto=".2s",
                         hover_data=df.columns,
                         title=f"{y_axis} by {x_axis}",
                         color_discrete_sequence=px.colors.qualitative.Pastel)
        elif chart_type == "Line":
            fig = px.line(df, x=x_axis, y=y_axis, markers=True,
                          hover_data=df.columns,
                          title=f"{y_axis} by {x_axis}",
                          color_discrete_sequence=px.colors.qualitative.Pastel)
        elif chart_type == "Scatter":
            fig = px.scatter(df, x=x_axis, y=y_axis, size_max=15,
                             hover_data=df.columns,
                             title=f"{y_axis} by {x_axis}",
                             color_discrete_sequence=px.colors.qualitative.Pastel)

        fig.update_layout(xaxis_title=x_axis, yaxis_title=y_axis)
        st.plotly_chart(fig, use_container_width=True)

