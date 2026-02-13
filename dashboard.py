"""
IDCW Fund Dashboard
===================
Visualizes mutual fund data from pre-generated CSV files.
This script only loads and displays data - it does NOT fetch or generate data.

Required files in data/ folder:
- idcw_fy_output.csv
- final_schemes_master.csv
- trend_kpis.csv
"""

import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import numpy as np
import os

# ========================
# CONFIGURATION
# ========================
DATA_DIR = "data"

st.set_page_config(
    page_title="IDCW Fund Analyzer",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ========================
# LOAD DATA
# ========================
@st.cache_data
def load_data():
    """Load all CSV files from data directory"""
    
    # Check if data directory exists
    if not os.path.exists(DATA_DIR):
        st.error(f"❌ Data directory '{DATA_DIR}/' not found!")
        st.info("Please run `python data_generator.py` first to generate the data files.")
        st.stop()
    
    # Define required files
    files = {
        "fy_data": "idcw_fy_output.csv",
        "scheme_data": "final_schemes_master.csv",
        "trend_kpis": "trend_kpis.csv"
    }
    
    # Check all files exist
    missing_files = []
    for key, filename in files.items():
        filepath = os.path.join(DATA_DIR, filename)
        if not os.path.exists(filepath):
            missing_files.append(filename)
    
    if missing_files:
        st.error(f"❌ Missing required files in '{DATA_DIR}/' folder:")
        for f in missing_files:
            st.write(f"  - {f}")
        st.info("Please run `python data_generator.py` first to generate these files.")
        st.stop()
    
    # Load the data
    try:
        fy_data = pd.read_csv(os.path.join(DATA_DIR, files["fy_data"]))
        scheme_data = pd.read_csv(os.path.join(DATA_DIR, files["scheme_data"]))
        trend_kpis = pd.read_csv(os.path.join(DATA_DIR, files["trend_kpis"]))
        
        # Clean data types
        fy_data["FY"] = fy_data["FY"].astype(str)
        fy_data["IDCW_Yield_pct"] = pd.to_numeric(fy_data["IDCW_Yield_pct"], errors="coerce")
        
        # Merge FY data with scheme attributes and trend KPIs
        merged = fy_data.merge(scheme_data, on="yahoo_symbol", how="left")
        
        return merged, scheme_data, trend_kpis
        
    except Exception as e:
        st.error(f"❌ Error loading data: {e}")
        st.stop()

# Load data
df, scheme_master, trend_kpis = load_data()

# Restrict to IDCW funds only
idcw_df = df[df["Option"] == "IDCW"].copy()

# ========================
# BUILD RANKINGS
# ========================

# Latest FY ranking
latest_fy = (
    idcw_df.groupby("yahoo_symbol")["FY"]
    .max()
    .reset_index(name="Latest_FY")
)

latest = idcw_df.merge(
    latest_fy,
    left_on=["yahoo_symbol", "FY"],
    right_on=["yahoo_symbol", "Latest_FY"],
    how="inner"
)

latest_rank = (
    latest[[
        "fund_name",
        "yahoo_symbol",
        "FY",
        "IDCW_Yield_pct",
        "AMC",
        "Plan",
        "Option"
    ]]
    .dropna()
    .sort_values("IDCW_Yield_pct", ascending=False)
    .reset_index(drop=True)
)

latest_rank["Latest_IDCW_Rank"] = latest_rank.index + 1

# Overall ranking (3+ years)
overall_rank = (
    idcw_df.groupby(["fund_name", "yahoo_symbol", "AMC", "Plan", "Option"])
    .agg(
        Avg_IDCW_Yield_pct=("IDCW_Yield_pct", "mean"),
        Total_Dividend=("Dividend", "sum"),
        Years=("FY", "nunique")
    )
    .reset_index()
)

overall_rank = overall_rank[overall_rank["Years"] >= 3]
overall_rank = overall_rank.sort_values(
    "Avg_IDCW_Yield_pct",
    ascending=False
).reset_index(drop=True)

overall_rank["Overall_IDCW_Rank"] = overall_rank.index + 1

# Merge with trend KPIs
ranking = latest_rank.merge(
    overall_rank,
    on=["fund_name", "yahoo_symbol", "AMC", "Plan", "Option"],
    how="left"
)

ranking = ranking.merge(
    trend_kpis,
    on=["yahoo_symbol", "fund_name"],
    how="left"
)

# Enhanced composite score with trend
ranking["Composite_Score"] = (
    0.4 * ranking["IDCW_Yield_pct"] +
    0.3 * ranking["Avg_IDCW_Yield_pct"] +
    0.2 * ranking["Trend_Slope"].fillna(0) +
    0.1 * ranking["Max_Consecutive_Increases"].fillna(0)
)

ranking = ranking.sort_values(
    "Composite_Score",
    ascending=False
).reset_index(drop=True)

ranking["Composite_Rank"] = ranking.index + 1

# ========================
# SIDEBAR FILTERS
# ========================
st.sidebar.title("🔎 Filters")

# Data freshness info
st.sidebar.info(f"📅 Data loaded from: `{DATA_DIR}/`")

if st.sidebar.button("🔄 Refresh Data"):
    st.cache_data.clear()
    st.rerun()

st.sidebar.divider()

min_years = st.sidebar.slider(
    "Minimum IDCW history (years)",
    1, 10, 3
)

# Trend filters
st.sidebar.subheader("📈 Trend Filters")

min_cagr = st.sidebar.number_input(
    "Min CAGR (%)",
    value=-100.0,
    step=1.0,
    help="Minimum compound annual growth rate of dividend yield"
)

min_consecutive = st.sidebar.number_input(
    "Min consecutive increases",
    min_value=0,
    max_value=10,
    value=0,
    help="Minimum consecutive years of yield increase"
)

positive_trend_only = st.sidebar.checkbox(
    "Positive trend only (slope > 0)",
    help="Show only funds with upward yield trend"
)

st.sidebar.divider()

# AMC and Plan filters
selected_amc = st.sidebar.multiselect(
    "AMC",
    sorted(ranking["AMC"].dropna().unique())
)

selected_plan = st.sidebar.multiselect(
    "Plan",
    sorted(ranking["Plan"].dropna().unique())
)

selected_option = st.sidebar.multiselect(
    "Option",
    sorted(ranking["Option"].dropna().unique()),
    default=["IDCW"]
)

selected_scheme = st.sidebar.multiselect(
    "Scheme",
    sorted(ranking["fund_name"].unique())
)

st.sidebar.divider()

ranking_view = st.sidebar.radio(
    "Ranking Type",
    [
        "Composite (with Trend)",
        "Latest FY Yield",
        "Overall Avg Yield",
        "Best CAGR",
        "Most Consistent"
    ]
)

# ========================
# APPLY FILTERS
# ========================
filtered = ranking[ranking["Years"] >= min_years]

# Apply trend filters
if min_cagr > -100:
    filtered = filtered[filtered["CAGR_Yield"].fillna(-999) >= min_cagr]

if min_consecutive > 0:
    filtered = filtered[filtered["Max_Consecutive_Increases"].fillna(0) >= min_consecutive]

if positive_trend_only:
    filtered = filtered[filtered["Trend_Slope"].fillna(-999) > 0]

if selected_amc:
    filtered = filtered[filtered["AMC"].isin(selected_amc)]

if selected_plan:
    filtered = filtered[filtered["Plan"].isin(selected_plan)]

if selected_option:
    filtered = filtered[filtered["Option"].isin(selected_option)]

if selected_scheme:
    filtered = filtered[filtered["fund_name"].isin(selected_scheme)]

# ========================
# MAIN DASHBOARD
# ========================

# Header
st.title("🏆 IDCW Mutual Fund Rankings")
st.markdown("*Analyze dividend yield trends across Indian mutual funds*")

# Summary metrics
col1, col2, col3, col4, col5 = st.columns(5)

with col1:
    st.metric("Total Funds", len(filtered))

with col2:
    if len(filtered) > 0:
        avg_yield = filtered['IDCW_Yield_pct'].mean()
        st.metric("Avg Latest Yield", f"{avg_yield:.2f}%")
    else:
        st.metric("Avg Latest Yield", "N/A")

with col3:
    if len(filtered) > 0:
        avg_cagr = filtered['CAGR_Yield'].mean()
        st.metric("Avg CAGR", f"{avg_cagr:.2f}%" if pd.notna(avg_cagr) else "N/A")
    else:
        st.metric("Avg CAGR", "N/A")

with col4:
    if len(filtered) > 0:
        positive_trend = (filtered['Trend_Slope'] > 0).sum()
        st.metric("Positive Trend", f"{positive_trend}/{len(filtered)}")
    else:
        st.metric("Positive Trend", "0/0")

with col5:
    if len(filtered) > 0:
        avg_consecutive = filtered['Max_Consecutive_Increases'].mean()
        st.metric("Avg Consecutive ↑", f"{avg_consecutive:.1f}" if pd.notna(avg_consecutive) else "N/A")
    else:
        st.metric("Avg Consecutive ↑", "N/A")

st.divider()

# ========================
# RANKINGS TABLE
# ========================

# Select view
if ranking_view == "Composite (with Trend)":
    show = filtered.sort_values("Composite_Rank")
    sort_col = "Composite_Rank"
elif ranking_view == "Latest FY Yield":
    show = filtered.sort_values("Latest_IDCW_Rank")
    sort_col = "Latest_IDCW_Rank"
elif ranking_view == "Overall Avg Yield":
    show = filtered.sort_values("Overall_IDCW_Rank")
    sort_col = "Overall_IDCW_Rank"
elif ranking_view == "Best CAGR":
    show = filtered.sort_values("CAGR_Yield", ascending=False)
    sort_col = "CAGR_Yield"
else:  # Most Consistent
    show = filtered.sort_values("Consistency_CV")
    sort_col = "Consistency_CV"

# Add dividend trend sparkline
spark_data = (
    idcw_df
    .sort_values("FY")
    .groupby("fund_name")["IDCW_Yield_pct"]
    .apply(lambda x: x.fillna(0).tolist())
    .reset_index()
)

show = show.merge(spark_data, on="fund_name", how="left")
show.rename(columns={"IDCW_Yield_pct_y": "Yield_Trend"}, inplace=True)

if len(show) > 0:
    # Display table
    st.dataframe(
        show[[
            "Composite_Rank",
            "fund_name",
            "AMC",
            "Plan",
            "FY",
            "IDCW_Yield_pct_x",
            "Avg_IDCW_Yield_pct",
            "CAGR_Yield",
            "Trend_Slope",
            "Max_Consecutive_Increases",
            "Consistency_CV",
            "Years",
            "Yield_Trend"
        ]].rename(columns={
            "IDCW_Yield_pct_x": "Latest_Yield",
            "Composite_Rank": "Rank"
        }),
        column_config={
            "Rank": st.column_config.NumberColumn(
                "Rank",
                format="%d"
            ),
            "fund_name": "Fund Name",
            "AMC": "AMC",
            "Plan": "Plan",
            "FY": "Latest FY",
            "Latest_Yield": st.column_config.NumberColumn(
                "Latest Yield (%)",
                format="%.2f%%"
            ),
            "Avg_IDCW_Yield_pct": st.column_config.NumberColumn(
                "Avg Yield (%)",
                format="%.2f%%"
            ),
            "CAGR_Yield": st.column_config.NumberColumn(
                "CAGR (%)",
                help="Compound Annual Growth Rate of yield",
                format="%.2f%%"
            ),
            "Trend_Slope": st.column_config.NumberColumn(
                "Trend (% /yr)",
                help="Linear trend: % increase per year",
                format="%.3f"
            ),
            "Max_Consecutive_Increases": st.column_config.NumberColumn(
                "Max Consecutive ↑",
                help="Maximum consecutive years of yield increase"
            ),
            "Consistency_CV": st.column_config.NumberColumn(
                "Consistency",
                help="Coefficient of Variation (lower = more consistent)",
                format="%.1f"
            ),
            "Years": "History",
            "Yield_Trend": st.column_config.LineChartColumn(
                "Yield Trend",
                help="Dividend yield % across financial years",
                width="medium"
            )
        },
        hide_index=True,
        use_container_width=True,
        height=600
    )
    
    # Export button
    st.download_button(
        "📥 Download Filtered Rankings CSV",
        show.to_csv(index=False),
        "idcw_fund_rankings.csv",
        "text/csv",
        key="download-rankings"
    )
else:
    st.warning("⚠️ No funds match your current filters. Try adjusting the criteria.")

# ========================
# FUND DEEP DIVE
# ========================
st.divider()
st.subheader("🔍 Fund Deep Dive")

available_funds = sorted(filtered["fund_name"].unique())

if len(available_funds) > 0:
    fund = st.selectbox("Select Fund for Detailed Analysis", available_funds)
    
    fdf = idcw_df[idcw_df["fund_name"] == fund].sort_values("FY")
    fund_info = filtered[filtered["fund_name"] == fund].iloc[0]
    
    # Fund metrics
    col1, col2, col3, col4, col5, col6 = st.columns(6)
    
    with col1:
        st.metric("AMC", fund_info["AMC"][:20] + "..." if len(str(fund_info["AMC"])) > 20 else fund_info["AMC"])
    
    with col2:
        st.metric("Plan", fund_info["Plan"])
    
    with col3:
        st.metric("Latest Yield", f"{fund_info['IDCW_Yield_pct_x']:.2f}%")
    
    with col4:
        cagr = fund_info['CAGR_Yield']
        st.metric("CAGR", f"{cagr:.2f}%" if pd.notna(cagr) else "N/A")
    
    with col5:
        slope = fund_info['Trend_Slope']
        st.metric("Trend", f"{slope:.3f}%/yr" if pd.notna(slope) else "N/A")
    
    with col6:
        consecutive = fund_info['Max_Consecutive_Increases']
        st.metric("Max Consecutive ↑", f"{int(consecutive)}" if pd.notna(consecutive) else "N/A")
    
    st.divider()
    
    # Charts
    col1, col2 = st.columns(2)
    
    with col1:
        # NAV trend
        fig = px.line(
            fdf,
            x="FY",
            y=["April_NAV", "March_NAV"],
            markers=True,
            title="NAV Trend (April vs March)",
            labels={"value": "NAV", "variable": "Period"}
        )
        fig.update_layout(
            legend=dict(
                orientation="h",
                yanchor="bottom",
                y=1.02,
                xanchor="right",
                x=1
            ),
            hovermode="x unified"
        )
        st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        # Yield trend with trendline
        fig = go.Figure()
        
        fig.add_trace(go.Bar(
            x=fdf["FY"],
            y=fdf["IDCW_Yield_pct"],
            name="Yield %",
            marker_color='lightblue'
        ))
        
        # Add trendline
        x_numeric = np.arange(len(fdf))
        if len(fdf) >= 2:
            z = np.polyfit(x_numeric, fdf["IDCW_Yield_pct"], 1)
            p = np.poly1d(z)
            
            fig.add_trace(go.Scatter(
                x=fdf["FY"],
                y=p(x_numeric),
                mode='lines',
                name='Trend',
                line=dict(color='red', dash='dash')
            ))
        
        fig.update_layout(
            title="IDCW Yield % by FY (with Trendline)",
            xaxis_title="Financial Year",
            yaxis_title="Yield (%)",
            hovermode="x unified"
        )
        
        st.plotly_chart(fig, use_container_width=True)
    
    # YoY Growth Chart
    if len(fdf) > 1:
        fdf_sorted = fdf.sort_values("FY").reset_index(drop=True)
        fdf_sorted["YoY_Growth"] = fdf_sorted["IDCW_Yield_pct"].pct_change() * 100
        
        fig = go.Figure()
        
        colors = ['green' if x > 0 else 'red' if x < 0 else 'gray' 
                  for x in fdf_sorted["YoY_Growth"][1:]]
        
        fig.add_trace(go.Bar(
            x=fdf_sorted["FY"][1:],
            y=fdf_sorted["YoY_Growth"][1:],
            marker_color=colors,
            name="YoY Growth %"
        ))
        
        fig.update_layout(
            title="Year-over-Year Yield Growth (%)",
            xaxis_title="Financial Year",
            yaxis_title="Growth (%)",
            showlegend=False,
            hovermode="x"
        )
        
        st.plotly_chart(fig, use_container_width=True)
    
    # Detailed data table
    st.subheader("Historical Data")
    st.dataframe(
        fdf[[
            "FY",
            "April_NAV",
            "March_NAV",
            "Dividend",
            "FY_Return",
            "IDCW_Yield_pct"
        ]].round(2),
        column_config={
            "FY": "Financial Year",
            "April_NAV": st.column_config.NumberColumn("April NAV", format="%.2f"),
            "March_NAV": st.column_config.NumberColumn("March NAV", format="%.2f"),
            "Dividend": st.column_config.NumberColumn("Dividend", format="%.2f"),
            "FY_Return": st.column_config.NumberColumn("FY Return", format="%.2f"),
            "IDCW_Yield_pct": st.column_config.NumberColumn("IDCW Yield (%)", format="%.2f%%")
        },
        hide_index=True,
        use_container_width=True
    )
else:
    st.info("👆 Select filters in the sidebar to view funds")

# ========================
# FOOTER WITH INFO
# ========================
st.divider()

with st.expander("📖 Understanding Trend KPIs"):
    st.markdown("""
    **CAGR (Compound Annual Growth Rate)**: Measures the annual growth rate of dividend yield over time.
    - Positive CAGR = yield is increasing
    - Negative CAGR = yield is decreasing
    
    **Trend Slope**: Linear regression slope showing % change per year.
    - Higher slope = steeper upward trend
    - Negative slope = downward trend
    
    **Max Consecutive Increases**: Longest streak of year-over-year yield increases.
    - Higher number = more consistent growth
    
    **Consistency (CV)**: Coefficient of Variation - measures volatility.
    - Lower CV = more stable/predictable yields
    - Higher CV = more variable yields
    
    **Composite Score**: Weighted combination of:
    - 40% Latest yield
    - 30% Average yield
    - 20% Trend slope
    - 10% Consecutive increases
    """)

with st.expander("ℹ️ About This Dashboard"):
    st.markdown(f"""
    **Data Source**: Pre-generated CSV files from `{DATA_DIR}/` directory
    
    **How to Update Data**:
    1. Run `python data_generator.py` to fetch fresh data
    2. Click "🔄 Refresh Data" in the sidebar
    
    **Files Used**:
    - `idcw_fy_output.csv` - FY-wise performance data
    - `final_schemes_master.csv` - Scheme attributes
    - `trend_kpis.csv` - Calculated trend metrics
    
    **Note**: This dashboard only visualizes data. To generate new data, run the data generator script.
    """)

st.caption("📊 Data Source: AMFI & Yahoo Finance | ⚠️ For informational purposes only. Not investment advice.")
