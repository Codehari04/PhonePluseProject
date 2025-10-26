# phonepe_pulse_pro_analytics.py
import streamlit as st
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
import json
import os
from datetime import datetime
import warnings
from sklearn.linear_model import LinearRegression
import numpy as np

warnings.filterwarnings('ignore')

# Page Configuration
st.set_page_config(
    page_title="PhonePe Pulse ",
    page_icon="💳",
    layout="wide",
    initial_sidebar_state="expanded"
)

# CSS (keeps your PhonePe style)
st.markdown("""
    <style>
    .main {
        background: linear-gradient(135deg, #5f259f 0%, #3d1a5f 100%);
    }
    .stMetric {
        background: linear-gradient(135deg, rgba(139, 92, 246, 0.3) 0%, rgba(109, 40, 217, 0.3) 100%);
        padding: 20px;
        border-radius: 15px;
        border: 1px solid rgba(167, 139, 250, 0.3);
        backdrop-filter: blur(10px);
    }
    .metric-card {
        background: linear-gradient(135deg, rgba(139, 92, 246, 0.2) 0%, rgba(109, 40, 217, 0.2) 100%);
        padding: 14px;
        border-radius: 12px;
        border: 1px solid rgba(167, 139, 250, 0.3);
        margin: 8px 0;
    }
    h1, h2, h3 { color: #ffffff !important; }
    .plot-container {
        background: rgba(255,255,255,0.03);
        padding: 12px;
        border-radius: 12px;
    }
    div[data-testid="stSidebar"] {
        background: linear-gradient(180deg, #2d1b4e 0%, #1a0f2e 100%);
    }
    </style>
""", unsafe_allow_html=True)

# Configure your data path here
DATA_PATH = r"C:\Users\ADMIN\OneDrive\文档\project hari-20251021T031049Z-1-001\project hari\pulse-master\pulse-master\data"  # update if needed

# ---------------------------
# Data Loading Helpers (cached)
# ---------------------------

@st.cache_data(ttl=600)
def load_aggregated_transaction_data(year, quarter, state=None):
    """Load aggregated transaction data from JSON files"""
    all_data = []
    base_path = os.path.join(DATA_PATH, "aggregated", "transaction", "country", "india", "state")
    if not os.path.exists(base_path):
        return pd.DataFrame()
    if state and state != "All India":
        states_to_process = [state.lower().replace(" ", "-").replace("&", "and")]
    else:
        try:
            states_to_process = [d for d in os.listdir(base_path) if os.path.isdir(os.path.join(base_path, d))]
        except:
            return pd.DataFrame()
    for state_folder in states_to_process:
        state_path = os.path.join(base_path, state_folder, str(year), f"{quarter}.json")
        if os.path.exists(state_path):
            try:
                with open(state_path, 'r') as f:
                    data = json.load(f)
                if 'data' in data and 'transactionData' in data['data']:
                    for transaction in data['data']['transactionData']:
                        all_data.append({
                            'state': state_folder.replace("-", " ").title(),
                            'year': year,
                            'quarter': quarter,
                            'transaction_type': transaction.get('name'),
                            'transaction_count': transaction.get('paymentInstruments', [{}])[0].get('count', 0),
                            'transaction_amount': transaction.get('paymentInstruments', [{}])[0].get('amount', 0)
                        })
            except Exception:
                continue
    return pd.DataFrame(all_data)

@st.cache_data(ttl=600)
def load_top_transaction_data(year, quarter, category='states'):
    """Load top transaction data from JSON files"""
    all_data = []
    category_map = {'states': 'state', 'districts': 'district', 'pincodes': 'pincode'}
    folder_name = category_map.get(category, 'state')
    base_path = os.path.join(DATA_PATH, "top", "transaction", "country", "india", folder_name)
    if not os.path.exists(base_path):
        return pd.DataFrame()
    try:
        entities = [d for d in os.listdir(base_path) if os.path.isdir(os.path.join(base_path, d))]
    except:
        return pd.DataFrame()
    for entity_folder in entities:
        entity_path = os.path.join(base_path, entity_folder, str(year), f"{quarter}.json")
        if os.path.exists(entity_path):
            try:
                with open(entity_path, 'r') as f:
                    data = json.load(f)
                collection_name = folder_name + 's'
                if 'data' in data and collection_name in data['data']:
                    for item in data['data'][collection_name]:
                        all_data.append({
                            'entity_name': item.get('entityName'),
                            'year': year,
                            'quarter': quarter,
                            'transaction_count': item.get('metric', {}).get('count', 0),
                            'transaction_amount': item.get('metric', {}).get('amount', 0)
                        })
            except Exception:
                continue
    df = pd.DataFrame(all_data)
    if not df.empty:
        df = df.sort_values('transaction_amount', ascending=False).head(10)
    return df

@st.cache_data(ttl=600)
def load_map_transaction_data(year, quarter, state):
    """Load district map data"""
    all_data = []
    state_folder = state.lower().replace(" ", "-").replace("&", "and")
    base_path = os.path.join(DATA_PATH, "map", "transaction", "hover", "country", "india", "state", state_folder)
    if not os.path.exists(base_path):
        return pd.DataFrame()
    file_path = os.path.join(base_path, str(year), f"{quarter}.json")
    if os.path.exists(file_path):
        try:
            with open(file_path, 'r') as f:
                data = json.load(f)
            if 'data' in data and 'hoverDataList' in data['data']:
                for item in data['data']['hoverDataList']:
                    all_data.append({
                        'district': item.get('name'),
                        'year': year,
                        'quarter': quarter,
                        'transaction_count': item.get('metric', [{}])[0].get('count', 0),
                        'transaction_amount': item.get('metric', [{}])[0].get('amount', 0)
                    })
        except Exception:
            pass
    df = pd.DataFrame(all_data)
    if not df.empty:
        df = df.sort_values('transaction_amount', ascending=False)
    return df

@st.cache_data(ttl=600)
def load_aggregated_user_data(year, quarter):
    """Load aggregated user data"""
    all_data = []
    base_path = os.path.join(DATA_PATH, "aggregated", "user", "country", "india", "state")
    if not os.path.exists(base_path):
        return pd.DataFrame()
    try:
        states = [d for d in os.listdir(base_path) if os.path.isdir(os.path.join(base_path, d))]
    except:
        return pd.DataFrame()
    for state_folder in states:
        state_path = os.path.join(base_path, state_folder, str(year), f"{quarter}.json")
        if os.path.exists(state_path):
            try:
                with open(state_path, 'r') as f:
                    data = json.load(f)
                if 'data' in data and 'aggregated' in data['data']:
                    agg = data['data']['aggregated']
                    all_data.append({
                        'state': state_folder.replace("-", " ").title(),
                        'year': year,
                        'quarter': quarter,
                        'registered_users': agg.get('registeredUsers', 0),
                        'app_opens': agg.get('appOpens', 0)
                    })
            except Exception:
                continue
    return pd.DataFrame(all_data)

@st.cache_data(ttl=600)
def load_aggregated_insurance_data(year, quarter):
    """Load aggregated insurance data"""
    all_data = []
    base_path = os.path.join(DATA_PATH, "aggregated", "insurance", "country", "india", "state")
    if not os.path.exists(base_path):
        return pd.DataFrame()
    try:
        states = [d for d in os.listdir(base_path) if os.path.isdir(os.path.join(base_path, d))]
    except:
        return pd.DataFrame()
    for state_folder in states:
        state_path = os.path.join(base_path, state_folder, str(year), f"{quarter}.json")
        if os.path.exists(state_path):
            try:
                with open(state_path, 'r') as f:
                    data = json.load(f)
                if 'data' in data and 'transactionData' in data['data']:
                    total_count = 0
                    total_amount = 0
                    for transaction in data['data']['transactionData']:
                        pi = transaction.get('paymentInstruments', [{}])[0]
                        total_count += pi.get('count', 0)
                        total_amount += pi.get('amount', 0)
                    all_data.append({
                        'state': state_folder.replace("-", " ").title(),
                        'year': year,
                        'quarter': quarter,
                        'transaction_count': total_count,
                        'transaction_amount': total_amount
                    })
            except Exception:
                continue
    return pd.DataFrame(all_data)

# ---------------------------
# Formatting utilities
# ---------------------------
def format_number(num):
    """Format numbers in Indian style"""
    try:
        num = float(num)
    except:
        return str(num)
    if num >= 1e7:
        return f"₹{num/1e7:.2f} Cr"
    elif num >= 1e5:
        return f"₹{num/1e5:.2f} L"
    elif num >= 1e3:
        return f"{num/1e3:.2f}K"
    else:
        return f"₹{num:.0f}" if num != 0 else "₹0"

def format_count(num):
    try:
        num = float(num)
    except:
        return str(num)
    if num >= 1e7:
        return f"{num/1e7:.2f} Cr"
    elif num >= 1e5:
        return f"{num/1e5:.2f} L"
    elif num >= 1e3:
        return f"{num/1e3:.2f}K"
    else:
        return f"{int(num)}"

# ---------------------------
# Plot utilities
# ---------------------------
def create_line_chart(df, x_col, y_col, title):
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=df[x_col], y=df[y_col], mode='lines+markers', line=dict(color='#a78bfa', width=3),
                             marker=dict(size=6)))
    fig.update_layout(title=title, paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)',
                      font=dict(color='white'), height=360)
    return fig

def create_bar_chart(df, x_col, y_col, title, orientation='v', color=None):
    if orientation == 'h':
        fig = go.Figure(data=[go.Bar(x=df[y_col], y=df[x_col], orientation='h',
                                     marker=dict(line=dict(color='rgba(167,139,250,0.3)', width=1),
                                                 color=color if color else '#a78bfa'),
                                     text=df[y_col].apply(lambda v: format_number(v)),
                                     textposition='auto')])
    else:
        fig = go.Figure(data=[go.Bar(x=df[x_col], y=df[y_col],
                                     marker=dict(line=dict(color='rgba(167,139,250,0.3)', width=1),
                                                 color=color if color else '#a78bfa'),
                                     text=df[y_col].apply(lambda v: format_number(v)),
                                     textposition='auto')])
    fig.update_layout(title=title, paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)',
                      font=dict(color='white'), height=420)
    return fig

def create_pie_chart(df, names_col, values_col, title):
    fig = go.Figure(data=[go.Pie(labels=df[names_col], values=df[values_col], hole=0.4, textinfo='label+percent')])
    fig.update_layout(title=title, paper_bgcolor='rgba(0,0,0,0)', font=dict(color='white'), height=360)
    return fig

def create_choropleth_states(df, value_col, title):
    # Plotly choropleth by state name isn't perfect offline; use px.choropleth with locations as 'state'
    try:
        fig = px.choropleth(df, locations='state', locationmode='country names', color=value_col,
                            scope='asia', color_continuous_scale='Purples')
        fig.update_layout(title=title, paper_bgcolor='rgba(0,0,0,0)', font=dict(color='white'), height=480)
        return fig
    except Exception:
        # fallback simple bar chart if geo mapping fails
        return create_bar_chart(df.sort_values(value_col, ascending=False).head(15), 'state', value_col, title)

# ---------------------------
# Analytical helpers
# ---------------------------
def build_trend_series(start_year=2018, end_year=None):
    """Build aggregate time series for transactions, users and insurance across quarters"""
    if end_year is None:
        end_year = datetime.now().year
    records = []
    for y in range(start_year, end_year + 1):
        for q in range(1, 5):
            # aggregate across all states
            tdf = load_aggregated_transaction_data(y, q)
            udf = load_aggregated_user_data(y, q)
            idf = load_aggregated_insurance_data(y, q)
            records.append({
                'year': y,
                'quarter': q,
                'period': f"{y}-Q{q}",
                'transaction_amount': tdf['transaction_amount'].sum() if not tdf.empty else 0,
                'transaction_count': tdf['transaction_count'].sum() if not tdf.empty else 0,
                'registered_users': udf['registered_users'].sum() if not udf.empty else 0,
                'app_opens': udf['app_opens'].sum() if not udf.empty else 0,
                'insurance_amount': idf['transaction_amount'].sum() if not idf.empty else 0,
                'insurance_count': idf['transaction_count'].sum() if not idf.empty else 0
            })
    df = pd.DataFrame(records)
    # drop trailing periods with zero across all main metrics to avoid noise
    df = df[(df['transaction_amount'] != 0) | (df['registered_users'] != 0) | (df['insurance_amount'] != 0)]
    df = df.sort_values(['year', 'quarter'])
    df.reset_index(drop=True, inplace=True)
    return df

def compute_growth(current, previous):
    try:
        if previous == 0:
            return None
        return ((current - previous) / previous) * 100
    except:
        return None

def simple_next_quarter_forecast(series_values, periods_ahead=1):
    """Lightweight linear regression forecast on series_values (list/np)"""
    # Use index as X, values as y
    if len(series_values) < 3:
        return None  # not enough history
    X = np.arange(len(series_values)).reshape(-1, 1)
    y = np.array(series_values).reshape(-1, 1)
    model = LinearRegression()
    model.fit(X, y)
    future_X = np.arange(len(series_values), len(series_values) + periods_ahead).reshape(-1, 1)
    pred = model.predict(future_X)
    return float(pred[-1][0])

# ---------------------------
# App UI
# ---------------------------
def main():
    # header
    col1, col2 = st.columns([1, 4])
    with col1:
        st.markdown("### 💳")
    with col2:
        st.title("PhonePe Pulse ")
        st.markdown("*Deeper insights, trends & lightweight forecasting*")
    st.markdown("---")

    # Sidebar controls
    with st.sidebar:
        st.markdown("## 📊 Controls")
        data_type = st.radio("Select Data Type", ["Transactions", "Users", "Insurance"], index=0)
        years = list(range(datetime.now().year, 2017, -1))
        selected_year = st.selectbox("📅 Year", years, index=0)
        quarters = ["Q1 (Jan-Mar)", "Q2 (Apr-Jun)", "Q3 (Jul-Sep)", "Q4 (Oct-Dec)"]
        selected_quarter = st.selectbox("📆 Quarter", quarters)
        quarter_num = quarters.index(selected_quarter) + 1

        st.markdown("---")
        states = ["All India", "Andaman & Nicobar", "Andhra Pradesh", "Arunachal Pradesh", 
                  "Assam", "Bihar", "Chandigarh", "Chhattisgarh", "Dadra and Nagar Haveli and Daman and Diu",
                  "Delhi", "Goa", "Gujarat", "Haryana", "Himachal Pradesh", "Jammu & Kashmir",
                  "Jharkhand", "Karnataka", "Kerala", "Ladakh", "Lakshadweep", "Madhya Pradesh",
                  "Maharashtra", "Manipur", "Meghalaya", "Mizoram", "Nagaland", "Odisha",
                  "Puducherry", "Punjab", "Rajasthan", "Sikkim", "Tamil Nadu", "Telangana",
                  "Tripura", "Uttar Pradesh", "Uttarakhand", "West Bengal"]
        selected_state = st.selectbox("🗺️ Select State for Drill-down", states, index=0)

        st.markdown("---")
        st.markdown("### 🔎 Compare States")
        compare_state_1 = st.selectbox("State A", states, index=22)  # default Maharashtra
        compare_state_2 = st.selectbox("State B", states, index=15)  # default Karnataka
        st.markdown("---")
        st.info("Tip: Use the comparison panel to quickly benchmark two states.")

    # Build trend series for trend charts & forecasting
    trend_df = build_trend_series(start_year=2018, end_year=datetime.now().year)
    latest_period_mask = (trend_df['year'] == selected_year) & (trend_df['quarter'] == quarter_num)

    # Content by data type
    if data_type == "Transactions":
        display_transaction_analytics(selected_year, quarter_num, selected_state, compare_state_1, compare_state_2, trend_df)
    elif data_type == "Users":
        display_user_analytics(selected_year, quarter_num, selected_state, compare_state_1, compare_state_2, trend_df)
    else:
        display_insurance_analytics(selected_year, quarter_num, selected_state, compare_state_1, compare_state_2, trend_df)

# ---------------------------
# Transaction Analytics
# ---------------------------
def display_transaction_analytics(year, quarter, state, s1, s2, trend_df):
    st.header("💰 Transaction Analytics")

    # Load current and previous
    df_current = load_aggregated_transaction_data(year, quarter, None)  # all states
    if df_current.empty:
        st.warning("No transaction data available for the selected period.")
        return

    # Aggregated metrics for selected filters
    if state != "All India":
        df_state = load_aggregated_transaction_data(year, quarter, state)
    else:
        df_state = df_current

    total_tx_count = int(df_state['transaction_count'].sum())
    total_tx_amount = float(df_state['transaction_amount'].sum())

    # previous quarter
    prev_q = quarter - 1 if quarter > 1 else 4
    prev_y = year if quarter > 1 else year - 1
    df_prev = load_aggregated_transaction_data(prev_y, prev_q, None)
    prev_amount = df_prev['transaction_amount'].sum() if not df_prev.empty else 0

    # same quarter last year (YoY)
    df_yoy = load_aggregated_transaction_data(year - 1, quarter, None)
    yoy_amount = df_yoy['transaction_amount'].sum() if not df_yoy.empty else 0

    QoQ = compute_growth(total_tx_amount, prev_amount) if prev_amount else None
    YoY = compute_growth(total_tx_amount, yoy_amount) if yoy_amount else None
    avg_tx_value = total_tx_amount / total_tx_count if total_tx_count else 0

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("🔢 Total Transactions", format_count(total_tx_count))
    c2.metric("💵 Total Payment Value", format_number(total_tx_amount),
              delta=f"{QoQ:+.2f}%" if QoQ is not None else "N/A")
    c3.metric("📊 Avg Transaction Value", f"₹{avg_tx_value:.0f}")
    c4.metric("📈 YoY Growth", f"{YoY:+.2f}%" if YoY is not None else "N/A")

    st.markdown("---")
    # Row: Trend & Forecast
    row1_col1, row1_col2 = st.columns([2, 1])
    with row1_col1:
        st.markdown("### 📈 Transaction Value Trend (All India)")
        if not trend_df.empty:
            t = trend_df[['period', 'transaction_amount']].copy()
            fig = create_line_chart(t, 'period', 'transaction_amount', 'Transaction Value by Period')
            st.plotly_chart(fig, use_container_width=True)
    with row1_col2:
        st.markdown("### 🔮 Next Quarter Projection (Lightweight)")
        # build the series of transaction_amount from trend_df
        series_vals = list(trend_df['transaction_amount'].values)
        forecast = simple_next_quarter_forecast(series_vals, periods_ahead=1)
        if forecast is not None:
            st.metric("Projected Next Quarter Value", format_number(forecast),
                      delta=f"{compute_growth(forecast, series_vals[-1]):+.2f}%" if len(series_vals) > 0 and series_vals[-1] else "N/A")
            st.caption("Forecast uses a simple linear regression on historical quarters (lightweight).")
        else:
            st.info("Not enough history to forecast.")

    st.markdown("---")
    # Category breakdown pie + top states/districts
    category_data = df_state.groupby('transaction_type').agg({'transaction_amount': 'sum', 'transaction_count': 'sum'}).reset_index()
    col1, col2 = st.columns([1, 1])
    with col1:
        st.markdown("### 📊 Category Contribution (Selected State)")
        if not category_data.empty:
            fig = create_pie_chart(category_data, 'transaction_type', 'transaction_amount', 'Transaction Amount by Category')
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("No category data")

    with col2:
        st.markdown("### 🏆 Top States (by Amount)")
        top_states = load_top_transaction_data(year, quarter, 'states')
        if not top_states.empty:
            fig = create_bar_chart(top_states, 'entity_name', 'transaction_amount', 'Top States by Transaction Amount', orientation='v')
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("Top states data ")

    st.markdown("---")
    # Compare two states
    st.markdown("### ⚖️ Compare Two States")
    compare_table = []
    for st_name in [s1, s2]:
        df_st = load_aggregated_transaction_data(year, quarter, st_name) if st_name != "All India" else df_current
        compare_table.append({
            'State': st_name,
            'Total Amount': df_st['transaction_amount'].sum() if not df_st.empty else 0,
            'Total Count': int(df_st['transaction_count'].sum()) if not df_st.empty else 0
        })
    compare_df = pd.DataFrame(compare_table)
    compare_df['Total Amount (Formatted)'] = compare_df['Total Amount'].apply(format_number)
    compare_df['Total Count (Formatted)'] = compare_df['Total Count'].apply(format_count)
    st.table(compare_df[['State', 'Total Amount (Formatted)', 'Total Count (Formatted)']])

    st.markdown("---")
    # Market share
    st.markdown("### 📊 Market Share — Top 10 States Contribution")
    agg_by_state = df_current.groupby('state').agg({'transaction_amount': 'sum'}).reset_index().sort_values('transaction_amount', ascending=False)
    if not agg_by_state.empty:
        top10 = agg_by_state.head(10).copy()
        top10['pct'] = (top10['transaction_amount'] / top10['transaction_amount'].sum()) * 100
        fig = create_pie_chart(top10, 'state', 'transaction_amount', 'Top 10 States Contribution')
        st.plotly_chart(fig, use_container_width=True)

    st.markdown("---")
    # District drill-down
    if state != "All India":
        st.markdown(f"### 📍 District-wise Top (for {state})")
        dist_df = load_map_transaction_data(year, quarter, state)
        if not dist_df.empty:
            fig = create_bar_chart(dist_df.head(15), 'district', 'transaction_amount', f'Top Districts in {state}', orientation='h')
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("District-level map data not available for this state.")

    # Auto-generated text summary
    st.markdown("---")
    st.markdown("### 🧠 Key Takeaways")
    summary_lines = []
    top_state_row = agg_by_state.iloc[0] if not agg_by_state.empty else None
    if top_state_row is not None:
        summary_lines.append(f"- **{top_state_row['state']}** leads transactions with {format_number(top_state_row['transaction_amount'])}, contributing {top_state_row['transaction_amount']/agg_by_state['transaction_amount'].sum()*100:.1f}% of top-10 sum.")
    if QoQ is not None:
        summary_lines.append(f"- Overall transaction value for selected filters is **{format_number(total_tx_amount)}**, QoQ change **{QoQ:+.2f}%**.")
    else:
        summary_lines.append(f"- Overall transaction value for selected filters is **{format_number(total_tx_amount)}**.")
    if YoY is not None:
        summary_lines.append(f"- YoY change is **{YoY:+.2f}%**.")
    if forecast is not None:
        summary_lines.append(f"- Projected next quarter transaction value: **{format_number(forecast)}** (simple linear model).")
    st.markdown("\n".join(summary_lines))

# ---------------------------
# User Analytics
# ---------------------------
def display_user_analytics(year, quarter, state, s1, s2, trend_df):
    st.header("👥 User Analytics")
    df_all = load_aggregated_user_data(year, quarter)
    if df_all.empty:
        st.warning("No user data available for the selected period.")
        return

    if state != "All India":
        df_state = df_all[df_all['state'] == state] if not df_all.empty else pd.DataFrame()
    else:
        df_state = df_all

    total_users = int(df_state['registered_users'].sum())
    total_opens = int(df_state['app_opens'].sum())
    avg_opens = total_opens / total_users if total_users else 0

    # previous
    prev_q = quarter - 1 if quarter > 1 else 4
    prev_y = year if quarter > 1 else year - 1
    df_prev = load_aggregated_user_data(prev_y, prev_q)
    prev_users = df_prev['registered_users'].sum() if not df_prev.empty else 0
    QoQ = compute_growth(total_users, prev_users) if prev_users else None

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("👤 Registered Users", format_count(total_users))
    c2.metric("📱 App Opens", format_count(total_opens))
    c3.metric("🔄 Avg Opens/User", f"{avg_opens:.1f}")
    c4.metric("📈 QoQ Users Growth", f"{QoQ:+.2f}%" if QoQ is not None else "N/A")

    st.markdown("---")
    # Trends
    st.markdown("### 📈 User Trend (Registered Users & App Opens)")
    if not trend_df.empty:
        u = trend_df[['period', 'registered_users', 'app_opens']].copy()
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=u['period'], y=u['registered_users'], name='Registered Users', mode='lines+markers'))
        fig.add_trace(go.Scatter(x=u['period'], y=u['app_opens'], name='App Opens', mode='lines+markers'))
        fig.update_layout(title="Registered Users vs App Opens", paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)', font=dict(color='white'))
        st.plotly_chart(fig, use_container_width=True)

    st.markdown("---")
    # Top states by users
    st.markdown("### 🏆 Top States by Registered Users")
    top_users = df_all.sort_values('registered_users', ascending=False).head(10)
    if not top_users.empty:
        fig = create_bar_chart(top_users, 'state', 'registered_users', 'Top States by Users', orientation='v', color='#10b981')
        st.plotly_chart(fig, use_container_width=True)

    st.markdown("---")
    # Compare two states
    st.markdown("### ⚖️ Compare Two States (Users)")
    compare_table = []
    for st_name in [s1, s2]:
        df_st = load_aggregated_user_data(year, quarter)
        if st_name != "All India":
            df_st = df_st[df_st['state'] == st_name]
        compare_table.append({
            'State': st_name,
            'Registered Users': int(df_st['registered_users'].sum()) if not df_st.empty else 0,
            'App Opens': int(df_st['app_opens'].sum()) if not df_st.empty else 0
        })
    cdf = pd.DataFrame(compare_table)
    cdf['Registered Users Fmt'] = cdf['Registered Users'].apply(format_count)
    cdf['App Opens Fmt'] = cdf['App Opens'].apply(format_count)
    st.table(cdf[['State', 'Registered Users Fmt', 'App Opens Fmt']])

# ---------------------------
# Insurance Analytics
# ---------------------------
def display_insurance_analytics(year, quarter, state, s1, s2, trend_df):
    st.header("🛡️ Insurance Analytics")
    df_all = load_aggregated_insurance_data(year, quarter)
    if df_all.empty:
        st.warning("No insurance data available for the selected period.")
        return

    if state != "All India":
        df_state = df_all[df_all['state'] == state]
    else:
        df_state = df_all

    total_policies = int(df_state['transaction_count'].sum())
    total_premium = float(df_state['transaction_amount'].sum())
    avg_premium = total_premium / total_policies if total_policies else 0

    # previous
    prev_q = quarter - 1 if quarter > 1 else 4
    prev_y = year if quarter > 1 else year - 1
    df_prev = load_aggregated_insurance_data(prev_y, prev_q)
    prev_premium = df_prev['transaction_amount'].sum() if not df_prev.empty else 0
    QoQ = compute_growth(total_premium, prev_premium) if prev_premium else None

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("📋 Total Policies", format_count(total_policies))
    c2.metric("💰 Total Premium", format_number(total_premium), delta=f"{QoQ:+.2f}%" if QoQ is not None else "N/A")
    c3.metric("📊 Avg Premium", f"₹{avg_premium:.0f}")
    c4.metric("📈 QoQ Growth", f"{QoQ:+.2f}%" if QoQ is not None else "N/A")

    st.markdown("---")
    st.markdown("### 🏆 Top States by Insurance Premium")
    top_prem = df_all.sort_values('transaction_amount', ascending=False).head(10)
    if not top_prem.empty:
        fig = create_bar_chart(top_prem, 'state', 'transaction_amount', 'Top States by Premium', orientation='v', color='#ef4444')
        st.plotly_chart(fig, use_container_width=True)

    st.markdown("---")
    # Compare two states
    st.markdown("### ⚖️ Compare Two States (Insurance)")
    compare_table = []
    for st_name in [s1, s2]:
        df_st = load_aggregated_insurance_data(year, quarter)
        if st_name != "All India":
            df_st = df_st[df_st['state'] == st_name]
        compare_table.append({
            'State': st_name,
            'Premium Amount': df_st['transaction_amount'].sum() if not df_st.empty else 0,
            'Policies': int(df_st['transaction_count'].sum()) if not df_st.empty else 0
        })
    cdf = pd.DataFrame(compare_table)
    cdf['Premium Fmt'] = cdf['Premium Amount'].apply(format_number)
    cdf['Policies Fmt'] = cdf['Policies'].apply(format_count)
    st.table(cdf[['State', 'Premium Fmt', 'Policies Fmt']])

# Run
if __name__ == "__main__":
    main()
