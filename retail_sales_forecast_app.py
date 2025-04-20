import streamlit as st
import pandas as pd
import joblib
import time
from calendar import month_name

# ----------- Styling ----------
st.markdown("""
    <style>
    body {
        background-color: #0a1f44;
    }
    header, .css-18ni7ap.e8zbici2 {
            background-color: #001f3f !important;
        }
    .stApp {
        background-color: #0a1f44;
        color: white;
    }
    .stButton>button {
        background-color: #1f4788;
        color: white;
        font-weight: bold;
        border-radius: 10px;
        padding: 10px 24px;
        transition: transform 0.2s ease;
    }
    .stButton>button:hover {
        transform: scale(1.05);
        background-color: #2b5da8;
    }
    .stNumberInput label,
    .stSelectbox label {
        color: white !important;
    }
    </style>
""", unsafe_allow_html=True)

# ----------- Data & Model Loaders ----------
@st.cache_data
def load_data():
    return pd.read_csv("encoded_data.csv")

@st.cache_resource
def load_model_scaler():
    model = joblib.load("sales_forecast_model.pkl")
    scaler = joblib.load("sales_forecast_scaler.pkl")
    return model, scaler

data = load_data()
model, scaler = load_model_scaler()

# Lookup Tables
product_encoded_table = data[['product_name', 'product_encoded']].drop_duplicates()
city_encoded_table = data[['city', 'city_encoded']].drop_duplicates()

# ----------- App UI ----------
st.title("📈 Sales Profit Forecasting App (AL-HILAL RETAILS)")
st.markdown("""
<hr style="border: 1px solid white; margin-top: 20px; margin-bottom: 20px;">
""", unsafe_allow_html=True)
# st.markdown("### Enter Details for the Latest Known Values")
# Hints in compact horizontal layout with white font
st.markdown("""
<div style='color:white; font-size:12px; margin-top:-10px;'>
    <strong>Example Input:</strong>
    'quantity': 4, &nbsp;
    'price': 266.13, &nbsp;
    'product_encoded': T-shirt, &nbsp;
    'city_encoded': Katherineview, &nbsp;
    'Discount': 2%, &nbsp;
    'last_month_profit': 846.44, &nbsp;
    'avg_last_3_months_profit': 485.47, &nbsp;
    'month_over_month_change': 0.257644, &nbsp;
    'cumulative_sales_to_date': 554077.97, &nbsp;
    'season': Winter, &nbsp;
    'order_month': May, &nbsp;
    'order_day': 16, &nbsp;
    'order_weekday': Monday, &nbsp;
    'order_year': 2025
</div>
""", unsafe_allow_html=True)
col1, col2 = st.columns(2)
with col1:
    quantity = st.number_input("Quantity", min_value=0.0)
with col2:
    price = st.number_input("Price", min_value=0.0)

# Row 2
col3, col4 = st.columns(2)
with col3:
    discount = st.number_input("Discount (%)", min_value=0.0, max_value=100.0, value=0.0)
    discounted_price = price * (1 - discount / 100)
with col4:
    product_name = st.selectbox("Product Name", product_encoded_table['product_name'].unique())
    product_encoded = product_encoded_table.loc[product_encoded_table['product_name'] == product_name, 'product_encoded'].values[0]

# Row 3
col5, col6 = st.columns(2)
with col5:
    city = st.selectbox("City", city_encoded_table['city'].unique())
    city_encoded = city_encoded_table.loc[city_encoded_table['city'] == city, 'city_encoded'].values[0]
with col6:
    season = st.selectbox("Season", [1, 2, 3, 4], format_func=lambda x: ["Winter", "Spring", "Summer", "Fall"][x-1])

# Row 4
col7, col8 = st.columns(2)
with col7:
    order_month = st.selectbox("Order Month", list(range(1, 13)), format_func=lambda x: month_name[x])
with col8:
    order_day = st.number_input("Order Day", min_value=1, max_value=31, step=1)

# Row 5
col9, col10 = st.columns(2)
with col9:
    order_weekday = st.selectbox("Order Weekday", list(range(7)), format_func=lambda x: ["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"][x])
with col10:
    order_year = st.number_input("Order Year", min_value=2000, max_value=2100, step=1)

# Row 6
col11, col12 = st.columns(2)
with col11:
    last_month_profit = st.number_input("Last Month Profit")
with col12:
    avg_last_3_months_profit = st.number_input("Average Last 3 Months Profit")

# Row 7
col13, col14 = st.columns(2)
with col13:
    month_over_month_change = st.number_input("Month over Month Change")
with col14:
    cumulative_sales_to_date = st.number_input("Cumulative Sales to Date")

# ----------- Forecast Button and Logic -----------
if st.button("📊 Forecast Profit"):
    with st.spinner("Crunching the numbers..."):
        time.sleep(2)

        latest_row = {
            'quantity': quantity,
            'price': discounted_price,
            'product_encoded': product_encoded,
            'city_encoded': city_encoded,
            'last_month_profit': last_month_profit,
            'avg_last_3_months_profit': avg_last_3_months_profit,
            'month_over_month_change': month_over_month_change,
            'cumulative_sales_to_date': cumulative_sales_to_date,
            'season': season,
            'order_month': order_month,
            'order_day': order_day,
            'order_weekday': order_weekday,
            'order_year': order_year,
        }

        # First prediction
        input_df = pd.DataFrame([latest_row])
        scaled_input = scaler.transform(input_df)
        first_pred = model.predict(scaled_input)[0]

        # Second month prediction
        next_row = latest_row.copy()
        next_row['order_month'] = (order_month % 12) + 1
        next_row['order_year'] += 1 if next_row['order_month'] == 1 else 0
        next_row['last_month_profit'] = first_pred
        next_row['avg_last_3_months_profit'] = (avg_last_3_months_profit * 2 + first_pred) / 3
        next_row['month_over_month_change'] = (first_pred - last_month_profit) / last_month_profit if last_month_profit else 0.0
        next_row['cumulative_sales_to_date'] = cumulative_sales_to_date + first_pred
        next_row['season'] = ((next_row['order_month'] % 12 + 3) // 3)

        next_df = pd.DataFrame([next_row])
        next_scaled = scaler.transform(next_df)
        second_pred = model.predict(next_scaled)[0]

    # Wow factor: Balloons + Styled Profit Results
    st.balloons()
    st.markdown(f"""
        <div style='background-color:#1f4788; padding: 20px; border-radius: 12px; color:white; text-align:center;'>
            <h3>💰 Profit Forecast</h3>
            <p style='font-size:24px;'>For <strong>{month_name[order_month]}</strong>: 
                <span style='color:#00ffcc;'>{first_pred:.2f}</span></p>
            <p style='font-size:20px;'>Next Month (<strong>{month_name[next_row['order_month']]}</strong>): 
                <span style='color:#ffd700;'>{second_pred:.2f}</span></p>
        </div>
    """, unsafe_allow_html=True)

    st.toast(f"Forecast completed for {month_name[order_month]} & {month_name[next_row['order_month']]}", icon="📈")
