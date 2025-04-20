import streamlit as st
import pandas as pd
import joblib
from calendar import month_name
import streamlit.components.v1 as components

# Enhanced dark theme with white text and dark blue navbar/button
st.markdown("""
    <style>
        body, .stApp {
            background-color: #001f3f;
            color: white;
        }

        header, .css-18ni7ap.e8zbici2 {
            background-color: #001f3f !important;
        }

        h1, h2, h3, h4, h5, h6, p, label, .stTextInput label, .stNumberInput label, .stSelectbox label, .css-1cpxqw2 {
            color: white !important;
        }

        .stSelectbox>div>div>div {
            background-color: #003366;
            color: white;
        }

        .stNumberInput>div>div>input,
        .stTextInput>div>div>input {
            background-color: #003366;
            color: white;
            border: 1px solid #00d4ff;
        }

        .stSelectbox div[role="combobox"] {
            background-color: #003366 !important;
            color: white !important;
        }

            .stButton>button {
            background-color: white !important;
            color: #001f3f !important;  /* 🔵 Dark Blue Text */
            border-radius: 8px;
            height: 3em;
            width: 100%;
            font-size: 1.1em;
            border: none;
            transition: background-color 0.3s ease;
        }

        .stButton>button:hover {
            background-color: #00d4ff !important;
            color: #001f3f !important;
        }

        .wow-box {
            border: 2px solid #00d4ff;
            border-radius: 12px;
            padding: 20px;
            margin-top: 20px;
            background-color: #003366;
            box-shadow: 0 0 20px rgba(0, 212, 255, 0.3);
        }

        .css-1q8dd3e, .css-1d391kg {
            color: white !important;
        }

    </style>
""", unsafe_allow_html=True)

# Load data
@st.cache_data
def load_data():
    return pd.read_csv("encoded_data.csv")

data = load_data()

product_encoded_table = data[['product_name', 'product_encoded']].drop_duplicates()
city_encoded_table = data[['city', 'city_encoded']].drop_duplicates()

@st.cache_resource
def load_model_scaler():
    model = joblib.load("sales_forecast_model.pkl")
    scaler = joblib.load("sales_forecast_scaler.pkl")
    return model, scaler

model, scaler = load_model_scaler()

# App header
st.title("📊 Sales Profit Forecasting App")
st.markdown("### 📝 Enter Details Below to Forecast Profit")

# Form layout for input
with st.form("forecast_form"):
    st.markdown("#### 🔧 Input Features")
    col1, col2 = st.columns(2)
    with col1:
        quantity = st.number_input("Quantity", min_value=0.0)
        price = st.number_input("Price", min_value=0.0)
        discount = st.number_input("Discount (%)", min_value=0.0, max_value=100.0, value=0.0)
        discounted_price = price * (1 - discount / 100)
        product_name = st.selectbox("Product Name", product_encoded_table['product_name'].unique())
        product_encoded = product_encoded_table.loc[product_encoded_table['product_name'] == product_name, 'product_encoded'].values[0]
    with col2:
        city = st.selectbox("City", city_encoded_table['city'].unique())
        city_encoded = city_encoded_table.loc[city_encoded_table['city'] == city, 'city_encoded'].values[0]
        last_month_profit = st.number_input("Last Month Profit")
        avg_last_3_months_profit = st.number_input("Avg Last 3 Months Profit")
        month_over_month_change = st.number_input("Month over Month Change")
        cumulative_sales_to_date = st.number_input("Cumulative Sales to Date")

    col3, col4 = st.columns(2)
    with col3:
        season = st.selectbox("Season", [1, 2, 3, 4], format_func=lambda x: ["Winter", "Spring", "Summer", "Fall"][x - 1])
        order_month = st.selectbox("Order Month", list(range(1, 13)), format_func=lambda x: month_name[x])
    with col4:
        order_day = st.number_input("Order Day", min_value=1, max_value=31, step=1)
        order_weekday = st.selectbox("Order Weekday", list(range(7)), format_func=lambda x: ["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"][x])
        order_year = st.number_input("Order Year", min_value=2000, max_value=2100, step=1)

    submit = st.form_submit_button("🚀 Forecast Profit")

if submit:
    with st.spinner("🔮 Forecasting in progress..."):
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

        input_df = pd.DataFrame([latest_row])
        scaled_input = scaler.transform(input_df)
        first_pred = model.predict(scaled_input)[0]

        st.markdown(f"""
            <div class="wow-box">
                <h3>📈 Forecast Result for <u>{month_name[order_month]}</u>:</h3>
                <h2 style="color:lime;">💰 {first_pred:.2f}</h2>
            </div>
        """, unsafe_allow_html=True)

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

        st.markdown(f"""
            <div class="wow-box">
                <h3>🔁 Forecast Result for <u>{month_name[next_row['order_month']]}</u>:</h3>
                <h2 style="color:gold;">💰 {second_pred:.2f}</h2>
            </div>
        """, unsafe_allow_html=True)

# Optional animation
components.html(
    """
    <lottie-player src="https://assets1.lottiefiles.com/packages/lf20_pprxh53t.json"
     background="transparent"  speed="1"  style="width: 100%; height: 300px;" loop autoplay></lottie-player>
    """,
    height=300,
)
