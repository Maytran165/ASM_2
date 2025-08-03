import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error

# Load data
df = pd.read_csv("index_1 (1).csv")
df['date'] = pd.to_datetime(df['date'])
df['money'] = df['money'].astype(float)

st.title("🍵 Coffee Sales Dashboard + Revenue Forecast")
st.markdown("""
**Dataset:** Transactions of coffee shop by drink type, payment method, and date.
""")

# Sidebar filter
with st.sidebar:
    st.header("Filters")
    selected_drink = st.multiselect("Select Coffee Name", df['coffee_name'].unique(), default=df['coffee_name'].unique())
    selected_payment = st.multiselect("Select Payment Type", df['cash_type'].unique(), default=df['cash_type'].unique())

# Apply filters
filtered_df = df[(df['coffee_name'].isin(selected_drink)) & (df['cash_type'].isin(selected_payment))]

st.subheader("1. Doanh thu theo ngày")
daily_sales = filtered_df.groupby('date')['money'].sum().reset_index()
st.line_chart(daily_sales.set_index('date'))

st.subheader("2. Số đơn hàng theo loại đồ uống")
orders_by_coffee = filtered_df['coffee_name'].value_counts().reset_index()
orders_by_coffee.columns = ['coffee_name', 'orders']
st.bar_chart(orders_by_coffee.set_index('coffee_name'))

st.subheader("3. Tỉ lệ thanh toán")
payment_counts = filtered_df['cash_type'].value_counts()
st.plotly_chart(px.pie(names=payment_counts.index, values=payment_counts.values, title="Payment Method Proportion"))

st.subheader("4. Tổng doanh thu theo loại đồ uống")
revenue_by_coffee = filtered_df.groupby('coffee_name')['money'].sum().sort_values(ascending=False)
st.bar_chart(revenue_by_coffee)

st.subheader("5. Top 10 khách hàng chi tiêu nhiều nhất")
card_totals = filtered_df.groupby('card')['money'].sum().sort_values(ascending=False).head(10)
st.plotly_chart(px.bar(card_totals, orientation='v', title="Top 10 Customers by Spending"))

# Forecast
st.subheader("6. Dự báo doanh thu bằng Linear Regression")
daily = df.groupby('date')['money'].sum().reset_index()
daily['day_num'] = (daily['date'] - daily['date'].min()).dt.days
X = daily['day_num'].values.reshape(-1, 1)
y = daily['money'].values

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, shuffle=False)
model = LinearRegression()
model.fit(X_train, y_train)
y_pred = model.predict(X_test)

mse = mean_squared_error(y_test, y_pred)
st.metric("MSE (Test Set)", f"{mse:.2f}")

# Future prediction
future_days = pd.DataFrame({"day_num": np.arange(X.max() + 1, X.max() + 15).reshape(-1, 1).flatten()})
future_pred = model.predict(future_days[['day_num']])
future_dates = pd.date_range(start=daily['date'].max() + pd.Timedelta(days=1), periods=14)
forecast_df = pd.DataFrame({"date": future_dates, "forecasted_revenue": future_pred})

fig = px.line(forecast_df, x='date', y='forecasted_revenue', title="14-Day Forecast of Revenue")
st.plotly_chart(fig)

st.success("App ready! You can upload it to Streamlit Cloud to deploy.")
