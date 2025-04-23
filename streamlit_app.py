import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px

# Load synthetic customer data (replace with your S3 data later)
df = pd.read_csv("sample_customer_data.csv")

# Sidebar filters
st.sidebar.title("🔍 Filter Customers")
min_risk = st.sidebar.slider("Minimum Churn Risk", 0.0, 1.0, 0.5, 0.01)
tier_filter = st.sidebar.multiselect("Loyalty Tier", options=df["tier"].unique(), default=list(df["tier"].unique()))

# Filtered data
filtered_df = df[(df["churn_score"] >= min_risk) & (df["tier"].isin(tier_filter))]

# --- Dashboard Header ---
st.title("✈️ Travel Loyalty Churn Dashboard")

col1, col2, col3, col4 = st.columns(4)
col1.metric("Total Customers", len(df))
col2.metric("High Risk Customers", len(df[df["churn_score"] > 0.7]))
col3.metric("Gold/Platinum at Risk", len(df[(df["tier"].isin(["Gold", "Platinum"])) & (df["churn_score"] > 0.7)]))
col4.metric("Rewards Suggested", df["suggested_reward"].nunique())

# --- Charts ---
st.subheader("📉 Churn Risk Distribution")
fig = px.histogram(df, x="churn_score", nbins=20, title="Churn Score Histogram")
st.plotly_chart(fig, use_container_width=True)

st.subheader("🎯 Reward Suggestions Summary")
reward_counts = df["suggested_reward"].value_counts().reset_index()
reward_counts.columns = ["Reward", "Count"]
fig2 = px.bar(reward_counts, x="Reward", y="Count", title="Most Recommended Rewards")
st.plotly_chart(fig2, use_container_width=True)

# --- Customer Table ---
st.subheader("📋 Customer Insights Table")
st.dataframe(filtered_df[["name", "tier", "churn_score", "last_booking_days", "suggested_reward"]])

# --- Customer Detail View ---
st.subheader("🔎 View Customer Details")
customer_name = st.selectbox("Select a customer", filtered_df["name"].unique())
customer = filtered_df[filtered_df["name"] == customer_name].iloc[0]

st.markdown(f"**Name:** {customer['name']}")
st.markdown(f"**Loyalty Tier:** {customer['tier']}")
st.markdown(f"**Churn Score:** {customer['churn_score']:.2f}")
st.markdown(f"**Last Booking:** {customer['last_booking_days']} days ago")
st.markdown(f"**Suggested Reward:** 🏅 {customer['suggested_reward']}")

# Optional action simulation
if st.button("🎁 Simulate Sending Reward"):
    st.success(f"Reward '{customer['suggested_reward']}' sent to {customer['name']}!")

# --- Notes ---
st.caption("Data is synthetic. Dashboard designed for hackathon use with AWS-aware minimal deployment.")
