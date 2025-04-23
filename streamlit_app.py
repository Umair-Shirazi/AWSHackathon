import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime, timedelta

# Page configuration
st.set_page_config(layout="wide", page_title="Customer Churn Dashboard")

# Function to generate synthetic data
def generate_synthetic_data(n=1000):
    np.random.seed(42)
    
    # Customer IDs
    customer_ids = np.arange(1000, 1000+n)
    
    # Status
    status_options = ['Active', 'Inactive', 'New', 'Lost']
    status_weights = [0.65, 0.15, 0.10, 0.10]
    status = np.random.choice(status_options, size=n, p=status_weights)
    
    # Gender
    gender = np.random.choice(['Male', 'Female'], size=n)
    
    # Countries
    countries = np.random.choice(['France', 'Germany', 'Spain', 'UK', 'Italy', 'USA'], size=n, 
                              p=[0.2, 0.2, 0.15, 0.15, 0.1, 0.2])
    
    # Tenure (months)
    tenure = np.random.randint(1, 120, size=n)
    
    # Contract type
    contract_types = np.random.choice(['Monthly', 'Annual', 'Two-year'], size=n, p=[0.4, 0.35, 0.25])
    
    # Contract age (months)
    contract_age = np.random.randint(1, 60, size=n)
    
    # Contract volume
    contract_volume = np.random.randint(100, 1000, size=n)
    
    # Contract size
    contract_size = np.random.randint(1000, 50000, size=n)
    
    # Number of products
    num_products = np.random.randint(1, 5, size=n)
    
    # Balance
    balance = np.random.randint(0, 25000, size=n)
    
    # Spending 
    spending = np.random.randint(50, 5000, size=n)
    
    # Churn risk score
    churn_risk = np.random.beta(2, 5, size=n)  # Beta distribution for churn risk
    
    # Income in segments
    income_segments = np.random.randint(10000, 100000, size=n)
    
    # Location (US states)
    states = ['AL', 'AK', 'AZ', 'AR', 'CA', 'CO', 'CT', 'DE', 'FL', 'GA', 'HI', 'ID', 'IL', 'IN',
              'IA', 'KS', 'KY', 'LA', 'ME', 'MD', 'MA', 'MI', 'MN', 'MS', 'MO', 'MT', 'NE', 'NV',
              'NH', 'NJ', 'NM', 'NY', 'NC', 'ND', 'OH', 'OK', 'OR', 'PA', 'RI', 'SC', 'SD', 'TN',
              'TX', 'UT', 'VT', 'VA', 'WA', 'WV', 'WI', 'WY']
    location = np.random.choice(states, size=n)
    
    # Customer names
    first_names = ['James', 'John', 'Robert', 'Michael', 'William', 'David', 'Richard', 'Joseph', 'Thomas',
                  'Mary', 'Patricia', 'Jennifer', 'Linda', 'Elizabeth', 'Barbara', 'Susan', 'Jessica', 'Sarah']
    last_names = ['Smith', 'Johnson', 'Williams', 'Jones', 'Brown', 'Davis', 'Miller', 'Wilson', 'Moore',
                 'Taylor', 'Anderson', 'Thomas', 'Jackson', 'White', 'Harris', 'Martin', 'Thompson', 'Garcia']
    
    names = [f"{np.random.choice(first_names)} {np.random.choice(last_names)}" for _ in range(n)]
    
    # Create dataframe
    df = pd.DataFrame({
        'customer_id': customer_ids,
        'name': names,
        'status': status,
        'gender': gender,
        'country': countries,
        'tenure': tenure,
        'contract_type': contract_types,
        'contract_age': contract_age,
        'contract_volume': contract_volume,
        'contract_size': contract_size,
        'num_products': num_products,
        'balance': balance,
        'spending': spending,
        'churn_risk': churn_risk,
        'income': income_segments,
        'location': location
    })
    
    # Add a binary churn column (for illustration)
    df['churned'] = (df['churn_risk'] > 0.75).astype(int)
    
    return df

# Load or generate data
try:
    df = pd.read_csv("customer_churn_data.csv")
except:
    df = generate_synthetic_data(10000)

# Sidebar filters
st.sidebar.title("Filter Dashboard")
risk_filter = st.sidebar.slider("Churn Risk Threshold", 0.0, 1.0, 0.5)
status_filter = st.sidebar.multiselect("Status", options=df['status'].unique(), default=df['status'].unique())
country_filter = st.sidebar.multiselect("Country", options=df['country'].unique(), default=df['country'].unique())

# Apply filters
filtered_df = df[(df['churn_risk'] >= risk_filter) & 
                (df['status'].isin(status_filter)) & 
                (df['country'].isin(country_filter))]

# Main dashboard
st.title("Customer Churn Dashboard")

# Top KPI cards
col1, col2, col3, col4 = st.columns(4)

risky_customers = len(df[df['churn_risk'] > 0.7])
with col1:
    st.metric(label="Risky Customers", value=f"{risky_customers:,}")
    st.caption("Customers at high risk of churn")

impacted_revenue = int(sum(df[df['churn_risk'] > 0.7]['spending']) / 1000) * 1000
with col2:
    st.metric(label="Impacted Revenue From Risky Cohorts", value=f"${impacted_revenue/1000000:.1f}M")
    
avg_churn_rate = df['churned'].mean() * 100
with col3:
    st.metric(label="Average Churn Rate", value=f"{avg_churn_rate:.1f}%")
    
low_risk_revenue = int(sum(df[df['churn_risk'] < 0.3]['spending']) / 1000) * 1000
with col4:
    st.metric(label="Impacted Revenue From Low Churn Risk", value=f"${low_risk_revenue/1000000:.1f}M")

# Charts section
st.markdown("---")
col1, col2 = st.columns(2)

# Customers by Status chart
with col1:
    st.subheader("Customers By Status")
    
    # Get monthly data for each status
    months = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec']
    status_data = {status: np.random.randint(500, 5000, 12) for status in df['status'].unique()}
    
    fig = go.Figure()
    
    for status in df['status'].unique():
        fig.add_trace(go.Bar(
            x=months,
            y=status_data[status],
            name=status
        ))
    
    fig.update_layout(
        barmode='stack',
        xaxis_title="Month",
        yaxis_title="Number of Customers",
        legend_title="Status",
        height=300
    )
    
    st.plotly_chart(fig, use_container_width=True)

# Churn Risk by Income
with col2:
    st.subheader("Churn Risk By Income")
    
    # Group by income ranges
    df['income_range'] = pd.cut(df['income'], bins=10)
    income_risk = df.groupby('income_range')['churn_risk'].mean().reset_index()
    income_risk['income_pct'] = np.linspace(0, 100, len(income_risk))
    
    fig = px.bar(
        income_risk, 
        x='income_pct', 
        y='churn_risk',
        labels={'income_pct': '% Income', 'churn_risk': 'Churn Risk'},
        height=300
    )
    
    fig.update_layout(
        xaxis_title="Income Percentile",
        yaxis_title="Average Churn Risk"
    )
    
    st.plotly_chart(fig, use_container_width=True)

# Second row of charts
col1, col2 = st.columns(2)

# Segment scatter plot
with col1:
    st.subheader("Which Segments Are Likely To Leave?")
    
    # Create a scatter plot with spending vs churn risk
    # Sample a smaller number or use the entire dataset if it's small
    sample_size = min(100, len(df))  # Use at most 100 records or the entire dataset if smaller
    
    fig = px.scatter(
        df.sample(sample_size) if len(df) > sample_size else df,  # Sample appropriately
        x='spending',
        y='churn_risk',
        color='status',
        size='balance',
        hover_name='name',
        labels={'churn_risk': 'Churn Risk', 'spending': 'Spending'},
        height=350
    )
    
    fig.update_layout(
        xaxis_title="Avg. Spending",
        yaxis_title="Avg. Churn Risk"
    )
    
    st.plotly_chart(fig, use_container_width=True)

# Churn Risk by Location
with col2:
    st.subheader("Churn Risk By Location")
    
    # Aggregate by state
    state_risk = df.groupby('location')['churn_risk'].mean().reset_index()
    
    fig = px.choropleth(
        state_risk,
        locations='location',
        locationmode='USA-states',
        color='churn_risk',
        scope='usa',
        color_continuous_scale='Blues',
        labels={'churn_risk': 'Avg. Churn Risk'},
        height=350
    )
    
    st.plotly_chart(fig, use_container_width=True)

# Customers table
st.markdown("---")
st.subheader("Customer Details")

# Showing a sample of customers at high risk
high_risk_customers = df[df['churn_risk'] > 0.7].head(10)
displayed_cols = ['customer_id', 'name', 'gender', 'churn_risk', 'spending', 'location']
st.dataframe(high_risk_customers[displayed_cols], use_container_width=True)

# Churn Analytics Section
st.title("Churn Analytics Dashboard")

# Row 1
col1, col2, col3 = st.columns(3)

# Churn vs Not Churn pie chart
with col1:
    churned = len(df[df['churned'] == 1])
    not_churned = len(df) - churned
    
    fig = go.Figure(data=[go.Pie(
        labels=['Churned', 'Retained'],
        values=[churned, not_churned],
        hole=.3,
        marker_colors=['#FF6B6B', '#4ECDC4']
    )])
    
    fig.update_layout(
        title_text="Churn Vs Not Churn",
        height=300
    )
    
    fig.add_annotation(
        text=f"Churn Rate - {round(churned/len(df)*100, 1)}%",
        xref="paper", yref="paper",
        x=0.5, y=-0.1,
        showarrow=False
    )
    
    st.plotly_chart(fig, use_container_width=True)

# Gender Share
with col2:
    gender_counts = df.groupby('gender')['churned'].mean().reset_index()
    
    fig = px.pie(
        gender_counts,
        values='churned',
        names='gender',
        title='Gender Share',
        color_discrete_sequence=['#FF9F9F', '#4CB4C7'],
        height=300
    )
    
    st.plotly_chart(fig, use_container_width=True)

# Active To Churn
with col3:
    status_counts = df.groupby('status')['churned'].sum().reset_index()
    
    fig = px.bar(
        status_counts,
        x='status',
        y='churned',
        title='Active To Churn',
        color='status',
        height=300
    )
    
    st.plotly_chart(fig, use_container_width=True)

# Row 2
col1, col2, col3 = st.columns(3)

# Churn By Country
with col1:
    country_churn = df.groupby('country')['churned'].mean().reset_index()
    country_churn = country_churn.sort_values('churned', ascending=False).head(3)
    
    fig = px.bar(
        country_churn,
        x='country',
        y='churned',
        title='Churn By Country',
        color='country',
        height=300
    )
    
    st.plotly_chart(fig, use_container_width=True)

# Churn by Contract Age
with col2:
    df['contract_age_bin'] = pd.cut(df['contract_age'], bins=10)
    contract_age_churn = df.groupby('contract_age_bin')['churned'].mean().reset_index()
    
    fig = px.line(
        contract_age_churn,
        x='contract_age_bin',
        y='churned',
        title='Churn by Contract Age',
        height=300
    )
    
    st.plotly_chart(fig, use_container_width=True)

# Churn by Contract Size
with col3:
    df['contract_size_bin'] = pd.cut(df['contract_size'], bins=10)
    contract_size_churn = df.groupby('contract_size_bin')['churned'].mean().reset_index()
    
    fig = px.line(
        contract_size_churn,
        x='contract_size_bin',
        y='churned',
        title='Churn by Contract Size',
        height=300
    )
    
    st.plotly_chart(fig, use_container_width=True)

# Row 3
col1, col2, col3 = st.columns(3)

# Number of Products to Churn
with col1:
    product_churn = df.groupby('num_products')['churned'].mean().reset_index()
    
    fig = px.bar(
        product_churn,
        x='num_products',
        y='churned',
        title='No. Of Products To Churn',
        color='num_products',
        height=300
    )
    
    st.plotly_chart(fig, use_container_width=True)

# Churn by Contract Volume
with col2:
    df['contract_volume_bin'] = pd.cut(df['contract_volume'], bins=10)
    volume_churn = df.groupby('contract_volume_bin')['churned'].mean().reset_index()
    
    fig = px.line(
        volume_churn,
        x='contract_volume_bin',
        y='churned',
        title='Churn by Contract Volume',
        height=300
    )
    
    st.plotly_chart(fig, use_container_width=True)

# Tenure to Churn
with col3:
    df['tenure_bin'] = pd.cut(df['tenure'], bins=10)
    tenure_churn = df.groupby('tenure_bin')['churned'].mean().reset_index()
    
    fig = px.bar(
        tenure_churn,
        x='tenure_bin',
        y='churned',
        title='Tenure To Churn',
        color='tenure_bin',
        height=300
    )
    
    st.plotly_chart(fig, use_container_width=True)

# Row 4
col1, col2 = st.columns(2)

# Balance to Churn
with col1:
    df['balance_bin'] = pd.cut(df['balance'], bins=10)
    balance_churn = df.groupby('balance_bin')['churned'].mean().reset_index()
    
    fig = px.bar(
        balance_churn,
        x='balance_bin',
        y='churned',
        title='Balance To Churn',
        color='balance_bin',
        height=300
    )
    
    st.plotly_chart(fig, use_container_width=True)

# Churn by Partner
with col2:
    # Generate synthetic partner data
    df['partner'] = np.random.choice(['Visa Card', 'No Card'], size=len(df))
    partner_churn = df.groupby('partner')['churned'].sum().reset_index()
    
    fig = px.bar(
        partner_churn,
        x='partner',
        y='churned',
        title='Churn by Partner',
        color='partner',
        height=300
    )
    
    st.plotly_chart(fig, use_container_width=True)

# Add a footer
st.markdown("---")
st.caption("Dashboard created with Streamlit. Data is synthetic.")
