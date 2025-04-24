import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime, timedelta
import random
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans

# Page configuration
st.set_page_config(layout="wide", page_title="Travel Loyalty & Churn Dashboard")


# Function to generate synthetic travel customer data
def generate_travel_customer_data(n=1000):
    np.random.seed(42)

    # Customer IDs
    customer_ids = np.arange(1000, 1000 + n)

    # Basic demographics
    age = np.random.randint(18, 75, size=n)
    gender = np.random.choice(['Male', 'Female', 'Other'], size=n, p=[0.48, 0.48, 0.04])

    # Countries/locations
    countries = np.random.choice(['France', 'Germany', 'Spain', 'UK', 'Italy', 'USA'], size=n,
                                 p=[0.2, 0.2, 0.15, 0.15, 0.1, 0.2])

    # Names
    first_names = ['James', 'John', 'Robert', 'Michael', 'William', 'David', 'Richard', 'Joseph', 'Thomas',
                   'Mary', 'Patricia', 'Jennifer', 'Linda', 'Elizabeth', 'Barbara', 'Susan', 'Jessica', 'Sarah']
    last_names = ['Smith', 'Johnson', 'Williams', 'Jones', 'Brown', 'Davis', 'Miller', 'Wilson', 'Moore',
                  'Taylor', 'Anderson', 'Thomas', 'Jackson', 'White', 'Harris', 'Martin', 'Thompson', 'Garcia']
    names = [f"{np.random.choice(first_names)} {np.random.choice(last_names)}" for _ in range(n)]

    # Signup date (within the past 5 years)
    current_date = datetime.now()
    signup_dates = [(current_date - timedelta(days=np.random.randint(1, 365 * 5))).strftime('%Y-%m-%d') for _ in
                    range(n)]

    # Loyalty tier
    loyalty_tiers = np.random.choice(['Bronze', 'Silver', 'Gold', 'Platinum'], size=n, p=[0.4, 0.3, 0.2, 0.1])

    # Travel engagement metrics
    total_bookings = np.random.poisson(10, size=n)  # Poisson distribution for booking count
    total_spend = np.random.gamma(shape=2, scale=500, size=n)  # Gamma distribution for spending

    # Last booking date (more recent for active customers, longer ago for potentially churned)
    last_booking_days_ago = np.random.exponential(scale=100, size=n)
    last_booking_dates = [(current_date - timedelta(days=days)).strftime('%Y-%m-%d') for days in last_booking_days_ago]

    # Engagement metrics
    days_since_login = np.random.exponential(scale=30, size=n)
    search_activity = np.random.poisson(5, size=n)
    campaign_clicks = np.random.poisson(3, size=n)
    support_requests = np.random.poisson(1, size=n)

    # Create a risk score based on days since last booking, login frequency, etc.
    # Higher score = higher churn risk
    booking_recency_factor = np.clip(last_booking_days_ago / 365, 0, 1)  # Normalize to 0-1
    login_recency_factor = np.clip(days_since_login / 100, 0, 1)  # Normalize to 0-1

    # Low engagement increases risk
    engagement_factor = 1 - np.clip((search_activity + campaign_clicks * 2) / 20, 0, 1)

    # Calculate churn risk (weighted combination of factors)
    churn_risk = 0.5 * booking_recency_factor + 0.3 * login_recency_factor + 0.2 * engagement_factor

    # Create dataframe
    df = pd.DataFrame({
        'customer_id': customer_ids,
        'name': names,
        'age': age,
        'gender': gender,
        'location': countries,
        'signup_date': signup_dates,
        'loyalty_tier': loyalty_tiers,
        'total_bookings': total_bookings,
        'total_spend': total_spend,
        'last_booking_date': last_booking_dates,
        'days_since_login': days_since_login,
        'search_activity_last_30d': search_activity,
        'campaign_clicks_last_90d': campaign_clicks,
        'support_requests_last_90d': support_requests,
        'churn_risk': churn_risk
    })

    # Add a computed churn rate for each customer (not a real metric, just for visualization)
    df['churn_rate'] = churn_risk * 100

    # Add days since last booking
    df['last_booking_date'] = pd.to_datetime(df['last_booking_date'])
    df['days_since_last_booking'] = (current_date - df['last_booking_date']).dt.days

    return df


# Load or generate customer data
try:
    df = pd.read_csv("travel_customer_data.csv")
    df['last_booking_date'] = pd.to_datetime(df['last_booking_date'])
except:
    df = generate_travel_customer_data(1000)

# Load rewards data
try:
    rewards_df = pd.read_csv("Reward_Inventory.csv")
except:
    # If file not available, create a sample rewards dataframe
    st.warning("Using sample rewards data. Please provide the Reward_Inventory.csv file for actual rewards.")
    rewards_df = pd.DataFrame({
        'reward_id': [f'reward_{i}' for i in range(1, 51)],
        'reward_type': np.random.choice(['Bonus Points', 'Discount', 'Free Upgrade', 'Exclusive Experience'], size=50),
        'description': ["Sample reward description"] * 50,
        'target_tier': np.random.choice(['Bronze', 'Silver', 'Gold', 'Platinum'], size=50),
        'estimated_value_usd': np.random.randint(20, 500, size=50)
    })

# Dashboard header
st.title("Travel Loyalty & Churn Dashboard")
st.subheader("Customer Retention Analysis & Reward Recommendations")

# Sidebar filters
st.sidebar.title("Filter Dashboard")
risk_filter = st.sidebar.slider("Churn Risk Threshold", 0.0, 1.0, 0.5, 0.05)
loyalty_filter = st.sidebar.multiselect("Loyalty Tier", options=df['loyalty_tier'].unique(),
                                        default=df['loyalty_tier'].unique())
location_filter = st.sidebar.multiselect("Location", options=df['location'].unique(), default=df['location'].unique())

# Apply filters
filtered_df = df[(df['churn_risk'] >= risk_filter) &
                 (df['loyalty_tier'].isin(loyalty_filter)) &
                 (df['location'].isin(location_filter))]

# KPI Row
col1, col2, col3, col4 = st.columns(4)

# 1. At-Risk Customers
high_risk_customers = len(df[df['churn_risk'] > 0.7])
with col1:
    st.metric(label="At-Risk Customers", value=f"{high_risk_customers:,}")
    st.caption("Customers with high churn risk (>70%)")

# 2. Revenue at Risk
at_risk_revenue = int(sum(df[df['churn_risk'] > 0.7]['total_spend']) / 1000) * 1000
with col2:
    st.metric(label="Revenue at Risk", value=f"${at_risk_revenue / 1000:.1f}K")
    st.caption("From high-risk customers")

# 3. Average Customer Lifetime Value
avg_clv = df['total_spend'].mean()
with col3:
    st.metric(label="Avg. Customer Lifetime Value", value=f"${avg_clv:.2f}")
    st.caption("Average spend per customer")

# 4. Retention Rate
retention_rate = 100 - (len(df[df['churn_risk'] > 0.7]) / len(df) * 100)
with col4:
    st.metric(label="Est. Retention Rate", value=f"{retention_rate:.1f}%")
    st.caption("Based on current churn risks")

# Charts section
st.markdown("---")

# First Row of Charts
col1, col2 = st.columns(2)

# Churn Risk by Loyalty Tier
with col1:
    st.subheader("Churn Risk by Loyalty Tier")

    tier_risk = df.groupby('loyalty_tier')['churn_risk'].mean().reset_index()
    tier_order = ['Bronze', 'Silver', 'Gold', 'Platinum']
    tier_risk['loyalty_tier'] = pd.Categorical(tier_risk['loyalty_tier'], categories=tier_order, ordered=True)
    tier_risk = tier_risk.sort_values('loyalty_tier')

    fig = px.bar(
        tier_risk,
        x='loyalty_tier',
        y='churn_risk',
        color='loyalty_tier',
        labels={'loyalty_tier': 'Loyalty Tier', 'churn_risk': 'Average Churn Risk'},
        height=300
    )

    fig.update_layout(
        xaxis_title="Loyalty Tier",
        yaxis_title="Average Churn Risk",
        showlegend=False
    )

    st.plotly_chart(fig, use_container_width=True)

# Days Since Last Booking vs Churn Risk
with col2:
    st.subheader("Booking Recency vs Churn Risk")

    # Create booking recency bins
    recency_bins = [0, 30, 90, 180, 365, float('inf')]
    recency_labels = ['< 30 days', '30-90 days', '90-180 days', '180-365 days', '> 1 year']

    df['recency_bucket'] = pd.cut(df['days_since_last_booking'], bins=recency_bins, labels=recency_labels)
    recency_risk = df.groupby('recency_bucket')['churn_risk'].mean().reset_index()

    fig = px.bar(
        recency_risk,
        x='recency_bucket',
        y='churn_risk',
        color='churn_risk',
        color_continuous_scale='Reds',
        labels={'recency_bucket': 'Time Since Last Booking', 'churn_risk': 'Average Churn Risk'},
        height=300
    )

    fig.update_layout(
        xaxis_title="Time Since Last Booking",
        yaxis_title="Average Churn Risk",
        showlegend=False
    )

    st.plotly_chart(fig, use_container_width=True)

# Second Row of Charts
col1, col2 = st.columns(2)

# Churn Risk by Location
with col1:
    st.subheader("Churn Risk by Location")

    location_risk = df.groupby('location')['churn_risk'].mean().reset_index()
    location_risk = location_risk.sort_values('churn_risk', ascending=False)

    fig = px.bar(
        location_risk,
        x='location',
        y='churn_risk',
        color='location',
        labels={'location': 'Location', 'churn_risk': 'Average Churn Risk'},
        height=300
    )

    fig.update_layout(
        xaxis_title="Location",
        yaxis_title="Average Churn Risk",
        showlegend=False
    )

    st.plotly_chart(fig, use_container_width=True)

# Engagement Metrics for At-Risk Customers
with col2:
    st.subheader("Engagement Metrics for At-Risk Customers")

    # Compare engagement metrics for high vs low risk customers
    high_risk = df[df['churn_risk'] > 0.7]
    low_risk = df[df['churn_risk'] < 0.3]

    high_risk_metrics = {
        'Avg Search Activity': high_risk['search_activity_last_30d'].mean(),
        'Avg Campaign Clicks': high_risk['campaign_clicks_last_90d'].mean(),
        'Avg Support Requests': high_risk['support_requests_last_90d'].mean(),
        'Avg Days Since Login': high_risk['days_since_login'].mean()
    }

    low_risk_metrics = {
        'Avg Search Activity': low_risk['search_activity_last_30d'].mean(),
        'Avg Campaign Clicks': low_risk['campaign_clicks_last_90d'].mean(),
        'Avg Support Requests': low_risk['support_requests_last_90d'].mean(),
        'Avg Days Since Login': low_risk['days_since_login'].mean()
    }

    metrics_df = pd.DataFrame({
        'Metric': list(high_risk_metrics.keys()),
        'High Risk': list(high_risk_metrics.values()),
        'Low Risk': list(low_risk_metrics.values())
    })

    metrics_df_long = pd.melt(metrics_df, id_vars=['Metric'], value_vars=['High Risk', 'Low Risk'],
                              var_name='Customer Group', value_name='Value')

    fig = px.bar(
        metrics_df_long,
        x='Metric',
        y='Value',
        color='Customer Group',
        barmode='group',
        height=300,
        labels={'Value': 'Average Value', 'Metric': 'Engagement Metric'}
    )

    fig.update_layout(
        xaxis_title="Engagement Metric",
        yaxis_title="Average Value"
    )

    st.plotly_chart(fig, use_container_width=True)

# Third Row - Customer Segments Analysis
st.subheader("Customer Segments Analysis")

# Use KMeans to identify distinct customer segments based on behavior
# Prepare data for clustering
features = df[['total_bookings', 'total_spend', 'days_since_last_booking',
               'days_since_login', 'search_activity_last_30d', 'campaign_clicks_last_90d',
               'churn_risk']]

# Standardize the features
scaler = StandardScaler()
features_scaled = scaler.fit_transform(features)

# Perform clustering (K-means with k=4)
kmeans = KMeans(n_clusters=4, random_state=42)
df['segment'] = kmeans.fit_predict(features_scaled)

# Map segment numbers to meaningful names based on characteristics
segment_centers = pd.DataFrame(scaler.inverse_transform(kmeans.cluster_centers_),
                               columns=features.columns)

# Determine segment names based on characteristics
segment_names = []
for i in range(len(segment_centers)):
    bookings = segment_centers.iloc[i]['total_bookings']
    spend = segment_centers.iloc[i]['total_spend']
    recency = segment_centers.iloc[i]['days_since_last_booking']
    risk = segment_centers.iloc[i]['churn_risk']

    if risk > 0.7:
        if spend > 1000:
            name = "High-Value At Risk"
        else:
            name = "Dormant Customers"
    else:
        if bookings > 10 and spend > 800:
            name = "Loyal Travelers"
        else:
            name = "Occasional Travelers"

    segment_names.append(name)

# Map segment numbers to names
segment_map = {i: name for i, name in enumerate(segment_names)}
df['segment_name'] = df['segment'].map(segment_map)

# Display segment analysis
col1, col2 = st.columns(2)

with col1:
    segment_counts = df['segment_name'].value_counts().reset_index()
    segment_counts.columns = ['Segment', 'Count']

    fig = px.pie(
        segment_counts,
        values='Count',
        names='Segment',
        height=350,
        hole=0.4
    )

    fig.update_layout(
        title_text="Customer Segments Distribution"
    )

    st.plotly_chart(fig, use_container_width=True)

with col2:
    segment_metrics = df.groupby('segment_name').agg({
        'churn_risk': 'mean',
        'total_spend': 'mean',
        'total_bookings': 'mean'
    }).reset_index()

    fig = px.scatter(
        segment_metrics,
        x='total_spend',
        y='churn_risk',
        size='total_bookings',
        color='segment_name',
        labels={'total_spend': 'Avg. Total Spend', 'churn_risk': 'Avg. Churn Risk',
                'total_bookings': 'Avg. Bookings', 'segment_name': 'Segment'},
        height=350
    )

    fig.update_layout(
        title_text="Segment Analysis: Spend vs. Churn Risk",
        xaxis_title="Average Total Spend ($)",
        yaxis_title="Average Churn Risk"
    )

    st.plotly_chart(fig, use_container_width=True)

# Reward Recommendations Section
st.markdown("---")
st.title("Reward Recommendations")


# Create a function to recommend rewards based on customer profile
def recommend_rewards(customer_row, rewards_df):
    tier = customer_row['loyalty_tier']
    churn_risk = customer_row['churn_risk']
    days_since_booking = customer_row['days_since_last_booking']

    # Filter rewards by loyalty tier (customer can access their tier and below)
    tier_mapping = {'Platinum': 4, 'Gold': 3, 'Silver': 2, 'Bronze': 1}
    customer_tier_level = tier_mapping.get(tier, 1)
    eligible_tiers = [k for k, v in tier_mapping.items() if v <= customer_tier_level]
    eligible_rewards = rewards_df[rewards_df['target_tier'].isin(eligible_tiers)]

    # Choose reward type based on churn risk and other factors
    if churn_risk > 0.7:
        # High churn risk - prioritize high value rewards
        if days_since_booking > 180:
            # Hasn't booked in a while - discount to encourage booking
            preferred_types = ['Discount', 'Free Upgrade', 'Bonus Points']
        else:
            # Recent booker but high risk - special experience
            preferred_types = ['Exclusive Experience', 'Free Upgrade', 'Discount']
    elif 0.4 <= churn_risk <= 0.7:
        # Medium risk - mix of incentives
        preferred_types = ['Bonus Points', 'Discount', 'Free Upgrade']
    else:
        # Low risk - maintain relationship
        preferred_types = ['Bonus Points', 'Exclusive Experience']

    # Filter for preferred reward types
    recommended_rewards = eligible_rewards[eligible_rewards['reward_type'].isin(preferred_types)]

    # Sort by estimated value
    if churn_risk > 0.5:
        # For high risk, sort by value descending (offer best rewards)
        recommended_rewards = recommended_rewards.sort_values('estimated_value_usd', ascending=False)
    else:
        # For low risk, sort by value ascending
        recommended_rewards = recommended_rewards.sort_values('estimated_value_usd', ascending=True)

    # Return top recommendation
    if len(recommended_rewards) > 0:
        return recommended_rewards.iloc[0]
    else:
        return None


# Display reward recommendations for high-risk customers
st.subheader("Targeted Reward Recommendations for At-Risk Customers")

# Filter high-risk customers
high_risk_customers = df[df['churn_risk'] > risk_filter].sort_values('churn_risk', ascending=False).head(10)

# Create recommendations
recommendations = []
for _, customer in high_risk_customers.iterrows():
    reward = recommend_rewards(customer, rewards_df)
    if reward is not None:
        recommendations.append({
            'customer_id': customer['customer_id'],
            'name': customer['name'],
            'loyalty_tier': customer['loyalty_tier'],
            'churn_risk': customer['churn_risk'],
            'days_since_booking': customer['days_since_last_booking'],
            'recommended_reward': reward['reward_type'],
            'reward_description': reward['description'],
            'reward_value': reward['estimated_value_usd']
        })

# Display recommendations
if recommendations:
    recommendations_df = pd.DataFrame(recommendations)
    st.dataframe(recommendations_df, use_container_width=True)
else:
    st.write("No recommendations available based on current filters.")

# Aggregate Reward Strategy
st.markdown("---")
st.subheader("Reward Strategy by Customer Segment")

# For each segment, show recommended reward distribution
segment_rewards = []
for segment in df['segment_name'].unique():
    segment_df = df[df['segment_name'] == segment].sample(min(50, len(df[df['segment_name'] == segment])))

    reward_counts = {}
    for _, customer in segment_df.iterrows():
        reward = recommend_rewards(customer, rewards_df)
        if reward is not None:
            reward_type = reward['reward_type']
            if reward_type in reward_counts:
                reward_counts[reward_type] += 1
            else:
                reward_counts[reward_type] = 1

    for reward_type, count in reward_counts.items():
        segment_rewards.append({
            'segment': segment,
            'reward_type': reward_type,
            'count': count,
            'percentage': count / len(segment_df) * 100
        })

# Create dataframe
segment_rewards_df = pd.DataFrame(segment_rewards)

if not segment_rewards_df.empty:
    # Create grouped bar chart
    fig = px.bar(
        segment_rewards_df,
        x='segment',
        y='percentage',
        color='reward_type',
        barmode='group',
        height=400,
        labels={'segment': 'Customer Segment', 'percentage': 'Recommendation %', 'reward_type': 'Reward Type'}
    )

    fig.update_layout(
        xaxis_title="Customer Segment",
        yaxis_title="% of Recommendations"
    )

    st.plotly_chart(fig, use_container_width=True)

    # Summary of strategy
    st.subheader("Reward Strategy Summary")
    st.write("""
    **Key Recommendations:**

    1. **High-Value At Risk:** Focus on exclusive experiences and substantial upgrades to recognize their value and rebuild engagement
    2. **Dormant Customers:** Offer significant discounts to encourage them to make a new booking
    3. **Loyal Travelers:** Reward with special experiences and bonus points to maintain their loyalty
    4. **Occasional Travelers:** Provide bonus points and small discounts to increase booking frequency
    """)
else:
    st.write("Not enough data to generate reward strategy visualization.")

# Customer Explorer Section
st.markdown("---")
st.title("Customer Explorer")

# Search by customer ID or name
search_term = st.text_input("Search Customer by ID or Name")

if search_term:
    results = df[df['name'].str.contains(search_term, case=False) |
                 df['customer_id'].astype(str).str.contains(search_term)]

    if not results.empty:
        customer = results.iloc[0]

        col1, col2 = st.columns(2)

        with col1:
            st.subheader(f"Customer Profile: {customer['name']}")
            st.write(f"**Customer ID:** {customer['customer_id']}")
            st.write(f"**Age:** {customer['age']}")
            st.write(f"**Gender:** {customer['gender']}")
            st.write(f"**Location:** {customer['location']}")
            st.write(f"**Loyalty Tier:** {customer['loyalty_tier']}")
            st.write(f"**Signup Date:** {customer['signup_date']}")
            st.write(f"**Segment:** {customer['segment_name']}")

        with col2:
            st.subheader("Customer Metrics")
            st.write(f"**Total Bookings:** {customer['total_bookings']}")
            st.write(f"**Total Spend:** ${customer['total_spend']:.2f}")
            st.write(
                f"**Last Booking:** {customer['last_booking_date'].strftime('%Y-%m-%d')} ({customer['days_since_last_booking']} days ago)")
            st.write(f"**Days Since Login:** {customer['days_since_login']:.0f}")
            st.write(f"**Search Activity (30d):** {customer['search_activity_last_30d']}")
            st.write(f"**Campaign Clicks (90d):** {customer['campaign_clicks_last_90d']}")

            # Calculate and display churn risk with color coding
            churn_risk = customer['churn_risk'] * 100
            risk_color = "red" if churn_risk > 70 else "orange" if churn_risk > 40 else "green"
            st.markdown(f"**Churn Risk:** <span style='color:{risk_color};font-weight:bold'>{churn_risk:.1f}%</span>",
                        unsafe_allow_html=True)

        # Recommend rewards for this customer
        st.subheader("Personalized Reward Recommendations")
        reward = recommend_rewards(customer, rewards_df)

        if reward is not None:
            st.write(f"**Recommended Reward Type:** {reward['reward_type']}")
            st.write(f"**Description:** {reward['description']}")
            st.write(f"**Estimated Value:** ${reward['estimated_value_usd']:.2f}")

            # Recommend additional actions
            st.subheader("Recommended Actions")
            if customer['churn_risk'] > 0.7:
                st.write("1. **High Priority Outreach:** Personal contact from customer service")
                st.write("2. **Special Offer:** Time-limited discount on next booking")
                st.write("3. **Feedback Request:** Survey to understand pain points")
            elif customer['churn_risk'] > 0.4:
                st.write("1. **Email Campaign:** Targeted re-engagement content")
                st.write("2. **Loyalty Reminder:** Summary of available points and rewards")
                st.write("3. **App Notification:** New destinations based on past preferences")
            else:
                st.write("1. **Maintain Relationship:** Regular newsletter and updates")
                st.write("2. **Cross-sell:** Suggestions for complementary travel products")
        else:
            st.write("No specific reward recommendations available.")
    else:
        st.write("No customers found matching that search term.")

# Footer
st.markdown("---")
st.caption("Travel Loyalty & Churn Dashboard. Data is synthetic for demonstration purposes.")