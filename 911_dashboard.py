
import streamlit as st
import pandas as pd
import plotly.express as px

df = pd.read_csv('911.csv')

# Set page title
st.title("911 Call Data Analysis Dashboard")

# Sidebar options
st.sidebar.header("Dashboard Options")

# Filter by Department
selected_department = st.sidebar.selectbox("Select Department", ['All'] + df['Department'].unique())
if selected_department != 'All':
    df = df[df['Department'] == selected_department]

# Filter by Reason
selected_reason = st.sidebar.selectbox("Select Reason", ['All'] + df['Reason'].unique())
if selected_reason != 'All':
    df = df[df['Reason'] == selected_reason]

# Visualizations
st.header("Incident Distribution by Department and Reason")
fig_reasons = px.histogram(df, x='Department', color='Reason', labels={'Department': 'Department', 'Reason': 'Reason'},
                           title='Distribution of Incidents by Department and Reason')
st.plotly_chart(fig_reasons)

st.header("Incident Counts by Date")
# Convert the 'timeStamp' column to datetime
df['timeStamp'] = pd.to_datetime(df['timeStamp'])
# Extract the date from the timestamp
df['Date'] = df['timeStamp'].dt.date
# Group data by date and calculate incident counts
byDate = df.groupby('Date').count()['twp'].reset_index()
# Create an interactive line plot using Plotly Express
fig_date = px.line(byDate, x='Date', y='twp', labels={'Date': 'Date', 'twp': 'Incident Counts'},
                   title='Incident Counts by Date')
st.plotly_chart(fig_date)

# Additional visualizations can be added here

# Show footer
st.sidebar.text("Created with Streamlit by Ahmed NasrElDin")
