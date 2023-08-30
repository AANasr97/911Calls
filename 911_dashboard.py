
import streamlit as st
import pandas as pd
import plotly.express as px

df = pd.read_csv('911.csv')
df.drop_duplicates(inplace = True , ignore_index=True)
df.reset_index(drop = True, inplace = True)
df = df.dropna(subset=['twp'])

# Get unique township values from the 'twp' column
unique_townships = df['twp'].unique()
# Create a dictionary mapping township to zip code
township_to_zip_mapping = {twp: None for twp in unique_townships}
# Fill the dictionary with zip codes
for twp in unique_townships:
    zip_code = None  # Initialize zip_code as None
    rows_with_twp = df[df['twp'] == twp]  # Filter rows with the same twp
    for index, row in rows_with_twp.iterrows():
        if pd.notna(row['zip']):  # Check if zip is not NaN
            zip_code = row['zip']
            break  # Stop searching once a non-null zip is found
    township_to_zip_mapping[twp] = zip_code
# Fill missing zip codes based on township
df['zip'] = df.apply(lambda row: township_to_zip_mapping.get(row['twp'], row['zip']), axis=1)

df = df.drop(columns=['e'])
df['zip'] = df['zip'].astype(int).astype(str)

df['timeStamp'] = pd.to_datetime(df['timeStamp'])

df['Department'] = df['title'].apply(lambda title: title.split(':')[0])
df['Reason'] = df['title'].apply(lambda title: title.split(':')[1])

df['Date'] = df['timeStamp'].dt.date
df['Hour'] = df['timeStamp'].apply(lambda time: time.hour)
df['Day'] = df['timeStamp'].dt.day
df['nDay_of_Week'] = df['timeStamp'].apply(lambda time: time.dayofweek)
dmap = {0:'Mon',1:'Tue',2:'Wed',3:'Thu',4:'Fri',5:'Sat',6:'Sun'}
df['Day_of_Week'] = df['nDay_of_Week'].map(dmap)
df['Month'] = df['timeStamp'].apply(lambda time: time.month)
df['Year'] = df['timeStamp'].apply(lambda time: time.year)

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
