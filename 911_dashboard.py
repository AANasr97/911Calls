
import streamlit as st
import pandas as pd
import plotly.express as px

st.set_page_config(layout='wide',
                  page_title = '911 Call Data Analysis Dashboard')

df = pd.read_csv('M911.csv')

def main_page():
    # Set page title
    st.title("911 Call Data Analysis Dashboard")

    st.image('Police.png')

    # Project Overview
    st.markdown("""
                
    ## Overview

    ### Understanding the Background

    * Montgomery County

    Montgomery County, locally also referred to as Montco, is a county located in the Commonwealth of Pennsylvania. As of the 2010 census, the population was 799,874, making it the third-most populous county in Pennsylvania, after Philadelphia and Allegheny Counties. The county seat is Norristown. Montgomery County is very diverse, ranging from farms and open land in Upper Hanover to densely populated rowhouse streets in Cheltenham.

    * 911 Calls

    Created by Congress in 2004 as the 911 Implementation and Coordination Office (ICO), the National 911 Program is housed within the National Highway Traffic Safety Administration at the U.S. Department of Transportation and is a joint program with the National Telecommunication and Information Administration in the Department of Commerce.

    ### Goal:

    * Locations from which 911 calls are most frequent
    * Time daily, month, weekly patterns of 911 calls
    * Major Causes of 911 calls


    **This analysis will help to deploy more agents in specific location and save/help people at right time**

    ---
    ----

    ### The Data

    `Acknowledgements`: Data provided by  <a href='http://www.pieriandata.com'>montcoalert.org</a>

    we will be analyzing some 911 call data from [Kaggle](https://www.kaggle.com/mchirico/montcoalert). The data contains the following fields:

    Column | Definition
    --- | -----------
    lat | Latitude
    lng | Longitude
    desc | Description of the Emergency Call
    zip | Zipcode
    title | Title of Emergency
    timeStamp | YYYY-MM-DD HH:MM:SS
    twp | Township
    addr | Address
    e | Dummy variable (always 1)


                """)

    st.markdown("""
    ---
    ----
    """)

    datasample = st.checkbox('Show Data Sample', False, key =1)
    if datasample:
        st.dataframe(df.head())

def page2():
    df = pd.read_csv('M911.csv')
    # Sidebar options
    st.sidebar.header("Dashboard Options")

    # Filter by Department
    selected_department = st.sidebar.selectbox("Select Department", ['All'] + df['Department'].unique().tolist())
    if selected_department != 'All':
        df = df[df['Department'] == selected_department]

    # Filter by Reason
    selected_reason = st.sidebar.selectbox("Select Reason", ['All'] + df['Reason'].unique().tolist())
    if selected_reason != 'All':
        df = df[df['Reason'] == selected_reason]

    # Visualizations
    st.header("Incident Distribution by Department and Reason")
    fig_reasons = px.histogram(df, x='Department', color='Reason', labels={'Department': 'Department', 'Reason': 'Reason'},
                            title='Distribution of Incidents by Department and Reason')
    st.plotly_chart(fig_reasons)

    st.header("Incident Counts by Date")
    # Group data by date and calculate incident counts
    byDate = df.groupby('Date').count()['twp'].reset_index()
    # Create an interactive line plot using Plotly Express
    fig_date = px.line(byDate, x='Date', y='twp', labels={'Date': 'Date', 'twp': 'Incident Counts'},
                    title='Incident Counts by Date')
    st.plotly_chart(fig_date)
    
def page3():
    df = pd.read_csv('M911.csv')
    # Additional visualizations can be added here
    df.rename(columns={'lng': 'lon'}, inplace=True)
    map_data = df[['lat', 'lon']]

    st.map(map_data)

pages = { '1':main_page,
          '2':page2,
          '3': page3
}    

select_page = st.sidebar.selectbox('select ur page',pages.keys())
pages[select_page]()


# Show footer
st.sidebar.text("Created with Streamlit by Ahmed NasrElDin & Gehad Samir")
