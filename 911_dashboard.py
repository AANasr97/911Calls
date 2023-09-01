
import pandas as pd
import numpy as np
import statsmodels.api as sm
import matplotlib.pyplot as plt
import plotly.express as px
import seaborn as sns
from datetime import datetime
import os
import streamlit as st


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
    
    st.markdown("""
        ### Lets take a look on both Numerical and Categorical Features. 
        """)
    
    st.subheader("""Numerical Summary""")
    
    numerical_columns = ['lat', 'lng', 'Hour', 'Day', 'nDay_of_Week', 'Month', 'Year']
    numerical_summary = df[numerical_columns].describe()
    st.dataframe(numerical_summary)

    # Categorical columns in the DataFrame
    categorical_columns = ['title', 'twp', 'Department', 'Reason', 'Day_of_Week']

    # Create a dictionary to store class balance DataFrames
    class_balance_dfs = {}

    # Calculate class balance for each categorical column
    for column in categorical_columns:
        class_balance = df[column].value_counts(normalize=True).head(7).reset_index()
        class_balance.columns = [column, 'Class Balance']
        class_balance_dfs[column] = class_balance
    st.markdown("""
        ---
        ----
        """)
    st.subheader("""Categorical Summary""")
    # Calculate class balance for each categorical column and create interactive bar charts
    for column in categorical_columns:
        class_balance = df[column].value_counts(normalize=True).head(7).reset_index()
        class_balance.columns = [column, 'Class Balance']

        # Create an interactive bar chart using Plotly Express
        fig = px.bar(class_balance, x=column, y='Class Balance',
                    labels={column: column, 'Class Balance': 'Class Balance'},
                    title=f'{column}')
        
        # Display the DataFrame and the associated chart together for each column
        
        st.markdown(f"#### {column}")
        st.dataframe(class_balance)
        st.plotly_chart(fig)
        st.markdown("""
        ---
        ----
        """)

def page3():
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
    # Filter by Year
    selected_year = st.sidebar.selectbox("Select Year", ['All'] + df['Year'].unique().tolist())
    if selected_year != 'All':
        df = df[df['Year'] == selected_year]
    # Filter by Month
    selected_month = st.sidebar.selectbox("Select Month", ['All'] + sorted(df['Month'].unique().tolist()))
    if selected_month != 'All':
        df = df[df['Month'] == selected_month]
    # Filter by Year
    selected_day_of_week = st.sidebar.selectbox("Select Day Of Week", ['All'] + df['Day_of_Week'].unique().tolist())
    if selected_day_of_week != 'All':
        df = df[df['Day_of_Week'] == selected_day_of_week]

    showgraph = st.checkbox('Show Graphs', False, key =1)
    
    if showgraph:
        # Visualizations
        st.subheader("Incident Distribution by Department and Reason")
        # Calculate the incident distribution by department and reason
        incident_distribution = df.groupby(['Department', 'Reason']).size().reset_index(name='Count')
        
        # Create a sunburst chart using Plotly Express
        fig = px.sunburst(incident_distribution, path=['Department', 'Reason'], values='Count', title='Incident Distribution by Department and Reason')
        st.plotly_chart(fig)
        
        st.subheader("Incident Distribution by TownShip")
        # Calculate the incident distribution by 'twp'
        incident_distribution_twp = df.groupby(['twp']).size().reset_index(name='Count')
        # Create a sunburst chart for incident distribution by 'twp' using Plotly Express
        fig_twp = px.sunburst(incident_distribution_twp, path=['twp'], values='Count', title='Incident Distribution by Township')
        st.plotly_chart(fig_twp)

def page4():
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
    # Filter by Year
    selected_year = st.sidebar.selectbox("Select Year", ['All'] + df['Year'].unique().tolist())
    if selected_year != 'All':
        df = df[df['Year'] == selected_year]
    # Filter by Month
    selected_month = st.sidebar.selectbox("Select Month", ['All'] + sorted(df['Month'].unique().tolist()))
    if selected_month != 'All':
        df = df[df['Month'] == selected_month]
    # Filter by Year
    selected_day_of_week = st.sidebar.selectbox("Select Day Of Week", ['All'] + df['Day_of_Week'].unique().tolist())
    if selected_day_of_week != 'All':
        df = df[df['Day_of_Week'] == selected_day_of_week]

    showgraph = st.checkbox('Show Graphs', False, key =1)
    
    if showgraph:
        df['twp'].value_counts().head(5)

        # Calculate incident counts for the top 5 Townships
        top_Townships = df['twp'].value_counts().head(5).reset_index()
        top_Townships.columns = ['Township', 'Incident Counts']

        # Create an interactive bar chart using Plotly Express
        fig = px.bar(top_Townships, x='Township', y='Incident Counts', labels={'Township': 'Township', 'Incident Counts': 'Incident Counts'},
                    title='Top 5 Townships with Highest Incident Counts')
        st.plotly_chart(fig)
        
        st.markdown("""
        **Are there specific time periods with unusually high or low incident counts?**
        """)
        st.code('''
        # Group data by hour (you can adjust this to day, week, etc.)
        incident_counts_by_hour = df.groupby(df['timeStamp'].dt.hour).size()

        # Calculate mean and standard deviation of incident counts
        mean_incidents = incident_counts_by_hour.mean()
        std_incidents = incident_counts_by_hour.std()

        # Set a threshold for identifying outliers
        threshold = mean_incidents + 2 * std_incidents

        # Identify time periods with unusually high or low incident counts
        outliers = incident_counts_by_hour[(incident_counts_by_hour > threshold) | (incident_counts_by_hour < mean_incidents - std_incidents)]
        ''')
        df['timeStamp'] = pd.to_datetime(df['timeStamp'])

        # Group data by hour (you can adjust this to day, week, etc.)
        incident_counts_by_hour = df.groupby(df['timeStamp'].dt.hour).size()

        # Calculate mean and standard deviation of incident counts
        mean_incidents = incident_counts_by_hour.mean()
        std_incidents = incident_counts_by_hour.std()

        # Set a threshold for identifying outliers (you can adjust this)
        threshold = mean_incidents + 2 * std_incidents

        # Identify time periods with unusually high or low incident counts
        outliers = incident_counts_by_hour[(incident_counts_by_hour > threshold) | (incident_counts_by_hour < mean_incidents - std_incidents)]

        # Create a bar plot with outliers highlighted
        fig, ax = plt.subplots(figsize=(10, 6))
        ax.bar(incident_counts_by_hour.index, incident_counts_by_hour.values, color='blue', alpha=0.6)
        ax.bar(outliers.index, outliers.values, color='red', alpha=0.6)
        ax.set_xlabel('Hour of the Day')
        ax.set_ylabel('Incident Counts')
        ax.set_title('Incident Counts by Hour of the Day')
        ax.legend(['Incident Counts', 'Outliers'])
        ax.set_xticks(incident_counts_by_hour.index)
        ax.set_xticklabels(incident_counts_by_hour.index, rotation=0)

        # Display the plot using Streamlit
        st.pyplot(fig)

        # Calculate the count of each category
        department_counts = df['Department'].value_counts()
        # Convert to a DataFrame
        department_counts_df = pd.DataFrame({'Department': department_counts.index, 'Count': department_counts.values})
        # Create an interactive bar chart using Plotly Express
        fig = px.bar(department_counts_df, x='Department', y='Count', color='Department', 
                    color_discrete_sequence=px.colors.qualitative.Pastel, title='Incident Counts by Department')
        # Display the plot using Streamlit
        st.plotly_chart(fig)

        # Calculate the count of incidents for each department and day
        department_day_counts = df.groupby(['Day_of_Week', 'Department']).size().reset_index(name='Count')

        # Create a bar chart using Plotly Express
        fig = px.bar(department_day_counts, x='Day_of_Week', y='Count', color='Department', barmode='group', 
                    color_discrete_sequence=px.colors.qualitative.Pastel, title='Incident Counts by Department and Day')

        # Add text labels for the counts on the bars
        fig.update_traces(texttemplate='%{text}', textposition='outside')

        # Relocate the legend
        fig.update_layout(legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1))

        # Display the plot using Streamlit
        st.plotly_chart(fig)


        # Calculate the count of incidents for each department and month
        department_month_counts = df.groupby(['Month', 'Department']).size().reset_index(name='Count')

        # Create a bar chart using Plotly Express
        fig = px.bar(department_month_counts, x='Month', y='Count', color='Department', barmode='group', 
                    color_discrete_sequence=px.colors.qualitative.Pastel, title='Incident Counts by Department and Month')

        # Set the x-axis labels to months
        fig.update_layout(
            xaxis=dict(
                tickmode='array',
                tickvals=list(range(1, 13)),
                ticktext=[
                    'Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun',
                    'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec'
                ]
            )
        )

        # Add text labels for the counts on the bars
        fig.update_traces(texttemplate='%{text}', textposition='outside')

        # Relocate the legend
        fig.update_layout(legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1))

        # Display the plot using Streamlit
        st.plotly_chart(fig)
        
        # Calculate the count of incidents for each department and year
        department_year_counts = df.groupby(['Year', 'Department']).size().reset_index(name='Count')

        # Create a bar chart using Plotly Express
        fig = px.bar(department_year_counts, x='Year', y='Count', color='Department', barmode='group', 
                    color_discrete_sequence=px.colors.qualitative.Pastel, title='Incident Counts by Department and Year')

        # Add text labels for the counts on the bars
        fig.update_traces(texttemplate='%{text}', textposition='outside')

        # Relocate the legend
        fig.update_layout(legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1))

        # Display the plot using Streamlit
        st.plotly_chart(fig)

def page5():
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
    # Filter by Year
    selected_year = st.sidebar.selectbox("Select Year", ['All'] + df['Year'].unique().tolist())
    if selected_year != 'All':
        df = df[df['Year'] == selected_year]
    # Filter by Month
    selected_month = st.sidebar.selectbox("Select Month", ['All'] + sorted(df['Month'].unique().tolist()))
    if selected_month != 'All':
        df = df[df['Month'] == selected_month]
    # Filter by Year
    selected_day_of_week = st.sidebar.selectbox("Select Day Of Week", ['All'] + df['Day_of_Week'].unique().tolist())
    if selected_day_of_week != 'All':
        df = df[df['Day_of_Week'] == selected_day_of_week]

    showgraph = st.checkbox('Show Graphs', False, key =1)
    
    if showgraph:
        st.subheader("Incident Counts by Date")
        # Group data by date and calculate incident counts
        byDate = df.groupby('Date').count()['twp'].reset_index()
        # Convert 'Date' column to datetime type
        byDate['Date'] = pd.to_datetime(byDate['Date'])
        # Create an area chart using Plotly Express
        fig = px.area(byDate, x='Date', y='twp', labels={'Date': 'Date', 'twp': 'Incident Counts'},
                    title='Incident Counts by Date')
        st.plotly_chart(fig)
        
        st.subheader("Incident Chart")
        # Convert 'timeStamp' column to datetime format
        df['timeStamp'] = pd.to_datetime(df['timeStamp'])
        
        # Group data by hour and calculate incident counts
        byHour = df.groupby('Hour').size().reset_index(name='Count')
        # Create a line chart time series for hourly counts using Plotly Express
        fig_hourly = px.line(byHour, x='Hour', y='Count', labels={'Hour': 'Hour of Day', 'Count': 'Incident Counts'},
                            title='Hourly Incident Counts Time Series')
        st.plotly_chart(fig_hourly)
        
        # Group data by hour and calculate incident counts
        byDay = df.groupby('Day_of_Week').size().reset_index(name='Count')
        # Create a line chart time series for hourly counts using Plotly Express
        fig_daily = px.line(byDay, x='Day_of_Week', y='Count', labels={'Day_of_Week': 'Day of Week', 'Count': 'Incident Counts'},
                            title='Daily Incident Counts Time Series')
        st.plotly_chart(fig_daily)

        # Group data by month
        byMonth = df.groupby('Month').count().reset_index()

        # Create an interactive scatter plot using Plotly Express
        fig = px.scatter(byMonth, x='Month', y='twp', title='Incident Counts by Month')
        # Customize the scatter plot appearance
        fig.update_traces(marker=dict(color='blue'), mode='markers')
        # Display the plot using Streamlit
        st.plotly_chart(fig)


        # Group data by township and month
        byTownshipMonth = df.groupby(['twp', 'Month']).count().reset_index()

        # Create a list to store the generated scatter plots
        scatter_plots = []

        # Create individual linear regression plots for each numeric township
        numeric_townships = byTownshipMonth['twp'].loc[byTownshipMonth['twp'].apply(lambda x: x.isnumeric())].unique()
        for township in numeric_townships:
            township_data = byTownshipMonth[byTownshipMonth['twp'] == township]
            
            # Fit a linear regression model
            sns.lmplot(x='Month', y='twp', data=township_data)
            plt.title(f'Linear Regression for {township}')
            scatter_plots.append(plt.gcf())
            plt.clf()

        # Create an interactive scatter plot using Plotly Express
        scatter_fig = px.scatter(byMonth, x='Month', y='twp', title='Incident Counts by Month')
        scatter_fig.update_traces(mode='lines+markers', marker=dict(color='blue'), line=dict(color='red'))

        # Display the individual linear regression plots and scatter plot in the Streamlit app
        st.write("Individual Linear Regression Plots by Township")
        for idx, scatter_plot in enumerate(scatter_plots):
            st.pyplot(scatter_plot, caption=f'Township: {numeric_townships[idx]}')
        st.write("Incident Counts by Month")
        st.plotly_chart(scatter_fig)

        # Group data by month
        byMonth = df.groupby('Month').count().reset_index()
        # Create an lmplot using seaborn
        sns.set(style="whitegrid")
        plt.figure(figsize=(10, 6))
        lmplot = sns.lmplot(x='Month', y='twp', data=byMonth)
        plt.title('Linear Regression Plot of Incident Counts by Month')
        # Display the lmplot using Streamlit
        st.pyplot(lmplot)
    
    

    # Filter data for the desired date range (from January 2015 to December 2020)
    start_date = '2015-01-01'
    end_date = '2020-12-31'
    df['timeStamp'] = pd.to_datetime(df['timeStamp'])
    filtered_df = df[(df['timeStamp'] >= start_date) & (df['timeStamp'] <= end_date)]

    # Extract year and month from the 'timeStamp' column
    filtered_df['Year'] = filtered_df['timeStamp'].dt.year
    filtered_df['Month'] = filtered_df['timeStamp'].dt.month

    # Group data by year and month and count incidents
    incident_counts = filtered_df.groupby(['Year', 'Month']).size().reset_index(name='Incident Counts')

    # Create a scatter plot using Plotly Express
    fig = px.scatter(incident_counts, x=incident_counts['Year'].astype(str) + '-' + incident_counts['Month'].astype(str).str.zfill(2), y='Incident Counts',
                    title='Scatter Plot of Incident Counts by Month (2015-2020)', labels={'x': 'Month (Year-Month)', 'Incident Counts': 'Incident Counts'})
    fig.update_xaxes(type='category')

    # Display the scatter plot using Streamlit
    st.plotly_chart(fig)






def page6():
    df = pd.read_csv('M911.csv')
    # Additional visualizations can be added here
    df.rename(columns={'lng': 'lon'}, inplace=True)
    map_data = df[['lat', 'lon']]

    st.map(map_data)

pages = { 'Overview':main_page,
          'Data Describe':page2,
          'Data Distribution': page3,
          'EDA': page4,
          'Predict': page5,
          'Mapping': page6
}    

select_page = st.sidebar.selectbox('Page Select',pages.keys())
pages[select_page]()

# Show footer
st.sidebar.text("Created with Streamlit by Ahmed NasrElDin & Gehad Samir")
