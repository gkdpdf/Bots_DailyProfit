
import time as t
import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
from datetime import datetime
from datetime import time as tt
import seaborn as sns
import numpy as np

from pymongo.mongo_client import MongoClient

#replace username and pwd from mongodb atlas
uri = "mongodb+srv://readb:readb@cluster0.ay8hame.mongodb.net/?retryWrites=true&w=majority"


# Create a new client and connect to the server
client = MongoClient(uri)

# Send a ping to confirm a successful connection
try:
    client.admin.command('ping')
    print("Pinged your deployment. You successfully connected to MongoDB!")
except Exception as e:
    print(e)
st.set_page_config(layout="wide")

t.sleep(5)
# Function to run the main code

def run_main_code():
    import pandas as pd
    st.title("Bots Result")
    col1, col2 = st.columns([5, 5])


    with col1:
        st.header('DNS_BNF :Intraday', divider='rainbow')
        


        # create database 
        mydb = client["DNS_BNF"]

        #create collection
        records =mydb['records']


        ab=records.find()

        df=pd.DataFrame(ab)
        df['Profit'] =df['Profit'].astype(float)
        df['Date String'] = df['Date'].str[:10]


        date_max=df['Date'].iloc[-1][:10]
        t.sleep(4)
        df=df[df['Date String']==date_max]
        t.sleep(3)
        df = df.reset_index(drop=True)
        t.sleep(3)

        import seaborn as sns
        # Seaborn line plot
        fig, ax = plt.subplots(figsize=(30, 20))
        sns.lineplot(data=df,x='Date',y='Profit',color='black')



        # Set x-axis ticks at regular intervals
        
        interval = round(len(df)/8)  # Set the interval you want
        t.sleep(3)
        indices = range(0, len(df['Date'].str.split(" ").str[1]), interval)
        plt.xticks(indices, [df['Date'].str.split(" ").str[1][i] for i in indices],fontsize=20)
        plt.yticks(fontsize=20)
        plt.axhline(max(df['Profit']),color='g',linestyle='dotted')
        plt.axhline(min(df['Profit']),color='r',linestyle='dotted')

        # Fill the area below the line with red if below 0, and with green if above 0
        plt.fill_between(df['Date'], df['Profit'], where=(df['Profit'] < 0), color='red', alpha=0.3, label='Below 0')
        plt.fill_between(df['Date'], df['Profit'], where=(df['Profit'] >= 0), color='green', alpha=0.3, label='Above 0')

        date= "Date of Analysis: " + str(df['Date'].max())[:11]
        plt.xlabel('Time', fontsize=24)
        plt.ylabel('MTM', fontsize=24)
        plt.title(date)

        # Seaborn scatter plot
        st.pyplot(fig)
        current_profit= df['Profit'].tail(1).values[0]
        st.text("Profit is {}".format(current_profit))


        ############### DNS ######################

        st.header('DNS_BNF')
        # create database 
        mydb_dns = client["DNS_BNF"]
        #create collection for DNS
        records =mydb_dns['records']    
        ab=records.find()

        df=pd.DataFrame(ab)
        df['Profit'] =df['Profit'].astype(float)
        df['Date String'] = df['Date'].str[:10]

        # Filter the DataFrame for the desired dates
        dates_to_extract = list(df['Date String'].unique())
        filtered_df = df[df['Date String'].isin(dates_to_extract)]

        # Group by 'Date String' and retrieve the last row for each group
        result = filtered_df.groupby('Date String').last().reset_index()

        # Convert the 'Date' column to datetime format
        result['Date'] = pd.to_datetime(result['Date'], format='%d/%m/%Y %H:%M:%S:%f')

        # Extract day from the datetime column
        result['Day'] = result['Date'].dt.day_name()


        st.dataframe(result[['Date String','Profit','Day']])


    with col2:

        st.header("Ratio_Spread BNF",divider='rainbow')
        # create database 
        mydb = client["Ratio_spread"]

        #create collection
        records =mydb['records']


        ab=records.find()

        df_rs=pd.DataFrame(ab)

        df_rs['Date String'] = df_rs['Date'].str[:10]
        df_rs['Profit'] =df_rs['Profit'].astype(float)

        date_max_rs=df_rs['Date'].iloc[-1][:10]
        df_rs=df_rs[df_rs['Date String']==date_max_rs]
        df_rs = df_rs.reset_index(drop=True)

        

        import seaborn as sns
        # Seaborn line plot
        fig, ax = plt.subplots(figsize=(30, 20))
        sns.lineplot(data=df_rs,x='Date',y='Profit',color='black')



        # Set x-axis ticks at regular intervals
        interval = round(len(df_rs)/3) # Set the interval you want
        indices = range(0, len(df_rs['Date'].str.split(" ").str[1]), interval)
        plt.xticks(indices, [df_rs['Date'].str.split(" ").str[1][i] for i in indices],fontsize=24)
        plt.axhline(max(df_rs['Profit']),color='g',linestyle='dotted')
        plt.axhline(min(df_rs['Profit']),color='r',linestyle='dotted')

        # Fill the area below the line with red if below 0, and with green if above 0
        plt.fill_between(df_rs['Date'], df_rs['Profit'], where=(df_rs['Profit'] < 0), color='red', alpha=0.3, label='Below 0')
        plt.fill_between(df_rs['Date'], df_rs['Profit'], where=(df_rs['Profit'] >= 0), color='green', alpha=0.3, label='Above 0')

        date= "Date of Analysis: " + str(df_rs['Date'].max())[:11]
        plt.ylabel('MTM',fontsize=24)
        plt.title(date)
        # Seaborn scatter plot
        st.pyplot(fig)
        current_profitrs= df_rs['Profit'].tail(1).values[0]
        st.text("Profit is {}".format(current_profitrs))
        

      

        ############### Ratio Spread ######################

        st.header('Ratio Spread_BNF :Intraday')
        # create database 
        mydb_rs = client["Ratio_spread"]
        #create collection for DNS
        records =mydb_rs['records']    
        cd=records.find()

        df_rs=pd.DataFrame(cd)
        df_rs['Profit'] =df_rs['Profit'].astype(float)
        df_rs['Date String'] = df_rs['Date'].str[:10]

        # Filter the DataFrame for the desired dates
        dates_to_extract = list(df_rs['Date String'].unique())
        filtered_df_rs = df_rs[df_rs['Date String'].isin(dates_to_extract)]

        # Group by 'Date String' and retrieve the last row for each group
        result_rs = filtered_df_rs.groupby('Date String').last().reset_index()

        # Convert the 'Date' column to datetime format
        result_rs['Date'] = pd.to_datetime(result_rs['Date'], format='%d/%m/%Y %H:%M:%S:%f')

        # Extract day from the datetime column
        result_rs['Day'] = result_rs['Date'].dt.day_name()
        result_rs= result_rs[['Date String','Profit','Day']]


        st.dataframe(result_rs[['Date String','Profit','Day']])


def total_profit_correlation():
    col1, col2 = st.columns([5, 5])

    with col1:
    
        ############### DNS ######################
        st.header('Total Profit', divider='rainbow')
        # create database 
        mydb_dns = client["DNS_BNF"]
        #create collection for DNS
        records =mydb_dns['records']    
        ab=records.find()

        df=pd.DataFrame(ab)
        df['Profit'] =df['Profit'].astype(float)
        df['Date String'] = df['Date'].str[:10]

        # Filter the DataFrame for the desired dates
        dates_to_extract = list(df['Date String'].unique())
        filtered_df = df[df['Date String'].isin(dates_to_extract)]

        # Group by 'Date String' and retrieve the last row for each group
        result = filtered_df.groupby('Date String').last().reset_index()

        # Convert the 'Date' column to datetime format
        result['Date'] = pd.to_datetime(result['Date'], format='%d/%m/%Y %H:%M:%S:%f')

        # Extract day from the datetime column
        result['Day'] = result['Date'].dt.day_name()
        result=result[['Date String','Profit','Day']]

        #st.dataframe(result[['Date String','Profit','Day']])   


        ############### Ratio Spread ######################

        #st.header('Ratio Spread_BNF')
        # create database 
        mydb_rs = client["Ratio_spread"]
        #create collection for DNS
        records =mydb_rs['records']    
        cd=records.find()

        df_rs=pd.DataFrame(cd)
        df_rs['Profit'] =df_rs['Profit'].astype(float)
        df_rs['Date String'] = df_rs['Date'].str[:10]

        # Filter the DataFrame for the desired dates
        dates_to_extract = list(df_rs['Date String'].unique())
        filtered_df_rs = df_rs[df_rs['Date String'].isin(dates_to_extract)]

        # Group by 'Date String' and retrieve the last row for each group
        result_rs = filtered_df_rs.groupby('Date String').last().reset_index()

        # Convert the 'Date' column to datetime format
        result_rs['Date'] = pd.to_datetime(result_rs['Date'], format='%d/%m/%Y %H:%M:%S:%f')

        # Extract day from the datetime column
        result_rs['Day'] = result_rs['Date'].dt.day_name()
        result_rs= result_rs[['Date String','Profit','Day']]


        #st.dataframe(result_rs[['Date String','Profit','Day']])
    
        # create database 
        mydb_dns = client["DNS_BNF"]
        #create collection for DNS
        records =mydb_dns['records']    
        ab=records.find()

        df=pd.DataFrame(ab)
        df['Profit'] =df['Profit'].astype(float)
        df['Date String'] = df['Date'].str[:10]

        dates_to_extract = list(df['Date String'].unique())

        # Merge the two DataFrames on the basis of the 'date' and 'datestring' columns
        merged_df = pd.merge(result, result_rs, left_on='Date String', right_on='Date String', suffixes=('_dns', '_rs'))
        merged_df = merged_df[['Date String','Profit_dns','Profit_rs','Day_dns']]
        merged_df['Total Profit']=merged_df['Profit_dns']+merged_df['Profit_rs']

         # Method 1: Using st.line_chart
        st.line_chart(merged_df.drop('Day_dns',axis=1).set_index('Date String'),color=["#fd0", "#f0f", "#04f"])

        st.dataframe(merged_df)

    
    with col2:
        st.header('Correlation Matrix', divider='rainbow')
        import plotly.express as px
        a=pd.DataFrame(merged_df[['Profit_dns','Profit_rs']].corr())
        fig = px.imshow(merged_df[['Profit_dns','Profit_rs']].corr(), text_auto=True)

        tab1, tab2 = st.tabs(["Streamlit theme (default)", "Plotly native theme"])
        with tab1:
            st.plotly_chart(fig, theme="streamlit")
        with tab2:
            st.plotly_chart(fig, theme=None)


def Accuracy():
    col1, col2 = st.columns([5, 5])

    with col1:
    
        ############### DNS ######################
        st.header('Total Profit', divider='rainbow')
        # create database 
        mydb_dns = client["DNS_BNF"]
        #create collection for DNS
        records =mydb_dns['records']    
        ab=records.find()

        df=pd.DataFrame(ab)
        df['Profit'] =df['Profit'].astype(float)
        df['Date String'] = df['Date'].str[:10]

        # Filter the DataFrame for the desired dates
        dates_to_extract = list(df['Date String'].unique())
        filtered_df = df[df['Date String'].isin(dates_to_extract)]

        # Group by 'Date String' and retrieve the last row for each group
        result = filtered_df.groupby('Date String').last().reset_index()

        # Convert the 'Date' column to datetime format
        result['Date'] = pd.to_datetime(result['Date'], format='%d/%m/%Y %H:%M:%S:%f')

        # Extract day from the datetime column
        result['Day'] = result['Date'].dt.day_name()
        result=result[['Date String','Profit','Day']]

        #st.dataframe(result[['Date String','Profit','Day']])   


        ############### Ratio Spread ######################

        #st.header('Ratio Spread_BNF')
        # create database 
        mydb_rs = client["Ratio_spread"]
        #create collection for DNS
        records =mydb_rs['records']    
        cd=records.find()

        df_rs=pd.DataFrame(cd)
        df_rs['Profit'] =df_rs['Profit'].astype(float)
        df_rs['Date String'] = df_rs['Date'].str[:10]

        # Filter the DataFrame for the desired dates
        dates_to_extract = list(df_rs['Date String'].unique())
        filtered_df_rs = df_rs[df_rs['Date String'].isin(dates_to_extract)]

        # Group by 'Date String' and retrieve the last row for each group
        result_rs = filtered_df_rs.groupby('Date String').last().reset_index()

        # Convert the 'Date' column to datetime format
        result_rs['Date'] = pd.to_datetime(result_rs['Date'], format='%d/%m/%Y %H:%M:%S:%f')

        # Extract day from the datetime column
        result_rs['Day'] = result_rs['Date'].dt.day_name()
        result_rs= result_rs[['Date String','Profit','Day']]


        #st.dataframe(result_rs[['Date String','Profit','Day']])
    
        # create database 
        mydb_dns = client["DNS_BNF"]
        #create collection for DNS
        records =mydb_dns['records']    
        ab=records.find()

        df=pd.DataFrame(ab)
        df['Profit'] =df['Profit'].astype(float)
        df['Date String'] = df['Date'].str[:10]

        dates_to_extract = list(df['Date String'].unique())

        # Merge the two DataFrames on the basis of the 'date' and 'datestring' columns
        merged_df = pd.merge(result, result_rs, left_on='Date String', right_on='Date String', suffixes=('_dns', '_rs'))
        merged_df = merged_df[['Date String','Profit_dns','Profit_rs','Day_dns']]
        merged_df['Total Profit']=merged_df['Profit_dns']+merged_df['Profit_rs']

        def replace_negative_with_zero(value):
            return 0 if value < 0 else 1

        result['Accuracy']=result['Profit'].apply(replace_negative_with_zero)
        a=result['Accuracy'].sum()
        b=len(result)

        Accuracy_dns= (a/b)

        result_rs['Accuracy']=result_rs['Profit'].apply(replace_negative_with_zero)
        c=result_rs['Accuracy'].sum()
        d=len(result_rs)

        Accuracy_rs= (c/d)

        accuracy_matrix=pd.DataFrame({"Bots":["DNS_BNF","Ratio_BNF"]
                                      ,"Accuracy":[Accuracy_dns,Accuracy_rs] 
                                      })
        
        import plotly.express as px
        #data_canada = px.data.gapminder().query("country == 'Canada'")
        fig = px.bar(accuracy_matrix, x='Bots', y='Accuracy')

        tab1, tab2 = st.tabs(["Streamlit theme (default)", "Plotly native theme"])
        with tab1:
            st.plotly_chart(fig, theme="streamlit")
        with tab2:
            st.plotly_chart(fig, theme=None)

        st.dataframe(accuracy_matrix)

    with col2:
        st.header('Total Points captured by bots', divider='rainbow')
        from plotly import graph_objects as go

        Net_Profit_dns= result['Profit'].sum()
        Net_Profit_rs=result_rs['Profit'].sum()


        fig = go.Figure(go.Funnel(
        y = ["DNS_BNF", "Ratio_Spread_BNF"],
        x = [Net_Profit_dns, Net_Profit_rs]))


        tab1, tab2 = st.tabs(["Streamlit theme (default)", "Plotly native theme"])
        with tab1:
            st.plotly_chart(fig, theme="streamlit")
        with tab2:
            st.plotly_chart(fig, theme=None)

        a=pd.DataFrame({"Bots":["DNS_BNF", "Ratio_Spread_BNF"],
                        "Profit":[Net_Profit_dns, Net_Profit_rs]})
        st.dataframe(a)
        
# Initial run
current_time = datetime.now().time()
run_main_code()
st.divider()
st.divider()
total_profit_correlation()
st.divider()
st.divider()
Accuracy()


# Auto-refresh every 10 seconds
while True and current_time <=tt(9,55) and current_time >=tt(3,55):
    t.sleep(100)
    # Initial run
    current_time = datetime.now().time()
    st.rerun()
