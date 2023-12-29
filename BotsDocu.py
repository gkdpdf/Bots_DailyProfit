
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

# create database 
mydb = client["DNS_BNF"]

#create collection
records =mydb['records']

ab=records.find()

df=pd.DataFrame(ab)
date_max=df['Date'].iloc[-1][:10]

st.title(f"Summary of BankNifty Bots till {date_max}")



def summary_trend_roi():
    col1, col2 = st.columns([5, 5])
    # create database 
    mydb_dns = client["DNS_BNF"]
    #create collection for DNS
    records =mydb_dns['records']    
    ab=records.find()

    
    df=pd.DataFrame(ab)
    df['Profit'] =df['Profit'].astype(float)
    df['Date String'] = df['Date'].str[:10]

   

    # Group by 'Date' and replace values as needed
    df['Profit'] = df.groupby('Date String')['Profit'].transform(lambda x: -60 if any(x < -60) else x)

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



    # Group by 'Date' and replace values as needed
    df_rs['Profit'] = df_rs.groupby('Date String')['Profit'].transform(lambda x: -60 if any(x < -60) else x)

    # Filter the DataFrame for the desired damtes
    dates_to_extract = list(df_rs['Date String'].unique())
    filtered_df_rs = df_rs[df_rs['Date String'].isin(dates_to_extract)]

    # Group by 'Date String' and retrieve the last row for each group
    result_rs = filtered_df_rs.groupby('Date String').last().reset_index()

    # Convert the 'Date' column to datetime format
    result_rs['Date'] = pd.to_datetime(result_rs['Date'], format='%d/%m/%Y %H:%M:%S:%f')

    # Extract day from the datetime column
    result_rs['Day'] = result_rs['Date'].dt.day_name()
    result_rs= result_rs[['Date String','Profit','Day']]
    result_rs['Bot Name']='Ratio Spread'

    #st.dataframe(result_rs[['Date String','Profit','Day']])

    # create database 
    mydb_dns = client["DNS_BNF"]
    #create collection for DNS
    records =mydb_dns['records']    
    ab=records.find()

    df=pd.DataFrame(ab)
    df['Profit'] =df['Profit'].astype(float)
    df['Date String'] = df['Date'].str[:10]


    # Group by 'Date' and replace values as needed
    df['Profit'] = df.groupby('Date String')['Profit'].transform(lambda x: -60 if any(x < -60) else x)

    dates_to_extract = list(df['Date String'].unique())

    # Merge the two DataFrames on the basis of the 'date' and 'datestring' columns
    merged_df = pd.merge(result, result_rs, left_on='Date String', right_on='Date String', suffixes=('_dns', '_rs'))
    merged_df = merged_df[['Date String','Profit_dns','Profit_rs','Day_dns']]
    merged_df['Total Profit']=merged_df['Profit_dns']+merged_df['Profit_rs']
    merged_df['Cumulative Profit']=merged_df['Total Profit'].cumsum()
    merged_df['ROI % Intraday']=(merged_df['Total Profit']*15/170000)*100
    merged_df['ROI % till now']=(merged_df['Cumulative Profit']*15/170000)*100

    with col2:
    
        
        import plotly.express as px
        
        fig = px.line(merged_df, x='Date String', y=['ROI % Intraday','ROI % till now'], markers=True)

        tab1, tab2 = st.tabs(["Streamlit theme (default)", "Plotly native theme"])
        with tab1:
            st.plotly_chart(fig, theme="streamlit")
        with tab2:
            st.plotly_chart(fig, theme=None)

        #st.dataframe(merged_df)

    
    with col1:
        df_summ = pd.DataFrame({"Bot_Name":['DNS_BNF','Ratio_spread_BNF','Straddle_BNF'],
                            "Instrument" : ['Bank Nifty','Bank Nifty','Bank Nifty'],
                            "Margin" : ['50K','120K','120K']})
        st.dataframe(df_summ)
        import plotly.express as px
        df = px.data.tips()
        fig = px.bar(merged_df, x="Total Profit", y="Day_dns",orientation='h',text_auto=True)
        
        # Customize data labels
       # fig.update_traces(textposition='outside', insidetextfont=dict(color='black'))

        # Customize bar color
        fig.update_traces( marker_color='rgba(191, 230, 221, 0.8)')  # Pinkish color with alpha transparency

        # Set layout properties
        fig.update_layout(title='Horizontal Bar Chart with Customized Data Labels',
                        xaxis_title='Profit',
                        yaxis_title='Day')
        st.plotly_chart(fig, theme=None)


def run_main_code():
    import pandas as pd
    
    col1, col2 = st.columns([5, 5])

    with col1:
        st.text('Intraday:DNS')
        
        # create database 
        mydb = client["DNS_BNF"]

        #create collection
        records =mydb['records']


        ab=records.find()

        df=pd.DataFrame(ab)
        df['Profit'] =df['Profit'].astype(float)
        df['Date String'] = df['Date'].str[:10]

 

        # Group by 'Date' and replace values as needed
        df['Profit'] = df.groupby('Date String')['Profit'].transform(lambda x: -60 if any(x < -60) else x)

        df['Time']=df['Date'].str[10:16]


        date_max=df['Date'].iloc[-1][:10]
        
        df=df[df['Date String']==date_max]

        df = df.reset_index(drop=True)
        
    

        import plotly.express as px

       
        fig = px.line(df, x='Time', y="Profit")
        fig.update_xaxes(nticks=5)
        tab1, tab2 = st.tabs(["Streamlit theme (default)", "Plotly native theme"])
        with tab1:
            st.plotly_chart(fig, theme="streamlit")
        with tab2:
            st.plotly_chart(fig, theme=None)
        
        
        
        current_profit= df['Profit'].tail(1).values[0]
        st.text("Profit is {}".format(current_profit))

        ############### DNS ######################

        #st.header('DNS_BNF')
        # create database 
        mydb_dns = client["DNS_BNF"]
        #create collection for DNS
        records =mydb_dns['records']    
        ab=records.find()

        df=pd.DataFrame(ab)
        df['Profit'] =df['Profit'].astype(float)
        df['Date String'] = df['Date'].str[:10]

      

        # Group by 'Date' and replace values as needed
        df['Profit'] = df.groupby('Date String')['Profit'].transform(lambda x: -60 if any(x < -60) else x)
        

        # Filter the DataFrame for the desired dates
        dates_to_extract = list(df['Date String'].unique())
        filtered_df = df[df['Date String'].isin(dates_to_extract)]

        # Group by 'Date String' and retrieve the last row for each group
        result = filtered_df.groupby('Date String').last().reset_index()

        # Convert the 'Date' column to datetime format
        result['Date'] = pd.to_datetime(result['Date'], format='%d/%m/%Y %H:%M:%S:%f')

        # Extract day from the datetime column
        result['Day'] = result['Date'].dt.day_name()
        result['ROI %']=(result['Profit']*15/50000)*100


        st.dataframe(result[['Date String','Profit','Day','ROI %']])


    with col2:

        st.text("Intraday:Ratio Spread")
        # create database 
        mydb = client["Ratio_spread"]

        #create collection
        records =mydb['records']


        ab=records.find()

        df_rs=pd.DataFrame(ab)

        df_rs['Date String'] = df_rs['Date'].str[:10]
        df_rs['Profit'] =df_rs['Profit'].astype(float)


        # Group by 'Date' and replace values as needed
        df_rs['Profit'] = df_rs.groupby('Date String')['Profit'].transform(lambda x: -60 if any(x < -60) else x)


        df_rs['Time']=df_rs['Date'].str[10:16]

        date_max_rs=df_rs['Date'].iloc[-1][:10]
        df_rs=df_rs[df_rs['Date String']==date_max_rs]
        df_rs = df_rs.reset_index(drop=True)

        
        
        import plotly.express as px

       
        fig = px.line(df_rs, x='Time', y="Profit")
        fig.update_xaxes(nticks=5)
        tab1, tab2 = st.tabs(["Streamlit theme (default)", "Plotly native theme"])
        with tab1:
            st.plotly_chart(fig, theme="streamlit")
        with tab2:
            st.plotly_chart(fig, theme=None)

        
        
        current_profit_rs= df_rs['Profit'].tail(1).values[0]
        st.text("Profit is {}".format(current_profit_rs))

      

        ############### Ratio Spread ######################

        #st.header('Ratio Spread_BNF :Intraday')
        # create database 
        mydb_rs = client["Ratio_spread"]
        #create collection for DNS
        records =mydb_rs['records']    
        cd=records.find()

        df_rs=pd.DataFrame(cd)
        df_rs['Profit'] =df_rs['Profit'].astype(float)
        df_rs['Date String'] = df_rs['Date'].str[:10]


        # Group by 'Date' and replace values as needed
        df_rs['Profit'] = df_rs.groupby('Date String')['Profit'].transform(lambda x: -60 if any(x < -60) else x)

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
        result_rs['ROI %']=(result_rs['Profit']*15/120000)*100

        

        st.dataframe(result_rs[['Date String','Profit','Day','ROI %']])
        
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

        

        # Group by 'Date' and replace values as needed
        df['Profit'] = df.groupby('Date String')['Profit'].transform(lambda x: -60 if any(x < -60) else x)

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

        
        # Group by 'Date' and replace values as needed
        df_rs['Profit'] = df_rs.groupby('Date String')['Profit'].transform(lambda x: -60 if any(x < -60) else x)

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

        

        # Group by 'Date' and replace values as needed
        df['Profit'] = df.groupby('Date String')['Profit'].transform(lambda x: -60 if any(x < -60) else x)

        dates_to_extract = list(df['Date String'].unique())

        # Merge the two DataFrames on the basis of the 'date' and 'datestring' columns
        merged_df = pd.merge(result, result_rs, left_on='Date String', right_on='Date String', suffixes=('_dns', '_rs'))
        merged_df = merged_df[['Date String','Profit_dns','Profit_rs','Day_dns']]
        merged_df['Total Profit']=merged_df['Profit_dns']+merged_df['Profit_rs']
        merged_df['Cumulative Profit']=merged_df['Total Profit'].cumsum()
        merged_df['ROI % Intraday']=(merged_df['Total Profit']*15/170000)*100
        merged_df['ROI % till now']=(merged_df['Cumulative Profit']*15/170000)*100
    
        
        import plotly.express as px
        
        fig = px.line(merged_df, x='Date String', y=['Profit_dns','Profit_rs','Total Profit','Cumulative Profit'], markers=True)

        tab1, tab2 = st.tabs(["Streamlit theme (default)", "Plotly native theme"])
        with tab1:
            st.plotly_chart(fig, theme="streamlit")
        with tab2:
            st.plotly_chart(fig, theme=None)

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
        
        st.dataframe(merged_df[['Profit_dns','Profit_rs']].corr())

def Accuracy():
    col1, col2 = st.columns([5, 5])

    with col1:
    
        ############### DNS ######################
        st.header('Accuracy', divider='rainbow')
        # create database 
        mydb_dns = client["DNS_BNF"]
        #create collection for DNS
        records =mydb_dns['records']    
        ab=records.find()

        df=pd.DataFrame(ab)
        df['Profit'] =df['Profit'].astype(float)
        df['Date String'] = df['Date'].str[:10]


        # Group by 'Date' and replace values as needed
        df['Profit'] = df.groupby('Date String')['Profit'].transform(lambda x: -60 if any(x < -60) else x)

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
   

        # Group by 'Date' and replace values as needed
        df_rs['Profit'] = df_rs.groupby('Date String')['Profit'].transform(lambda x: -60 if any(x < -60) else x)

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

        # Group by 'Date' and replace values as needed
        df['Profit'] = df.groupby('Date String')['Profit'].transform(lambda x: -60 if any(x < -60) else x)

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
        ROI_dns = (result['Profit'].sum()*15/50000)*100
        ROI_rs = (result_rs['Profit'].sum()*15/120000)*100


        fig = go.Figure(go.Funnel(
        y = ["DNS_BNF", "Ratio_Spread_BNF"],
        x = [Net_Profit_dns, Net_Profit_rs]))


        tab1, tab2 = st.tabs(["Streamlit theme (default)", "Plotly native theme"])
        with tab1:
            st.plotly_chart(fig, theme="streamlit")
        with tab2:
            st.plotly_chart(fig, theme=None)

        a=pd.DataFrame({"Bots":["DNS_BNF", "Ratio_Spread_BNF"],
                        "Profit":[Net_Profit_dns, Net_Profit_rs],
                        "ROI %" :[ROI_dns,ROI_rs]})
        st.dataframe(a)
        
# Initial run
current_time = datetime.now().time()
summary_trend_roi()
st.divider()
run_main_code()
st.divider()

total_profit_correlation()
st.divider()

Accuracy()


# Auto-refresh every 10 seconds
while True and current_time <=tt(9,55) and current_time >=tt(3,55):
    t.sleep(100)
    # Initial run
    current_time = datetime.now().time()
    st.rerun()

