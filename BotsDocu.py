
import time as t
import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
from datetime import datetime
from datetime import time as tt

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

# Function to run the main code
def run_main_code():
   
    st.title("Bots Result")
    col1, col2 = st.columns([5, 5])


    with col1:
        st.header("DNS_BNF")
        


        # create database 
        mydb = client["DNS_BNF"]

        #create collection
        records =mydb['records']


        ab=records.find()

        df=pd.DataFrame(ab)
        df['Profit'] =df['Profit'].astype(float)
        df['Date String'] = df['Date'].str[:10]

        date_max=df['Date'].iloc[-1][:10]
        df=df[df['Date String']==date_max]
        df = df.reset_index(drop=True)

        import seaborn as sns
        # Seaborn line plot
        fig, ax = plt.subplots(figsize=(30, 20))
        sns.lineplot(data=df,x='Date',y='Profit',color='black')



        # Set x-axis ticks at regular intervals
        interval = round(len(df)/8)  # Set the interval you want
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


    with col2:

        st.header("Ratio_Spread BNF")
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
        
        
# Initial run
current_time = datetime.now().time()
run_main_code()

# Auto-refresh every 10 seconds
while True and current_time <=tt(15,45) and current_time >=tt(9,25):
    t.sleep(10)
    # Initial run
    current_time = datetime.now().time()
    st.rerun()
