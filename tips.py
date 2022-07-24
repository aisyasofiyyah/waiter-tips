import streamlit as st
import pandas as pd
import numpy as np
import statsmodels.api as sm
import plotly.express as px
import plotly.graph_objects as go
from sklearn.ensemble import RandomForestClassifier

st.set_page_config(layout="wide")
st.markdown("<h1 style='text-align: center;'>Waiter Tips Prediction App</h1>", unsafe_allow_html=True)

tips = pd.read_csv('https://raw.githubusercontent.com/aisyasofiyyah/waiter-tips/main/tips.csv')

st.markdown("""Tips given to the waiters according to:
          \nthe total bill paid
          \nnumber of people at a table
          \nand the day of the week""")

figure = px.scatter(data_frame = tips, x="total_bill",y="tip", size="size", color= "day",trendline="ols")
st.write(figure)

st.markdown("""Tips given to the waiters according to: 
        \nthe total bill paid, the number of people at a table and the gender of the person paying the bill""")

figure = px.scatter(data_frame = tips, x="total_bill",
                    y="tip", size="size", color= "sex", trendline="ols")
st.write(figure)

st.markdown("""Tips given to the waiters according to the days to find out which day the most tips are given to the waiters""")

figure = px.pie(tips, 
             values='tip', 
             names='day',hole = 0.5)
st.write(figure)
