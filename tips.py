import streamlit as st
import pandas as pd
import numpy as np
import statsmodels.api as sm
import plotly.express as px
import plotly.graph_objects as go
from sklearn.ensemble import RandomForestClassifier

st.set_page_config(layout="wide")
st.markdown("<h1 style='text-align: center;'>Waiter Tips Prediction App</h1>", unsafe_allow_html=True)
st.text("Waiter Tips App is where we analyse the tips given to a waiter for serving the food in a restaurant depends on what kind of factors and predict the tips.")

tips = pd.read_csv('https://raw.githubusercontent.com/aisyasofiyyah/waiter-tips/main/tips.csv')

st.sidebar.markdown("""Reference: 
                    \nZahra Amini, [Waiter's Tips Dataset](https://www.kaggle.com/datasets/aminizahra/tips-dataset)
                    \nAman Kharwal, [Waiter Tips Prediction with Machine Learning](https://thecleverprogrammer.com/2022/02/01/waiter-tips-prediction-with-machine-learning/)
                    """)
tab1, tab2, tab3 = st.tabs(["Analysis 1", "Analysis 2", "Analysis 3"])

with tab1:
          st.markdown("""Tips given to the waiters according to:
          \nthe total bill paid, number of people at a table and the day of the week""")

          figure = px.scatter(data_frame = tips, x="total_bill",y="tip", size="size", color= "day",trendline="ols")
          st.write(figure)

with tab2:
          st.markdown("""Tips given to the waiters according to: 
                  \nthe total bill paid, the number of people at a table and the gender of the person paying the bill""")

          figure = px.scatter(data_frame = tips, x="total_bill", y="tip", size="size", color= "sex", trendline="ols")
          st.write(figure)

with tab3:
          st.markdown("""Tips given to the waiters according to the days to find out which day the most tips are given to the waiters""")

          figure = px.pie(tips, values='tip', names='day',hole = 0.5)
          st.write(figure)
