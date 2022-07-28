import streamlit as st
import pandas as pd
import numpy as np
import statsmodels.api as sm
import plotly.express as px
import matplotlib.pyplot as plt
import plotly.graph_objects as go
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score

st.set_page_config(layout="wide")
st.markdown("<h1 style='text-align: center;'>Waiter Tips Prediction App</h1>", unsafe_allow_html=True)

tips = pd.read_csv('https://raw.githubusercontent.com/aisyasofiyyah/waiter-tips/main/tips.csv')

st.sidebar.image('waiter.png',width=400)
st.sidebar.markdown("Waiter Tips App is where we analyse the tips given to a waiter for serving the food in a restaurant depends on certain factors and predict the tips that will given.")
option = st.sidebar.radio("This app contains two parts. Click to see.",('Analysis','Prediction'))

if option=='Analysis':
  
  tab1, tab2, tab3 = st.tabs(["Analysis 1", "Analysis 2", "Analysis 3"])

  with tab1:
            st.markdown("""Tips given to the waiters according to the total bill paid, number of people at a table (size) and the day of the week.""")

            figure = px.scatter(data_frame = tips, x="total_bill",y="tip", size="size", color= "day",trendline="ols")
            st.write(figure)

  with tab2:
            st.markdown("""Tips given to the waiters according to the total bill paid, number of people at a table (size) and the gender of the payer of the bill.""")

            figure = px.scatter(data_frame = tips, x="total_bill", y="tip", size="size", color= "sex", trendline="ols")
            st.write(figure)

  with tab3:
            st.markdown("""To find out which day the most tips are given to the waiters. Based on the pie chart below, the waiters are tipped more on Saturdays.""")

            figure = px.pie(tips, values='tip', names='day',hole = 0.5)
            st.write(figure)

else:
        
  #change string to numerical values
  tips["sex"] = tips["sex"].map({"Female": 0, "Male": 1})
  tips["smoker"] = tips["smoker"].map({"No": 0, "Yes": 1})
  tips["day"] = tips["day"].map({"Thur": 0, "Fri": 1, "Sat": 2, "Sun": 3})
  tips["time"] = tips["time"].map({"Lunch": 0, "Dinner": 1})

  #split to train and test
  X = pd.DataFrame(tips["total_bill"])
  y = tips.tip
  X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=0)

  #train a machine learning model for the task of waiter tips prediction
  model = LinearRegression()
  model.fit(X_train, y_train)
  
  #predict labels for the unknown data
  y_pred = model.predict(X_test)
  
  #evaluate the model performance
  st.markdown("Evaluation of the Linear Regression model performance.")
  st.markdown("Root mean squared error: {} ".format(mean_squared_error(y_test, y_pred)**0.5))
  st.markdown("Variance score: {} ".format(r2_score(y_test,y_pred)))
  st.markdown("*variance score near 1 means perfect prediction.")
  
  f= plt.figure()
  plt.scatter(X_test, y_test, color='blue', label='Total Bill Paid')
  plt.plot(X_test, y_pred, color='red', label='Predicted Tips Given to Waiter', linewidth=1)
  plt.xlabel("total_bill")
  plt.ylabel("tips")
  plt.legend()
  st.plotly_chart(f)
  
st.sidebar.markdown("""Reference: 
                      \n[1)](https://www.kaggle.com/datasets/aminizahra/tips-dataset) Zahra Amini, Waiter's Tips Dataset
                      \n[2)](https://thecleverprogrammer.com/2022/02/01/waiter-tips-prediction-with-machine-learning/) Aman Kharwal, Waiter Tips Prediction with Machine Learning
                      """)
