import streamlit as st
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

#examples https://huggingface.co/NYU-DS-4-Everyone
#For this project we will use
#https://www.kaggle.com/datasets/uom190346a/sleep-health-and-lifestyle-dataset


#to push
#git add .
#git commit -am "Message here"
#git push

st.title("Welcome!")

st.write("Elliot Zheng and Katie Liao")

df = pd.read_csv("health.csv")

app_mode = st.sidebar.selectbox("Select a page",["01 Introduction","02 Data Visualization"])

if app_mode == "01 Introduction":
    st.markdown("# Introduction:")
    st.write("Sleep is an exctremely important factor in one's health and wellbeing. We will be trying to predict one's sleep based on several lifestyle factors.")
    num = st.number_input('No. of Rows', 5, 100)
    st.dataframe(df.head(num))
    st.dataframe(df.describe())
    st.write(df.info())

    st.text('(Rows,Columns)')
    st.write(df.shape)


if app_mode == "02 Data Visualization":
    st.markdown("# Data Visualization:")
    st.write("Here is a pairplot to look at the relationship between all of the variables: ")
    df2 = df[['Age','Sleep Duration','Quality of Sleep', 'Physical Activity Level', 'Stress Level', 'Heart Rate', 'Daily Steps']]
    fig = sns.pairplot(df2)
    st.pyplot(fig)
    num = st.number_input('No. of Rows', 5, 10)

