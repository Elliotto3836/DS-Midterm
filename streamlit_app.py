import streamlit as st
from streamlit_extras.let_it_rain import rain
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from ydata_profiling import ProfileReport
import streamlit.components.v1 as components

#examples https://huggingface.co/NYU-DS-4-Everyone
#For this project we will use
#https://www.kaggle.com/datasets/uom190346a/sleep-health-and-lifestyle-dataset

#to push
#git add .
#git commit -am "Message here"
#git push

#git pull to pull

#st.title("Welcome!")


st.markdown("# :rainbow[How to Improve Sleep? An analysis of Sleep and Health Data.]")

st.header("Elliot Zheng and Katie Liao",divider="blue")

st.image("Sleeping.jpg")

df = pd.read_csv("health.csv")

st.header("",divider="blue")

app_mode = st.sidebar.selectbox("Select a page",["Introduction","Data Visualization","Predictions"])

st.markdown("""<style>
div[data-baseweb="tab-list"] button {
    color: #60b5ff !important;  /* Nord blue text color */
}

div[data-baseweb="tab-list"] button[aria-selected="true"] {
    color: #60b5ff !important; 
    background-color: transparent !important; 
    font-weight: bold !important;
}

.st-gb {
    background-color: #60b5ff !important;  /* Change the text color to blue */
}
            
.st-gb {
    background: #60b5ff !important;  /* Change the text color to blue */
}

.st-ej {
    background-color: #60b5ff !important;  /* Change the text color to blue */
}

.st-gb ul li {
    background-color: #60b5ff !important;  /* Change bullet points color to blue */
} </style>""", unsafe_allow_html=True)

if app_mode == "Introduction":
    rain(emoji="💤",font_size=54,falling_speed=5,animation_length="10",)
    st.markdown("# Introduction:")
    st.write("Sleep is an exctremely important factor in one's health and wellbeing. We will be trying to predict one's sleep based on several lifestyle factors.")
    num = st.slider("Select number of rows to view", min_value=5, max_value=100, value=10)
    st.dataframe(df.head(num))
    st.markdown("## Description of the Data")
    st.dataframe(df.describe())
    st.markdown("## Variables Used")

    st.markdown(":blue[Age] -- How old is the person?")
    st.markdown(":blue[Sleep Duration] -- How long do they sleep?")
    st.markdown(":blue[Quality of Sleep] -- How well do they sleep (0-10).")
    st.markdown(":blue[Physical Activity Level] -- How much physical activity do they get? (0-100).")
    st.markdown(":blue[Stress Level] -- How stressed are they? (0-10).")
    st.markdown(":blue[Heart Rate] -- Heart Rate in Beats Per Minute.")
    st.markdown(":blue[Daily Steps] -- How many steps do they get per day?")
    st.markdown("## Rows, Columns")
    st.write(df.shape)

    st.markdown("## Pandas Profiling Report")
    profile = ProfileReport(df, explorative=True)
    profile.to_file("profile_report.html")  # Save the report

    with open("profile_report.html", "r", encoding="utf-8") as f:
        html = f.read()

    components.html(html, height=800, scrolling=True)


if app_mode == "Data Visualization":
    st.markdown("# Data Visualization:")


    pairPlot,HeatMap,countPlots, BoxAndWhisker, PieChart = st.tabs(["Pairplot","Heat Map","Count Plots", "Box and Whisker Plots", "Pie Charts"])

    df2 = df[['Age','Sleep Duration','Quality of Sleep', 'Physical Activity Level', 'Stress Level', 'Heart Rate', 'Daily Steps']]
    df3 = df[['Occupation', 'BMI Category', 'Blood Pressure', 'Sleep Disorder']]

    with pairPlot:
        st.markdown("## :blue[Pairplot]")
        fig = sns.pairplot(df2)
        st.pyplot(fig)

    with HeatMap:
        st.markdown("## :blue[Correlation Heatmap]")
        
        fig, ax = plt.subplots(figsize=(10, 6))
        sns.heatmap(df2.corr(), annot=True, cmap="Blues", linewidths=0.5, ax=ax)
        
        st.pyplot(fig)

    with countPlots:
        st.markdown("## :blue[Bar Plot]")
        varCount = st.selectbox("Choose a variable:", df.columns, key="countplot_var")
        fig, ax = plt.subplots(figsize=(10, 6))
        sns.countplot(data=df, x=varCount, ax=ax)
        st.pyplot(fig)

    with BoxAndWhisker:
        st.markdown("## :blue[Box and Whisker Plot]")
        
        varBox = st.selectbox("Choose a variable:", df2.columns, key="boxplot_var")
        
        fig, ax = plt.subplots(figsize=(10, 6))
        sns.boxplot(data=df2, y=varBox, ax=ax)
        
        st.pyplot(fig)

    with PieChart:
        st.markdown("## :blue[Pie Chart]")

        varPie = st.selectbox("Choose a variable:", df.columns, key="piechart_var")
        
        pie_data = df[varPie].value_counts()
        
        fig, ax = plt.subplots(figsize=(8, 8))
        ax.pie(pie_data, labels=pie_data.index, autopct='%1.1f%%', startangle=90, colors=sns.color_palette("pastel"))
        
        st.pyplot(fig)






if app_mode == "Predictions":
    st.markdown("# Predictions")