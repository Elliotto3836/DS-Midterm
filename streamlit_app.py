import streamlit as st
from streamlit_extras.let_it_rain import rain
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from ydata_profiling import ProfileReport
import streamlit.components.v1 as components
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score

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

app_mode = st.sidebar.selectbox("Select a page",["Business Case and Data Presentation","Data Visualization","Model Prediction"])

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



.st-ej {
    background-color: #60b5ff !important;  /* Change the text color to blue */
}

.st-gb ul li {
    background-color: #60b5ff !important;  /* Change bullet points color to blue */
} </style>""", unsafe_allow_html=True)


if app_mode == "Business Case and Data Presentation":
    rain(emoji="💤",font_size=54,falling_speed=5,animation_length="10",)
    st.markdown("# Introduction:")
    st.write("Sleep is vital for everyone’s health and wellbeing. This app aims to help you improve your sleep duration based on personalized model predictions, considering several lifestyle factors. You are currently on the Business Case and Data Presentation page, where you can get an overview of our dataset, specifically the variables that will be used in your sleep duration prediction. In the drop-down menu on the left, you can also find the Data Visualization page and the Model Prediction page. ")
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
    
    profile = ProfileReport(df, minimal=True)
    html = profile.to_html()
    components.html(html, height=800, scrolling=True) 

df2 = df[['Age','Sleep Duration','Quality of Sleep', 'Physical Activity Level', 'Stress Level', 'Heart Rate', 'Daily Steps']]

if app_mode == "Data Visualization":
    st.markdown("# Data Visualization:")
    st.write("Please find below graphs that further underscore significant details and correlations in our dataset.")


    countPlots, HeatMap, BoxAndWhisker, PieChart, pairPlot = st.tabs(["Count Plots", "Heat Map", "Box and Whisker Plots", "Pie Charts", "Pairplot"])

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
        varCount = st.selectbox("Choose a variable:", df2.columns, key="countplot_var")
        fig, ax = plt.subplots(figsize=(10, 6))
        sns.countplot(data=df2, x=varCount, ax=ax)
        st.pyplot(fig)

    with BoxAndWhisker:
        st.markdown("## :blue[Box and Whisker Plot]")
        
        varBox = st.selectbox("Choose a variable:", df2.columns, key="boxplot_var")
        
        fig, ax = plt.subplots(figsize=(10, 6))
        sns.boxplot(data=df2, y=varBox, ax=ax)
        
        st.pyplot(fig)

    with PieChart:
        st.markdown("## :blue[Pie Chart]")

        varPie = st.selectbox("Choose a variable:", df3.columns, key="piechart_var")
        
        pie_data = df3[varPie].value_counts()
        
        fig, ax = plt.subplots(figsize=(8, 8))
        ax.pie(pie_data, labels=pie_data.index, autopct='%1.1f%%', startangle=90, colors=sns.color_palette("pastel"))
        
        st.pyplot(fig)



if app_mode == "Model Prediction":
    st.markdown("# Model Prediction")
    st.write("Please input your personal health information below to predict your sleep duration. Feel free to play around with different values to see how each factor impacts your sleep!")
    
    X = df2.drop("Quality of Sleep", axis=1)
    y = df["Quality of Sleep"]

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    model = LinearRegression()

    model.fit(X_train, y_train)

    predictions = model.predict(X_test)

    st.write(f"### Mean Squared Error: {mean_squared_error(y_test, predictions):.2f}")

    fig, ax = plt.subplots(figsize=(10, 6))
    ax.scatter(y_test, predictions, color="blue")
    ax.plot([min(y_test), max(y_test)], [min(y_test), max(y_test)], color='red', linestyle="--")
    ax.set_xlabel("Actual Sleep Quality")
    ax.set_ylabel("Predicted Sleep Quality")
    ax.set_title("Actual vs Predicted Sleep Quality")
    st.pyplot(fig)

    age = st.number_input("Age", min_value=0, max_value=120, value=25)
    sleep_duration = st.slider("Sleep Duration (0-16)", 0, 16, 8)
    physical_activity = st.slider("Physical Activity Level (0-100)", 0, 100, 50)
    stress_level = st.slider("Stress Level (0-10)", 0, 10, 5)
    heart_rate = st.number_input("Heart Rate (BPM)", min_value=40, max_value=200, value=70)
    daily_steps = st.number_input("Daily Steps", min_value=0, max_value=50000, value=5000)

    if st.button("Predict Sleep Quality"):
        user_input = np.array([[age, sleep_duration, physical_activity, stress_level, heart_rate, daily_steps]])
        predicted_sleep = model.predict(user_input)
        st.write(f"### Predicted Sleep Quality: {predicted_sleep[0]:.2f} / 10")

