import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import plotly_express as px
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

st.title("FOOTBALL ANALYSIS")
st.markdown("## OVERVIEW")

#import my csv file
st.markdown("### FIRST FIVE OBSERVATIONS")
df = pd.read_csv("key_stats.csv")
st.write(df.head())

st.markdown("### LAST FIVE OBSERVATIONS")
df = pd.read_csv("key_stats.csv")
st.write(df.tail())

st.markdown("### DATA INFO")
KS = df.shape
st.write(KS)

st.markdown("### PLAYERS NAME")
st.write(df["player_name"].describe())

st.markdown("### FIRST FIVE PLAYERS NAME")
st.write(df["player_name"].head())

st.markdown("### CLUBS NAME")
st.write(df["club"].describe())

st.markdown("### PLAYERS POSITION")
st.write(df["position"].describe())

st.markdown("### MINUTES PLAYED BY PLAYER")
st.write(df["minutes_played"].describe())

st.markdown("### MATCH PLAYED BY PLAYER")
st.write(df["match_played"].describe())


#UNIVARIATE ANALYSIS
st.markdown("# FOOTBALL UNIVARIATE ANALYSIS")
st.markdown("### MINUTES PLAYED ANALYSIS")
df = pd.read_csv("key_stats.csv")
st.write(df["match_played"].describe())

st.markdown("### CLUB ANALYSIS")
df = pd.read_csv("key_stats.csv")
st.write(df["club"].describe())

st.markdown("### POSITION ANALYSIS")
df = pd.read_csv("key_stats.csv")
st.write(df["position"].describe())

st.markdown("### MINUTES PLAYED ANALYSIS")
df = pd.read_csv("key_stats.csv")
st.write(df["minutes_played"].describe())

st.markdown("### GOALS ANALYSIS")
df = pd.read_csv("key_stats.csv")
st.write(df["goals"].describe())

st.markdown("### ASSISTS ANALYSIS")
df = pd.read_csv("key_stats.csv")
st.write(df["assists"].describe())

st.markdown("### DISTANCE COVERED ANALYSIS")
df = pd.read_csv("key_stats.csv")
st.write(df["distance_covered"].describe())

st.markdown("### SCATTER PLOTS FOR POSITION")
NAME = px.scatter(df["position"], y= "position", title="PLAYERS POSITION")
st.plotly_chart(NAME, use_container_width=True)

st.markdown("### LINE GRAPH REPRESENTATION FOR PLAYERS NAME")
PLAYER = px.line(df["player_name"], y= "player_name", title="NAME OF PLAYERS")
st.plotly_chart(PLAYER, use_container_width=True)

st.markdown("### BAR GRAPH REPRESENTATION FOR CLUB")
CLUB = px.bar(df["club"], y= "club", title="CLUBS NAME")
st.plotly_chart(CLUB, use_container_width=True)

st.markdown("### BAR REPRESENTATION FOR POSITION")
POSITION = px.bar(df["position"], y= "position", title="POSITION PLAYED")
st.plotly_chart(POSITION, use_container_width=True)

st.markdown("### PIE REPRESENTATION FOR POSITION")
POSITION = px.pie(df, names="position", title="POSITION PLAYED")
# Display the pie chart
st.plotly_chart(POSITION, use_container_width=True)

st.markdown("### HISTOGRAM REPRESENTATION FOR DISTANCE COVERED")
ABC = px.bar(df["distance_covered"], y ="distance_covered", title = "Pregnancies Distribution")
st.plotly_chart(ABC, use_container_width = True)

st.markdown("### LINE GRAPH REPRESENTATION FOR ASSISTS")
ASSIST = px.line(df["assists"], y ="assists", title = "Assist Contributions")
st.plotly_chart(ASSIST, use_container_width = True)

st.markdown("### HISTOGRAM REPRESENTATION FOR ASSISTS")
ASSIST = px.histogram(df["assists"], y ="assists", title = "Assist Contributions")
st.plotly_chart(ASSIST, use_container_width = True)

#BIVARIATE ANALYSIS
st.markdown("## BIVARIATE ANALYSIS")
st.markdown("### PLAYER NAME vs MINUTES PLAYED")
df2 = pd.DataFrame(df["player_name"],df["minutes_played"])
st.write(df2)

st.markdown("### CLUB vs POSITION")
df3 = pd.DataFrame(df["club"],df["position"])
st.write(df3)

st.markdown("### GOALS vs PLAYER NAME")
df4 = pd.DataFrame(df["goals"],df["player_name"])
st.write(df4)


st.markdown("# PREDICTIVE ANALYSIS")
X = df.drop("goals", axis=1)
Y = df["goals"]
X_train,X_test,Y_train,Y_test = train_test_split(X,Y,test_size=0.2)

model = LogisticRegression()
model.fit(X_train,Y_train) #training the model

st.markdown("## Goals Prediction")
prediction = model.predict(X_test)
st.write(prediction)

st.markdown("## Model Evaluation")
accuracy = accuracy_score(prediction, Y_test)
st.write(accuracy)



#download by typing "python -m pip install scikit-learn"
#download by typing "python -m pip install matlib"
#download by typing "python -m pip install seaborn"
#download by typing "python -m pip freeze > requirements.txt"