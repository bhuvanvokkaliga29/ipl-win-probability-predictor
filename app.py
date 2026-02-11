import streamlit as st
import pandas as pd
import numpy as np
import pickle
import plotly.graph_objects as go
import plotly.express as px
import os


# ======================================================
# AUTO TRAIN IF MODEL NOT FOUND (CLOUD SAFE)
# ======================================================
if not os.path.exists("pipe.pkl"):
    import train_model

pipe = pickle.load(open("pipe.pkl", "rb"))


# ======================================================
# PAGE CONFIG
# ======================================================
st.set_page_config(layout="wide")


# ======================================================
# PREMIUM UI
# ======================================================
st.markdown("""
<style>
.stApp {
    background: linear-gradient(135deg,#0b0f14,#111827,#1f2937);
    color:white;
}
.win {color:#00ff9f;font-size:40px;font-weight:bold;}
.lose {color:#ff4b4b;font-size:40px;font-weight:bold;}
</style>
""", unsafe_allow_html=True)


teams = [
    'Sunrisers Hyderabad','Mumbai Indians','Royal Challengers Bangalore',
    'Kolkata Knight Riders','Kings XI Punjab','Chennai Super Kings',
    'Rajasthan Royals','Delhi Capitals'
]

cities = ['Mumbai','Chennai','Bangalore','Hyderabad','Delhi','Kolkata','Jaipur']


st.title("🏏 IPL Win Probability Predictor")

left, right = st.columns([1,2])


# ======================================================
# INPUTS
# ======================================================
with left:

    city = st.selectbox("City", cities)
    bat = st.selectbox("Batting", teams)
    bowl = st.selectbox("Bowling", [t for t in teams if t != bat])

    target = st.slider("Target",100,250,180)
    score = st.slider("Score",0,target,80)
    overs = st.slider("Overs",0.1,19.5,10.0)
    wickets = st.slider("Wickets",0,9)

    predict = st.button("Predict")


# ======================================================
# RESULT
# ======================================================
with right:

    if predict:

        runs_left = target-score
        balls_left = 120-int(overs*6)
        wickets_left = 10-wickets

        crr = score/overs
        rrr = (runs_left*6)/balls_left

        pressure = rrr-crr
        rpw = runs_left/max(wickets_left,1)

        df = pd.DataFrame([{
            'batting_team': bat,
            'bowling_team': bowl,
            'city': city,
            'runs_left': runs_left,
            'balls_left': balls_left,
            'wickets': wickets_left,
            'total_runs_x': target,
            'crr': crr,
            'rrr': rrr,
            'pressure': pressure,
            'runs_per_wicket': rpw
        }])

        prob = pipe.predict_proba(df)[0]

        win = prob[1]*100

        st.markdown(f'<p class="win">Win Probability → {round(win,2)}%</p>', unsafe_allow_html=True)


        # Gauge
        fig = go.Figure(go.Indicator(
            mode="gauge+number",
            value=win,
            title={'text':"Win %"},
            gauge={'axis':{'range':[0,100]}}
        ))
        st.plotly_chart(fig, use_container_width=True)


        # Projection graph
        overs_list = np.arange(1,21)
        projection = [score+i*crr for i in overs_list]

        fig2 = px.line(x=overs_list,y=projection,title="Projected Score")
        st.plotly_chart(fig2, use_container_width=True)