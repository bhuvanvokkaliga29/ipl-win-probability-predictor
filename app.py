import streamlit as st
import pandas as pd
import numpy as np
import pickle
import os
import plotly.graph_objects as go
import plotly.express as px


# ======================================================
# AUTO TRAIN (cloud safe)
# ======================================================
if not os.path.exists("pipe.pkl"):
    import train_model

pipe = pickle.load(open("pipe.pkl","rb"))


# ======================================================
# PAGE CONFIG
# ======================================================
st.set_page_config(layout="wide")


# ======================================================
# PREMIUM CSS
# ======================================================
st.markdown("""
<style>
.stApp {
    background: linear-gradient(135deg,#0b0f14,#111827,#1f2937);
    color:white;
}

.big-win {
    font-size:50px;
    font-weight:bold;
    color:#00ff9f;
}

.big-lose {
    font-size:50px;
    font-weight:bold;
    color:#ff4b4b;
}

.card {
    background: rgba(255,255,255,0.06);
    padding:20px;
    border-radius:16px;
    backdrop-filter: blur(12px);
}
</style>
""", unsafe_allow_html=True)


# ======================================================
# LOGO MAP
# ======================================================
logos = {
    "Mumbai Indians":"https://i.imgur.com/tzGkO0F.png",
    "Chennai Super Kings":"https://i.imgur.com/8b2O1vS.png",
    "Royal Challengers Bangalore":"https://i.imgur.com/RG3U0yC.png",
    "Sunrisers Hyderabad":"https://i.imgur.com/xo0vL6S.png",
    "Kolkata Knight Riders":"https://i.imgur.com/xR9Xh3U.png",
    "Delhi Capitals":"https://i.imgur.com/M5n0rGF.png",
    "Rajasthan Royals":"https://i.imgur.com/Nc5C2aF.png",
    "Kings XI Punjab":"https://i.imgur.com/4QvT2XM.png"
}


teams = list(logos.keys())
cities = ['Mumbai','Chennai','Bangalore','Hyderabad','Delhi','Kolkata','Jaipur']


st.title("🏏 IPL Win Probability Predictor")


left, right = st.columns([1,2])


# ======================================================
# INPUTS
# ======================================================
with left:

    city = st.selectbox("City", cities)
    bat = st.selectbox("Batting Team", teams)
    bowl = st.selectbox("Bowling Team", [t for t in teams if t != bat])

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
        lose = prob[0]*100


        # =========================================
        # TEAM WINNER DISPLAY
        # =========================================
        col1, col2 = st.columns(2)

        with col1:
            st.image(logos[bat], width=120)
            st.write(f"### {bat}")

        with col2:
            st.image(logos[bowl], width=120)
            st.write(f"### {bowl}")

        st.divider()

        winner = bat if win>lose else bowl
        win_val = max(win,lose)

        st.markdown(f'<p class="big-win">🏆 {winner} → {round(win_val,2)}%</p>', unsafe_allow_html=True)


        # =========================================
        # PROGRESS BAR
        # =========================================
        st.progress(int(win))


        # =========================================
        # GAUGE
        # =========================================
        fig = go.Figure(go.Indicator(
            mode="gauge+number",
            value=win,
            title={'text':"Win Probability"},
            gauge={'axis':{'range':[0,100]}}
        ))
        st.plotly_chart(fig, use_container_width=True)


        # =========================================
        # PROJECTION GRAPH
        # =========================================
        overs_list = np.arange(1,21)
        projection = [score+i*crr for i in overs_list]

        fig2 = px.line(x=overs_list,y=projection,title="Projected Score Curve")
        st.plotly_chart(fig2, use_container_width=True)