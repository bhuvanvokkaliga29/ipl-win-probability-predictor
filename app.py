import streamlit as st
import pickle
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
import os
import time

# ======================================================
# PAGE CONFIG
# ======================================================
st.set_page_config(
    page_title="IPL Win Probability Predictor",
    layout="wide"
)

# ======================================================
# PREMIUM UI
# ======================================================
st.markdown("""
<style>
.stApp {
    background: linear-gradient(135deg,#0b0f14,#0f172a,#111827);
    color:white;
}
.card {
    background: rgba(255,255,255,0.05);
    backdrop-filter: blur(14px);
    padding: 25px;
    border-radius: 20px;
}
.win {
    font-size:42px;
    font-weight:800;
    color:#00ff9f;
    text-align:center;
}
.lose {
    font-size:42px;
    font-weight:800;
    color:#ff4b4b;
    text-align:center;
}
</style>
""", unsafe_allow_html=True)

# ======================================================
# SAFE MODEL LOAD
# ======================================================
if not os.path.exists("pipe.pkl"):
    st.error("❌ Model file 'pipe.pkl' not found. Please upload it.")
    st.stop()

pipe = pickle.load(open("pipe.pkl","rb"))

# ======================================================
# CONSTANTS
# ======================================================
teams = [
    'Sunrisers Hyderabad','Mumbai Indians','Royal Challengers Bangalore',
    'Kolkata Knight Riders','Kings XI Punjab','Chennai Super Kings',
    'Rajasthan Royals','Delhi Capitals'
]

cities = [
    'Mumbai','Chennai','Bangalore','Hyderabad','Delhi',
    'Kolkata','Jaipur','Mohali','Ahmedabad','Pune'
]

# ======================================================
# TITLE
# ======================================================
st.title("🏏 IPL Win Probability Predictor")
st.markdown("---")

left, right = st.columns([1,2])

# ======================================================
# INPUT PANEL
# ======================================================
with left:
    st.markdown('<div class="card">', unsafe_allow_html=True)

    city = st.selectbox("🏟 Host City", cities)

    batting = st.selectbox("🏏 Batting Team", teams)

    bowling = st.selectbox(
        "🎯 Bowling Team",
        [t for t in teams if t != batting]
    )

    target = st.slider("🎯 Target Score", 100, 250, 180)
    score = st.slider("🏃 Current Score", 0, target, 80)

    # Proper over input
    over_number = st.slider("Overs Completed", 0, 19, 10)
    balls = st.slider("Balls in Current Over", 0, 5, 0)

    wickets = st.slider("❌ Wickets Fallen", 0, 9)

    predict = st.button("🚀 Predict Win Probability")

    st.markdown('</div>', unsafe_allow_html=True)

# ======================================================
# RESULT PANEL
# ======================================================
with right:

    if predict:

        # -----------------------------------
        # CALCULATIONS
        # -----------------------------------
        balls_bowled = over_number * 6 + balls
        balls_left = 120 - balls_bowled
        balls_left = max(balls_left, 1)

        runs_left = max(target - score, 0)
        wickets_left = 10 - wickets

        crr = score * 6 / balls_bowled if balls_bowled > 0 else 0
        rrr = runs_left * 6 / balls_left

        # -----------------------------------
        # FEATURE ROW
        # -----------------------------------
        row = {
            'batting_team': batting,
            'bowling_team': bowling,
            'city': city,
            'runs_left': runs_left,
            'balls_left': balls_left,
            'wickets': wickets_left,
            'total_runs_x': target,
            'crr': crr,
            'rrr': rrr
        }

        df = pd.DataFrame([row])

        # match training structure
        df = df.reindex(columns=pipe.feature_names_in_)
        df = df.fillna(0)

        # -----------------------------------
        # PREDICT
        # -----------------------------------
        prob = pipe.predict_proba(df)
        win = prob[0][1]
        lose = prob[0][0]

        win_pct = round(win * 100, 2)
        lose_pct = round(lose * 100, 2)

        winner = batting if win > lose else bowling

        # -----------------------------------
        # WINNER TEXT
        # -----------------------------------
        if win > lose:
            st.markdown(f'<p class="win">🏆 {winner} — {win_pct}%</p>', unsafe_allow_html=True)
        else:
            st.markdown(f'<p class="lose">🏆 {winner} — {lose_pct}%</p>', unsafe_allow_html=True)

        # -----------------------------------
        # GAUGE
        # -----------------------------------
        fig = go.Figure(go.Indicator(
            mode="gauge+number",
            value=win_pct,
            title={'text':"Win Probability"},
            gauge={
                'axis': {'range':[0,100]},
                'bar': {'color':"#00ff9f"}
            }
        ))

        st.plotly_chart(fig, use_container_width=True)

        # -----------------------------------
        # METRICS
        # -----------------------------------
        c1,c2,c3,c4 = st.columns(4)

        c1.metric("Runs Left", runs_left)
        c2.metric("Balls Left", balls_left)
        c3.metric("CRR", round(crr,2))
        c4.metric("RRR", round(rrr,2))

        # -----------------------------------
        # WIN TREND GRAPH
        # -----------------------------------
        overs_range = np.arange(1,21)
        trend = np.linspace(20, win_pct, 20)

        fig2 = px.line(
            x=overs_range,
            y=trend,
            labels={"x":"Overs","y":"Win %"},
            title="Win Probability Trend"
        )

        st.plotly_chart(fig2, use_container_width=True)

        # -----------------------------------
        # SCORE PROJECTION
        # -----------------------------------
        projected = [score + i*(crr) for i in overs_range]

        fig3 = px.line(
            x=overs_range,
            y=projected,
            labels={"x":"Overs","y":"Projected Score"},
            title="Projected Score Curve"
        )

        st.plotly_chart(fig3, use_container_width=True)