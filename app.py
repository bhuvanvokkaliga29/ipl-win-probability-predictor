import streamlit as st
import pickle
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
import time

# ======================================================
# PAGE CONFIG
# ======================================================
st.set_page_config(
    page_title="IPL Win Probability Predictor",
    layout="wide"
)

# ======================================================
# PREMIUM ANIMATED UI
# ======================================================
st.markdown("""
<style>

.stApp {
    background: linear-gradient(135deg,#0b0f14,#0f172a,#111827);
    color: white;
}

/* Card */
.card {
    background: rgba(255,255,255,0.06);
    backdrop-filter: blur(16px);
    padding: 25px;
    border-radius: 20px;
    box-shadow: 0 0 20px rgba(0,255,150,0.08);
}

/* Winner text */
.win {
    font-size:48px;
    font-weight:800;
    color:#00ff9f;
    text-align:center;
}

.lose {
    font-size:48px;
    font-weight:800;
    color:#ff4b4b;
    text-align:center;
}

/* Smooth animation */
.fade {
    animation: fadein 1s ease-in;
}

@keyframes fadein {
  from {opacity:0;}
  to {opacity:1;}
}

</style>
""", unsafe_allow_html=True)


# ======================================================
# LOAD MODEL
# ======================================================
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
# TITLE (NO LOGO, CLEAN)
# ======================================================
st.markdown("# 🏏 IPL Win Probability Predictor")
st.markdown("---")


# ======================================================
# LAYOUT
# ======================================================
left, right = st.columns([1,2])


# ======================================================
# INPUT PANEL
# ======================================================
with left:

    st.markdown('<div class="card">', unsafe_allow_html=True)

    city = st.selectbox("Host City", cities)

    batting = st.selectbox("Batting Team", teams)

    bowling = st.selectbox(
        "Bowling Team",
        [t for t in teams if t != batting]
    )

    target = st.slider("Target", 100, 250, 180)

    score = st.slider("Current Score", 0, target, 80)

    overs = st.slider("Overs", 0.1, 19.5, 10.0, step=0.1)

    wickets = st.slider("Wickets Fallen", 0, 9)

    predict = st.button("🚀 Predict")

    st.markdown('</div>', unsafe_allow_html=True)


# ======================================================
# RESULT PANEL
# ======================================================
with right:

    if predict:

        # -------------------------
        # CALCULATIONS
        # -------------------------
        runs_left = max(target - score, 0)
        balls_left = max(120 - int(overs*6), 1)
        wickets_left = 10 - wickets

        crr = score / overs if overs > 0 else 0
        rrr = (runs_left*6)/balls_left


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
        df = df.reindex(columns=pipe.feature_names_in_)
        df = df.fillna(0)

        prob = pipe.predict_proba(df)

        win = prob[0][1]
        lose = prob[0][0]

        win_pct = round(win*100,2)


        # ======================================================
        # ANIMATED WIN BAR
        # ======================================================
        progress = st.progress(0)

        for i in range(int(win_pct)):
            progress.progress(i)
            time.sleep(0.01)


        # ======================================================
        # WINNER TEXT
        # ======================================================
        winner = batting if win > lose else bowling

        if win > lose:
            st.markdown(
                f'<p class="win fade">🏆 {winner} → {win_pct}%</p>',
                unsafe_allow_html=True
            )
        else:
            st.markdown(
                f'<p class="lose fade">🏆 {winner} → {round(lose*100,2)}%</p>',
                unsafe_allow_html=True
            )


        # ======================================================
        # GAUGE CHART
        # ======================================================
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


        # ======================================================
        # METRICS
        # ======================================================
        c1,c2,c3,c4 = st.columns(4)

        c1.metric("Runs Left", runs_left)
        c2.metric("Balls Left", balls_left)
        c3.metric("CRR", round(crr,2))
        c4.metric("RRR", round(rrr,2))


        # ======================================================
        # WIN TREND GRAPH
        # ======================================================
        overs_list = np.arange(1,21)
        curve = np.linspace(20, win_pct, 20)

        fig2 = px.line(
            x=overs_list,
            y=curve,
            title="Win Probability Trend",
            labels={"x":"Overs","y":"Win %"}
        )

        st.plotly_chart(fig2, use_container_width=True)