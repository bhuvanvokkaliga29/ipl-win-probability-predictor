import streamlit as st
import pickle
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
import time

# ======================
# PAGE CONFIG
# ======================
st.set_page_config(layout="wide", page_title="IPL Predictor")

# ======================
# DARK PREMIUM CSS
# ======================
st.markdown("""
<style>

.main {
    background: linear-gradient(135deg,#0b0f14,#111827,#1f2937);
}

.card {
    background: rgba(255,255,255,0.06);
    backdrop-filter: blur(14px);
    padding:20px;
    border-radius:18px;
    box-shadow:0 8px 32px rgba(0,0,0,0.5);
}

.bigwin {
    font-size:42px;
    font-weight:bold;
    color:#00ff9f;
}

.biglose {
    font-size:42px;
    font-weight:bold;
    color:#ff4b4b;
}

</style>
""", unsafe_allow_html=True)


# ======================
# LOAD MODEL
# ======================
pipe = pickle.load(open("pipe.pkl","rb"))

teams = [
    'Sunrisers Hyderabad','Mumbai Indians','Royal Challengers Bangalore',
    'Kolkata Knight Riders','Kings XI Punjab','Chennai Super Kings',
    'Rajasthan Royals','Delhi Capitals'
]

cities = [
    'Mumbai','Chennai','Bangalore','Hyderabad','Delhi',
    'Kolkata','Jaipur','Mohali','Ahmedabad','Pune'
]

st.title("🏏 IPL Win Probability Dashboard")


# ======================
# LAYOUT
# ======================
left, right = st.columns([1,2])


# ======================
# LEFT PANEL (INPUTS)
# ======================
with left:
    st.markdown('<div class="card">', unsafe_allow_html=True)

    city = st.selectbox("Host City", cities)

    batting = st.selectbox("Batting Team", teams)

    bowl_options = [t for t in teams if t != batting]
    bowling = st.selectbox("Bowling Team", bowl_options)

    target = st.slider("Target", 100, 250, 180)
    score = st.slider("Current Score", 0, target, 80)
    overs = st.slider("Overs Completed", 0.0, 19.0, 10.0, step=0.1)
    wickets = st.slider("Wickets Fallen", 0, 9)

    predict = st.button("🚀 Predict")

    st.markdown('</div>', unsafe_allow_html=True)


# ======================
# RIGHT PANEL
# ======================
with right:

    if predict:

        runs_left = target - score
        balls_left = 120 - int(overs*6)
        wickets_left = 10 - wickets

        crr = score/overs if overs>0 else 0
        rrr = (runs_left*6)/balls_left if balls_left>0 else 0

        df = pd.DataFrame({
            'batting_team':[batting],
            'bowling_team':[bowling],
            'runs_left':[runs_left],
            'balls_left':[balls_left],
            'wickets':[wickets_left],
            'target':[target],
            'crr':[crr],
            'rrr':[rrr]
        })

        prob = pipe.predict_proba(df)

        win = prob[0][1]
        lose = prob[0][0]


        # ======================
        # BIG RESULT TEXT
        # ======================
        if win > lose:
            st.markdown(f'<p class="bigwin">{batting} WIN PROBABILITY {round(win*100,2)}%</p>', unsafe_allow_html=True)
        else:
            st.markdown(f'<p class="biglose">{bowling} WIN PROBABILITY {round(lose*100,2)}%</p>', unsafe_allow_html=True)


        # ======================
        # GAUGE
        # ======================
        fig = go.Figure(go.Indicator(
            mode="gauge+number",
            value=win*100,
            gauge={
                'axis':{'range':[0,100]},
                'bar':{'color':'#00ff9f'}
            }
        ))
        st.plotly_chart(fig, use_container_width=True)


        # ======================
        # STATS CARDS
        # ======================
        c1,c2,c3,c4 = st.columns(4)
        c1.metric("Runs Left", runs_left)
        c2.metric("Balls Left", balls_left)
        c3.metric("CRR", round(crr,2))
        c4.metric("RRR", round(rrr,2))


        # ======================
        # GRAPH 1 - SCORE PROJECTION
        # ======================
        overs_list = list(range(1,21))
        proj = [score + i*crr for i in overs_list]

        fig2 = px.line(x=overs_list, y=proj,
                       labels={"x":"Overs","y":"Projected Score"},
                       title="Score Projection")
        st.plotly_chart(fig2, use_container_width=True)


        # ======================
        # GRAPH 2 - WIN TREND
        # ======================
        trend = [min(win*100 + i*1.5,100) for i in range(20)]
        fig3 = px.area(x=overs_list, y=trend, title="Win Probability Trend")
        st.plotly_chart(fig3, use_container_width=True)