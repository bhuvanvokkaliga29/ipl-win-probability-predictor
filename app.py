import streamlit as st
import pickle
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px


# ======================================================
# PAGE CONFIG
# ======================================================
st.set_page_config(
    page_title="IPL Win Predictor",
    layout="wide"
)


# ======================================================
# DARK PREMIUM UI
# ======================================================
st.markdown("""
<style>

.stApp {
    background: linear-gradient(135deg,#0b0f14,#111827,#1f2937);
    color:white;
}

.card {
    background: rgba(255,255,255,0.05);
    backdrop-filter: blur(14px);
    padding: 20px;
    border-radius: 18px;
}

.win {
    font-size:40px;
    font-weight:bold;
    color:#00ff9f;
}

.lose {
    font-size:40px;
    font-weight:bold;
    color:#ff4b4b;
}

</style>
""", unsafe_allow_html=True)


# ======================================================
# LOAD MODEL
# ======================================================
pipe = pickle.load(open("pipe.pkl", "rb"))


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


st.title("🏏 IPL Win Probability Predictor")


# ======================================================
# LAYOUT
# ======================================================
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

    overs = st.slider("⏳ Overs Completed", 0.1, 19.5, 10.0, step=0.1)

    wickets = st.slider("❌ Wickets Fallen", 0, 9)

    predict = st.button("🚀 Predict Win Probability")

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

        balls_left = max(120 - int(overs * 6), 1)

        wickets_left = 10 - wickets

        crr = score / overs if overs > 0 else 0

        rrr = (runs_left * 6) / balls_left


        # ======================================================
        # ⭐ PRODUCTION SAFE DATAFRAME (IMPORTANT)
        # ======================================================
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

        # exact column order as training
        df = df.reindex(columns=pipe.feature_names_in_)

        # remove NaN
        df = df.fillna(0)


        # -------------------------
        # PREDICTION
        # -------------------------
        prob = pipe.predict_proba(df)

        win = prob[0][1]
        lose = prob[0][0]


        # -------------------------
        # RESULT TEXT
        # -------------------------
        if win > lose:
            st.markdown(
                f'<p class="win">{batting} → {round(win*100,2)}%</p>',
                unsafe_allow_html=True
            )
        else:
            st.markdown(
                f'<p class="lose">{bowling} → {round(lose*100,2)}%</p>',
                unsafe_allow_html=True
            )


        # -------------------------
        # GAUGE CHART
        # -------------------------
        gauge = go.Figure(go.Indicator(
            mode="gauge+number",
            value=win*100,
            title={'text': "Win Probability"},
            gauge={
                'axis': {'range':[0,100]},
                'bar': {'color':'#00ff9f'}
            }
        ))

        st.plotly_chart(gauge, use_container_width=True)


        # -------------------------
        # METRICS
        # -------------------------
        c1,c2,c3,c4 = st.columns(4)

        c1.metric("Runs Left", runs_left)
        c2.metric("Balls Left", balls_left)
        c3.metric("CRR", round(crr,2))
        c4.metric("RRR", round(rrr,2))


        # -------------------------
        # WIN PROGRESSION GRAPH
        # -------------------------
        overs_list = np.arange(1,21)

        win_curve = np.linspace(30, win*100, 20)

        fig = px.line(
            x=overs_list,
            y=win_curve,
            labels={"x":"Overs", "y":"Win %"},
            title="Win Probability Progression"
        )

        st.plotly_chart(fig, use_container_width=True)


        # -------------------------
        # SCORE PROJECTION
        # -------------------------
        projection = [score + i*crr for i in overs_list]

        fig2 = px.line(
            x=overs_list,
            y=projection,
            labels={"x":"Overs", "y":"Projected Score"},
            title="Projected Score Curve"
        )

        st.plotly_chart(fig2, use_container_width=True)