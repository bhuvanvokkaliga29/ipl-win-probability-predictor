import streamlit as st
import pickle
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px

# =====================================================
# PAGE CONFIG
# =====================================================
st.set_page_config(layout="wide", page_title="IPL Win Predictor")

# =====================================================
# MODERN DARK UI
# =====================================================
st.markdown("""
<style>

.main {
    background: linear-gradient(135deg,#0b0f14,#111827,#1f2937);
}

.card {
    background: rgba(255,255,255,0.06);
    backdrop-filter: blur(12px);
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

# =====================================================
# LOAD MODEL
# =====================================================
pipe = pickle.load(open("pipe.pkl", "rb"))

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

# =====================================================
# LAYOUT
# =====================================================
left, right = st.columns([1, 2])

# =====================================================
# LEFT SIDE (INPUTS)
# =====================================================
with left:

    st.markdown('<div class="card">', unsafe_allow_html=True)

    city = st.selectbox("Host City", cities)

    batting = st.selectbox("Batting Team", teams)

    bowling_options = [t for t in teams if t != batting]
    bowling = st.selectbox("Bowling Team", bowling_options)

    target = st.slider("Target Score", 100, 250, 180)
    score = st.slider("Current Score", 0, target, 80)
    overs = st.slider("Overs Completed", 0.1, 19.5, 10.0, step=0.1)
    wickets = st.slider("Wickets Fallen", 0, 9)

    predict = st.button("🚀 Predict")

    st.markdown('</div>', unsafe_allow_html=True)

# =====================================================
# RIGHT SIDE (RESULTS)
# =====================================================
with right:

    if predict:

        # -------------------------
        # Calculations
        # -------------------------
        runs_left = target - score
        balls_left = 120 - int(overs * 6)
        wickets_left = 10 - wickets

        crr = score / overs
        rrr = (runs_left * 6) / balls_left if balls_left > 0 else 0

        # =====================================================
        # ⭐ CRITICAL PART (STRICT TRAINING ORDER)
        # DO NOT CHANGE ORDER
        # =====================================================
        columns = [
            'batting_team',
            'bowling_team',
            'city',
            'runs_left',
            'balls_left',
            'wickets',
            'total_runs_x',
            'crr',
            'rrr'
        ]

        values = [[
            batting,
            bowling,
            city,
            runs_left,
            balls_left,
            wickets_left,
            target,
            crr,
            rrr
        ]]

        df = pd.DataFrame(values, columns=columns)

        # -------------------------
        # Prediction
        # -------------------------
        prob = pipe.predict_proba(df)

        win = prob[0][1]
        lose = prob[0][0]

        # -------------------------
        # Result Text
        # -------------------------
        if win > lose:
            st.markdown(
                f'<p class="win">{batting} Win Chance {round(win*100,2)}%</p>',
                unsafe_allow_html=True
            )
        else:
            st.markdown(
                f'<p class="lose">{bowling} Win Chance {round(lose*100,2)}%</p>',
                unsafe_allow_html=True
            )

        # -------------------------
        # Gauge Chart
        # -------------------------
        fig = go.Figure(go.Indicator(
            mode="gauge+number",
            value=win * 100,
            title={'text': "Win Probability"},
            gauge={
                'axis': {'range': [0, 100]},
                'bar': {'color': "#00ff9f"}
            }
        ))

        st.plotly_chart(fig, use_container_width=True)

        # -------------------------
        # Stats Cards
        # -------------------------
        c1, c2, c3, c4 = st.columns(4)

        c1.metric("Runs Left", runs_left)
        c2.metric("Balls Left", balls_left)
        c3.metric("CRR", round(crr, 2))
        c4.metric("RRR", round(rrr, 2))

        # -------------------------
        # Score Projection Chart
        # -------------------------
        overs_list = list(range(1, 21))
        projection = [score + i * crr for i in overs_list]

        fig2 = px.line(
            x=overs_list,
            y=projection,
            labels={"x": "Overs", "y": "Projected Score"},
            title="Score Projection"
        )

        st.plotly_chart(fig2, use_container_width=True)