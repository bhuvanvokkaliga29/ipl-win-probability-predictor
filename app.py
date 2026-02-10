import streamlit as st
import pickle
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px

# ==================================================
# PAGE CONFIG
# ==================================================
st.set_page_config(layout="wide", page_title="IPL Win Predictor")

# ==================================================
# DARK GLASS UI
# ==================================================
st.markdown("""
<style>
.main {
    background: linear-gradient(135deg,#0b0f14,#111827,#1f2937);
}
.card {
    background: rgba(255,255,255,0.06);
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

# ==================================================
# LOAD MODEL
# ==================================================
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

st.title("🏏 IPL Win Probability Predictor")

# ==================================================
# LAYOUT
# ==================================================
left, right = st.columns([1,2])


# ==================================================
# LEFT PANEL (INPUTS)
# ==================================================
with left:
    st.markdown('<div class="card">', unsafe_allow_html=True)

    city = st.selectbox("Host City", cities)

    batting = st.selectbox("Batting Team", teams)

    # prevent same team
    bowling_options = [t for t in teams if t != batting]
    bowling = st.selectbox("Bowling Team", bowling_options)

    target = st.slider("Target Score", 100, 250, 180)
    score = st.slider("Current Score", 0, target, 90)
    overs = st.slider("Overs Completed", 0.1, 19.5, 10.0, step=0.1)
    wickets = st.slider("Wickets Fallen", 0, 9)

    predict = st.button("🚀 Predict")

    st.markdown('</div>', unsafe_allow_html=True)


# ==================================================
# RIGHT PANEL (RESULTS)
# ==================================================
with right:

    if predict:

        runs_left = target - score
        balls_left = 120 - int(overs * 6)
        wickets_left = 10 - wickets

        crr = score / overs
        rrr = (runs_left * 6) / balls_left if balls_left > 0 else 0

        # ==================================================
        # ⭐ IMPORTANT: MATCH TRAINING FEATURE NAMES EXACTLY
        # ==================================================
        df = pd.DataFrame({
            'batting_team':[batting],
            'bowling_team':[bowling],
            'city':[city],
            'runs_left':[runs_left],
            'balls_left':[balls_left],
            'wickets':[wickets_left],
            'total_runs_x':[target],   # MUST match training
            'crr':[crr],
            'rrr':[rrr]
        })

        prob = pipe.predict_proba(df)

        win = prob[0][1]
        lose = prob[0][0]

        # ==================================================
        # RESULT TEXT
        # ==================================================
        if win > lose:
            st.markdown(f'<p class="win">{batting} Win Chance {round(win*100,2)}%</p>', unsafe_allow_html=True)
        else:
            st.markdown(f'<p class="lose">{bowling} Win Chance {round(lose*100,2)}%</p>', unsafe_allow_html=True)


        # ==================================================
        # GAUGE CHART
        # ==================================================
        fig = go.Figure(go.Indicator(
            mode="gauge+number",
            value=win*100,
            title={'text':"Win Probability"},
            gauge={
                'axis':{'range':[0,100]},
                'bar':{'color':'#00ff9f'}
            }
        ))

        st.plotly_chart(fig, use_container_width=True)


        # ==================================================
        # STATS
        # ==================================================
        c1,c2,c3,c4 = st.columns(4)
        c1.metric("Runs Left", runs_left)
        c2.metric("Balls Left", balls_left)
        c3.metric("CRR", round(crr,2))
        c4.metric("RRR", round(rrr,2))


        # ==================================================
        # SCORE PROJECTION GRAPH
        # ==================================================
        overs_list = list(range(1,21))
        proj = [score + i*crr for i in overs_list]

        fig2 = px.line(x=overs_list, y=proj,
                       labels={"x":"Overs","y":"Projected Score"},
                       title="Score Projection")

        st.plotly_chart(fig2, use_container_width=True)