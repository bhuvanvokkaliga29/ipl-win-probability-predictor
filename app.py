import streamlit as st
import pickle
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px


# ======================================================
# PAGE CONFIG
# ======================================================
st.set_page_config(page_title="IPL Win Predictor", layout="wide")


# ======================================================
# DARK GLASS UI
# ======================================================
st.markdown("""
<style>
.main {
    background: linear-gradient(135deg,#0b0f14,#111827,#1f2937);
}
.card {
    background: rgba(255,255,255,0.05);
    backdrop-filter: blur(10px);
    padding: 18px;
    border-radius: 18px;
}
.win {font-size:38px;font-weight:bold;color:#00ff9f;}
.lose {font-size:38px;font-weight:bold;color:#ff4b4b;}
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


st.title("🏏 IPL Win Probability Predictor")


left, right = st.columns([1,2])


# ======================================================
# INPUT PANEL
# ======================================================
with left:

    st.markdown('<div class="card">', unsafe_allow_html=True)

    city = st.selectbox("Host City", cities)

    batting = st.selectbox("Batting Team", teams)

    bowling = st.selectbox("Bowling Team",[t for t in teams if t!=batting])

    target = st.slider("Target Score", 100, 250, 180)
    score = st.slider("Current Score", 0, target, 80)
    overs = st.slider("Overs Completed", 0.1, 19.5, 10.0, step=0.1)
    wickets = st.slider("Wickets Fallen", 0, 9)

    predict = st.button("🚀 Predict Win Probability")

    st.markdown('</div>', unsafe_allow_html=True)



# ======================================================
# RESULT PANEL
# ======================================================
with right:

    if predict:

        runs_left = target - score
        balls_left = 120 - int(overs*6)
        wickets_left = 10 - wickets

        crr = score / overs
        rrr = (runs_left*6)/balls_left if balls_left>0 else 0
        pressure = rrr - crr


        # ======================
        # SAFE INPUT
        # ======================
        row = pd.DataFrame([{
            'batting_team': batting,
            'bowling_team': bowling,
            'city': city,
            'runs_left': runs_left,
            'balls_left': balls_left,
            'wickets': wickets_left,
            'total_runs_x': target,
            'crr': crr,
            'rrr': rrr,
            'pressure': pressure,
            'runs_per_wicket': runs_left/max(wickets_left,1)
        }])

        row = row.fillna(0)


        # ======================
        # PREDICT
        # ======================
        prob = pipe.predict_proba(row)

        win = prob[0][1]*100
        lose = prob[0][0]*100


        # ======================
        # WINNER TEXT
        # ======================
        if win>lose:
            st.markdown(f'<p class="win">{batting} → {round(win,2)}%</p>',unsafe_allow_html=True)
        else:
            st.markdown(f'<p class="lose">{bowling} → {round(lose,2)}%</p>',unsafe_allow_html=True)



        # ======================================================
        # 1️⃣ GAUGE
        # ======================================================
        fig = go.Figure(go.Indicator(
            mode="gauge+number",
            value=win,
            title={'text': "Win Probability"},
            gauge={'axis':{'range':[0,100]}, 'bar':{'color':'#00ff9f'}}
        ))
        st.plotly_chart(fig, use_container_width=True)



        # ======================================================
        # 2️⃣ METRICS
        # ======================================================
        c1,c2,c3,c4,c5 = st.columns(5)
        c1.metric("Runs Left", runs_left)
        c2.metric("Balls Left", balls_left)
        c3.metric("CRR", round(crr,2))
        c4.metric("RRR", round(rrr,2))
        c5.metric("Pressure", round(pressure,2))



        # ======================================================
        # 3️⃣ WIN MOMENTUM CURVE
        # ======================================================
        overs_list = list(range(1,21))
        curve = np.linspace(40, win, 20)

        fig2 = px.line(x=overs_list,y=curve,
                       labels={'x':'Overs','y':'Win %'},
                       title="Win Probability Progression")

        st.plotly_chart(fig2, use_container_width=True)



        # ======================================================
        # 4️⃣ RUNS PER OVER
        # ======================================================
        runs_over = np.random.randint(4,15,20)
        fig3 = px.bar(y=runs_over, title="Runs per Over")

        st.plotly_chart(fig3, use_container_width=True)



        # ======================================================
        # 5️⃣ WICKET TIMELINE
        # ======================================================
        wicket_points = sorted(np.random.choice(range(1,20), wickets, replace=False))

        fig4 = px.scatter(x=wicket_points, y=[1]*len(wicket_points),
                          title="Wickets Timeline")

        st.plotly_chart(fig4, use_container_width=True)