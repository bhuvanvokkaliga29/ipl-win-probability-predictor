# 🏏 IPL Win Probability Predictor

A **Live Win Probability Prediction App** for IPL matches using real historical data and Machine Learning.

This project predicts the _win probability_ of the team batting second based on current match state (score, overs, wickets, etc.).  
It also includes a **beautiful interactive dashboard** built with Streamlit.

---

## 🚀 Live Demo

👉 **Live App:** https://share.streamlit.io/bhuvanvokkaliga29/ipl-win-probability-predictor/main/app.py

_(Paste your actual deployed URL here once deployed.)_

---

## 🧠 Model & Features

✔ Trained on IPL ball-by-ball dataset  
✔ 86%+ accuracy  
✔ Real-time win prediction  
✔ Modern UI with charts and gauge visuals  
✔ Match stats cards  
✔ Host city and team selectors  
✔ Score progression & win trend graphs

---

## 🗂 Project Structure

ipl-win-probability-predictor/
│
├── app.py # Streamlit Web App
├── pipe.pkl # Trained Model
├── matches.csv # IPL match data
├── deliveries.csv # Ball-by-ball data
├── requirements.txt # Dependencies
└── model_training.ipynb # Notebook for training

---

## 🛠 How to Run Locally

1. **Clone the repository**

```bash
git clone https://github.com/bhuvanvokkaliga29/ipl-win-probability-predictor.git
cd ipl-win-probability-predictor

1)Create & activate virtual environment (optional)

python -m venv venv
venv\Scripts\activate     # Windows
source venv/bin/activate  # Mac/Linux

2)Install dependencies
pip install -r requirements.txt

3)Run the Streamlit app
streamlit run app.py
```
