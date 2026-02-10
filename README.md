# 🏏 IPL Win Probability Predictor

> 🚀 A live Machine Learning web app that predicts **IPL match win probability in real-time** using ball-by-ball match data.

🔗 **Live App:**  
https://ipl-win-probability-predictor-by-bhuvan.streamlit.app/

---

## 📌 Project Overview

Cricket matches are highly dynamic — outcomes change every ball.

This project uses **Machine Learning + Real IPL Data** to estimate:

👉 *Which team is more likely to win at any moment during the chase.*

The app takes live match inputs like:

- Host City
- Batting Team
- Bowling Team
- Target
- Current Score
- Overs Completed
- Wickets Fallen

And instantly predicts:

✅ Win Probability  
✅ Required Run Rate  
✅ Current Run Rate  
✅ Score Projection  
✅ Win Trend Graphs  

All inside a **modern interactive dashboard**.

---

## ✨ Features

✅ Real-time win probability prediction  
✅ Trained on ball-by-ball IPL dataset  
✅ ~86% model accuracy  
✅ Modern dark glass UI  
✅ Animated probability gauge  
✅ Score projection charts  
✅ Win trend analytics  
✅ Team + City selectors  
✅ Streamlit Cloud deployment  

---

## 🧠 Machine Learning Details

### Dataset
- IPL historical matches
- Ball-by-ball deliveries
- 700+ matches
- 70,000+ records

### Feature Engineering
We created meaningful match-state features:

- Runs Left
- Balls Left
- Wickets Remaining
- Current Run Rate (CRR)
- Required Run Rate (RRR)
- Target Score
- City (venue)
- Batting Team
- Bowling Team

### Model
- Logistic Regression / RandomForest
- One-Hot Encoding for categorical features
- Scikit-Learn Pipeline

### Accuracy
Accuracy: ~86%
