# train_model.py
import pandas as pd
import numpy as np
import joblib

from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from sklearn.model_selection import GroupShuffleSplit


print("Loading data...")

matches = pd.read_csv("matches.csv")
deliveries = pd.read_csv("deliveries.csv")


# TARGET SCORE
total_score_df = deliveries.groupby(['match_id','inning'])['total_runs'].sum().reset_index()
total_score_df = total_score_df[total_score_df['inning']==1]

match_df = matches.merge(
    total_score_df[['match_id','total_runs']],
    left_on='id',
    right_on='match_id'
)

match_df = match_df[['match_id','city','winner','total_runs']]

delivery_df = match_df.merge(deliveries,on='match_id')
delivery_df = delivery_df[delivery_df['inning']==2].copy()


# FEATURES
delivery_df['current_score'] = delivery_df.groupby('match_id')['total_runs_y'].cumsum()
delivery_df['runs_left'] = delivery_df['total_runs_x'] - delivery_df['current_score']
delivery_df['balls_left'] = 120 - (delivery_df['over']*6 + delivery_df['ball'])

delivery_df['player_dismissed'] = delivery_df['player_dismissed'].notna().astype(int)
wickets = delivery_df.groupby('match_id')['player_dismissed'].cumsum()
delivery_df['wickets'] = 10 - wickets

delivery_df['crr'] = delivery_df['current_score']*6 / np.maximum(120-delivery_df['balls_left'],1)
delivery_df['rrr'] = delivery_df['runs_left']*6 / np.maximum(delivery_df['balls_left'],1)

delivery_df['result'] = (delivery_df['batting_team']==delivery_df['winner']).astype(int)


final_df = delivery_df[
    ['match_id','batting_team','bowling_team','city',
     'runs_left','balls_left','wickets','total_runs_x','crr','rrr','result']
].dropna()

X = final_df.drop('result',axis=1)
y = final_df['result']

groups = final_df['match_id']

gss = GroupShuffleSplit(test_size=0.2,random_state=42)
train_idx,test_idx = next(gss.split(X,y,groups))

X_train = X.iloc[train_idx].drop('match_id',axis=1)
X_test  = X.iloc[test_idx].drop('match_id',axis=1)
y_train = y.iloc[train_idx]
y_test  = y.iloc[test_idx]


categorical = ['batting_team','bowling_team','city']
numeric = ['runs_left','balls_left','wickets','total_runs_x','crr','rrr']

pipe = Pipeline([
    ('prep', ColumnTransformer([
        ('cat', OneHotEncoder(handle_unknown='ignore'), categorical),
        ('num', SimpleImputer(strategy='constant', fill_value=0), numeric)
    ])),
    ('model', RandomForestClassifier(n_estimators=200, random_state=42))
])


print("Training...")
pipe.fit(X_train,y_train)

print("Accuracy:", accuracy_score(y_test, pipe.predict(X_test)))

# ⭐ use joblib (smaller + safer)
joblib.dump(pipe,"pipe.pkl",compress=3)

print("pipe.pkl saved!")