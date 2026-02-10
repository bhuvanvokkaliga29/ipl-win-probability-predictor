# ======================================================
# IPL WIN PROBABILITY MODEL — FINAL PERFECT VERSION
# ======================================================

import pandas as pd
import numpy as np
import pickle

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


# ==============================
# TARGET SCORE (1st innings)
# ==============================
total_score_df = (
    deliveries
    .groupby(['match_id','inning'])['total_runs']
    .sum()
    .reset_index()
)

total_score_df = total_score_df[total_score_df['inning'] == 1]


match_df = matches.merge(
    total_score_df[['match_id','total_runs']],
    left_on='id',
    right_on='match_id'
)


# ==============================
# TEAM CLEANING
# ==============================
match_df['team1'] = match_df['team1'].str.replace('Delhi Daredevils','Delhi Capitals')
match_df['team2'] = match_df['team2'].str.replace('Delhi Daredevils','Delhi Capitals')

teams = [
    'Sunrisers Hyderabad','Mumbai Indians','Royal Challengers Bangalore',
    'Kolkata Knight Riders','Kings XI Punjab','Chennai Super Kings',
    'Rajasthan Royals','Delhi Capitals'
]

match_df = match_df[
    (match_df['team1'].isin(teams)) &
    (match_df['team2'].isin(teams)) &
    (match_df['dl_applied'] == 0)
]

match_df = match_df[['match_id','city','winner','total_runs']]


# ==============================
# MERGE
# ==============================
delivery_df = match_df.merge(deliveries, on='match_id')
delivery_df = delivery_df[delivery_df['inning'] == 2].copy()

print("Creating features...")


# ==============================
# FEATURE ENGINEERING
# ==============================

delivery_df['current_score'] = delivery_df.groupby('match_id')['total_runs_y'].cumsum()

delivery_df['runs_left'] = delivery_df['total_runs_x'] - delivery_df['current_score']

delivery_df['balls_left'] = 120 - (delivery_df['over']*6 + delivery_df['ball'])

delivery_df['player_dismissed'] = delivery_df['player_dismissed'].notna().astype(int)

wickets = delivery_df.groupby('match_id')['player_dismissed'].cumsum()
delivery_df['wickets'] = 10 - wickets

delivery_df['crr'] = delivery_df['current_score']*6 / np.maximum(120-delivery_df['balls_left'],1)
delivery_df['rrr'] = delivery_df['runs_left']*6 / np.maximum(delivery_df['balls_left'],1)

# ⭐ smart features (boost accuracy)
delivery_df['pressure'] = delivery_df['rrr'] - delivery_df['crr']
delivery_df['runs_per_wicket'] = delivery_df['runs_left'] / np.maximum(delivery_df['wickets'],1)

delivery_df['result'] = (delivery_df['batting_team'] == delivery_df['winner']).astype(int)


# ==============================
# FINAL DATASET
# ==============================

final_df = delivery_df[
    ['match_id','batting_team','bowling_team','city',
     'runs_left','balls_left','wickets','total_runs_x',
     'crr','rrr','pressure','runs_per_wicket','result']
].copy()

final_df = final_df.replace([np.inf,-np.inf],np.nan)
final_df.dropna(inplace=True)

# remove finished overs
final_df = final_df[(final_df['balls_left']>0) & (final_df['runs_left']>0)]

print("Dataset shape:", final_df.shape)


# ==============================
# GROUP SPLIT (NO LEAKAGE)
# ==============================
X = final_df.drop(['result'],axis=1)
y = final_df['result']

groups = final_df['match_id']

gss = GroupShuffleSplit(test_size=0.2,n_splits=1,random_state=42)
train_idx, test_idx = next(gss.split(X,y,groups))

X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]

X_train = X_train.drop('match_id',axis=1)
X_test  = X_test.drop('match_id',axis=1)


# ==============================
# PIPELINE
# ==============================
categorical = ['batting_team','bowling_team','city']
numeric = ['runs_left','balls_left','wickets','total_runs_x','crr','rrr','pressure','runs_per_wicket']

preprocessor = ColumnTransformer([
    ('cat', OneHotEncoder(handle_unknown='ignore'), categorical),
    ('num', SimpleImputer(strategy='constant', fill_value=0), numeric)
])

pipe = Pipeline([
    ('prep', preprocessor),
    ('model', RandomForestClassifier(n_estimators=250, random_state=42))
])


print("Training model...")
pipe.fit(X_train,y_train)

pred = pipe.predict(X_test)

print("✅ Accuracy:", accuracy_score(y_test,pred))


pickle.dump(pipe,open("pipe.pkl","wb"))
print("✅ pipe.pkl saved!")