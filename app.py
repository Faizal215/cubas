import pandas as pd
framingham = pd.read_csv('https://raw.githubusercontent.com/Natassha/streamlit_fhs/main/framingham.csv')# Dropping null values
framingham = framingham.dropna()
framingham.head()
framingham['TenYearCHD'].value_counts()

from sklearn.model_selection import train_test_split
from imblearn.over_sampling import RandomOverSampler
from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import Pipeline
from sklearn.metrics import accuracy_scoreX = framingham.drop('TenYearCHD',axis=1)
y = framingham['TenYearCHD']
X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=0.20)oversample = RandomOverSampler(sampling_strategy='minority')
X_over, y_over = oversample.fit_resample(X_train,y_train)rf = RandomForestClassifier()
rf.fit(X_over,y_over)

preds = rf.predict(X_test)
print(accuracy_score(y_test,preds))

import joblib
joblib.dump(rf, 'fhs_rf_model.pkl')

import streamlit as st
import joblib
import pandas as pd

st.write("# 10 Year Heart Disease Prediction")

streamlit run streamlit_fhs.py
