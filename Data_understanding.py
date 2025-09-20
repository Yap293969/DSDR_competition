import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import io

from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    classification_report, confusion_matrix, precision_recall_curve
)
from sklearn.metrics import ConfusionMatrixDisplay

import joblib

RANDOM_STATE = 42
np.random.seed(RANDOM_STATE)

df_excel = pd.read_csv("C:/Users/Yap Xiang Yang/Desktop/python/Python_file_import/Bank Customer Churn Prediction.csv")
#verify the dataset is loaded
print(df_excel.head)

churn_rate = df_excel['churn'].mean()
print(f"Overall churn rate: {churn_rate*100:.2f}%")

country_counts = df_excel['country'].value_counts()
print("\nCustomers by country:")
print(country_counts)

credit_by_churn = df_excel.groupby('churn')['credit_score'].mean()
print("\nAverage credit score by churn:")
print(credit_by_churn)

#Preproccessing
df_model = df_excel.copy() #backup to avoid any editted
df_model.drop(columns=['customer_id'],inplace = True) # Task not need customer_id

#Convert string to numerical
Encoder = LabelEncoder()
df_model ['gender'] = Encoder.fit_transform(df_model['gender']) # male 0 female 1
df_model ['country'] = Encoder.fit_transform(df_model['country']) #France 0 Spain 1 Germany 2

X = df_model.drop(columns=['churn']) # all column except churn
y = df_model['churn']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=RANDOM_STATE)

#Apply Standardization
Scaler = StandardScaler()
X_train_scaled = Scaler.fit_transform(X_train)
X_test_scaled = Scaler.transform(X_test)
models = {
    "Logistic Regression": LogisticRegression(max_iter=500),
    "Random Forest": RandomForestClassifier(n_estimators=100, random_state=RANDOM_STATE),}

log_reg = LogisticRegression(max_iter=500)
log_reg.fit(X_train, y_train)

rf_model = RandomForestClassifier(n_estimators=100, random_state=42)
rf_model.fit(X_train, y_train)

y_pred_log = log_reg.predict(X_test)
y_pred_rf = rf_model.predict(X_test)

print("=== Logistic Regression Evaluation ===")
print("Accuracy :", accuracy_score(y_test, y_pred_log))
print("Precision:", precision_score(y_test, y_pred_log))
print("Recall   :", recall_score(y_test, y_pred_log))
print("F1-score :", f1_score(y_test, y_pred_log))
print("\nClassification Report:\n", classification_report(y_test, y_pred_log))

print("\n=== Random Forest Evaluation ===")
print("Accuracy :", accuracy_score(y_test, y_pred_rf))
print("Precision:", precision_score(y_test, y_pred_rf))
print("Recall   :", recall_score(y_test, y_pred_rf))
print("F1-score :", f1_score(y_test, y_pred_rf))
print("\nClassification Report:\n", classification_report(y_test, y_pred_rf))

