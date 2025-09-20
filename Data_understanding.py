import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import io

from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    classification_report, confusion_matrix, precision_recall_curve
)
from sklearn.metrics import ConfusionMatrixDisplay

import joblib

df_excel = pd.read_csv("C:/Users/Yap Xiang Yang/Desktop/python/Python_file_import/Bank Customer Churn Prediction.csv")
#verify the dataset is loaded
#print(df_excel.head)

churn_rate = df_excel['churn'].mean()
print(f"Overall churn rate: {churn_rate:4f} ({churn_rate*100:2f}%)")