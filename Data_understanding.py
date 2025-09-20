import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import io
import seaborn as sns

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

# Cell 3: feature lists, train/test split, preprocessing pipeline
# Remove customer_id and prepare X, y
FEATURES = [c for c in df_excel.columns if c not in ['customer_id', 'churn']]
# numeric and categorical columns based on the problem statement
num_features = ['credit_score', 'age', 'tenure', 'balance', 'products_number', 'credit_card', 'active_member', 'estimated_salary']
cat_features = ['country', 'gender']

# Sanity: ensure columns exist
num_features = [c for c in num_features if c in df_excel.columns]
cat_features = [c for c in cat_features if c in df_excel.columns]

X = df_excel[FEATURES].copy()
y = df_excel['churn'].copy()

# stratify to preserve churn proportion
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, stratify=y, random_state=RANDOM_STATE
)

# Preprocessing:
numeric_transform = Pipeline([
    ('imputer', SimpleImputer(strategy='median')),
    ('scaler', StandardScaler())
])
categorical_transform = Pipeline([
    ('imputer', SimpleImputer(strategy='most_frequent')),
    ('ohe', OneHotEncoder(handle_unknown='ignore', sparse_output=False))
])

preprocessor = ColumnTransformer(transformers=[
    ('num', numeric_transform, num_features),
    ('cat', categorical_transform, cat_features)
])

# Cell 4: logistic regression model pipeline
lr_pipeline = Pipeline([
    ('pre', preprocessor),
    ('clf', LogisticRegression(max_iter=1000, class_weight='balanced', random_state=RANDOM_STATE))
])
lr_pipeline.fit(X_train, y_train)
print("Logistic Regression trained.")

# Cell 5: Random Forest model pipeline
rf_pipeline = Pipeline([
    ('pre', preprocessor),
    ('clf', RandomForestClassifier(n_estimators=200, class_weight='balanced', random_state=RANDOM_STATE))
])
rf_pipeline.fit(X_train, y_train)
print("Random Forest trained.")

# Cell 7: evaluation helper and run on test set
def evaluate_model(name, model, X_test, y_test):
    y_pred = model.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    prec = precision_score(y_test, y_pred, zero_division=0)
    rec = recall_score(y_test, y_pred, zero_division=0)
    f1 = f1_score(y_test, y_pred, zero_division=0)
    print(f"--- {name} ---")
    print(f"Accuracy:  {acc:.4f}")
    print(f"Precision: {prec:.4f}")
    print(f"Recall:    {rec:.4f}")
    print(f"F1-score:  {f1:.4f}")
    print("\nClassification report:")
    print(classification_report(y_test, y_pred, zero_division=0))
    cm = confusion_matrix(y_test, y_pred)
    disp = ConfusionMatrixDisplay(cm, display_labels=[0,1])
    disp.plot()
    plt.title(f"Confusion Matrix - {name}")
    plt.show()

evaluate_model("Logistic Regression", lr_pipeline, X_test, y_test)
evaluate_model("Random Forest", rf_pipeline, X_test, y_test)

# Cell 8: feature importance from Random Forest
# Fit the preprocessor so we can extract transformed feature names
preprocessor.fit(X_train)

# numeric feature names remain the same
num_names = num_features

# one-hot encoded feature names
ohe = preprocessor.named_transformers_['cat'].named_steps['ohe']
ohe_names = list(ohe.get_feature_names_out(cat_features)) if hasattr(ohe, 'get_feature_names_out') else []

feature_names = np.array(num_names + ohe_names)

# get RF classifier and its importances
rf_clf = rf_pipeline.named_steps['clf']
importances = rf_clf.feature_importances_

fi_df = pd.DataFrame({'feature': feature_names, 'importance': importances})
fi_df = fi_df.sort_values('importance', ascending=False).reset_index(drop=True)
fi_df.head(15)

plt.figure(figsize=(8,6))
sns.barplot(x='importance', y='feature', data=fi_df.head(12))
plt.title('Top 12 Feature Importances (Random Forest)')
plt.xlabel('Importance')
plt.show()

# Cell 9: Age groups analysis
bins = [0, 30, 40, 50, 120]                # adjust bins as you like
labels = ['<30', '30-39', '40-49', '50+']  # label names
df_excel['age_group'] = pd.cut(df_excel['age'], bins=bins, labels=labels, right=False)

age_group_stats = df_excel.groupby('age_group').agg(
    count=('churn','size'),
    churn_rate=('churn','mean')
).reset_index()
age_group_stats['churn_rate_percent'] = age_group_stats['churn_rate']*100
age_group_stats

# plot churn rate by age group
plt.figure(figsize=(6,4))
sns.barplot(x='age_group', y='churn_rate_percent', data=age_group_stats)
plt.ylabel('Churn rate (%)')
plt.title('Churn rate by age group')
plt.show()

# Cell 10: demonstrate recall vs threshold (choose LR)
y_proba = lr_pipeline.predict_proba(X_test)[:,1]
precision, recall, thresholds = precision_recall_curve(y_test, y_proba)

# Plot precision and recall vs threshold
plt.figure(figsize=(8,4))
plt.plot(thresholds, precision[:-1], label='Precision')
plt.plot(thresholds, recall[:-1], label='Recall')
plt.xlabel('Threshold')
plt.legend()
plt.title('Precision & Recall vs Threshold (Logistic Regression)')
plt.show()

# Show effect of lowering threshold to increase recall
for thr in [0.5, 0.4, 0.3, 0.2]:
    y_pred_thr = (y_proba >= thr).astype(int)
    print(f"Threshold {thr:.2f} -> Recall: {recall_score(y_test, y_pred_thr):.4f}, Precision: {precision_score(y_test, y_pred_thr, zero_division=0):.4f}")

# Cell 11: quick summary from feature importance
print("Top 5 features by importance (RF):")
print(fi_df.head(5))

# Example: logistic regression coefficients (optional)
coef_map = None
try:
    # fit preprocessor separately to transform features
    X_train_trans = preprocessor.transform(X_train)
    # if pipeline uses sparse False, it's array
    lr_coef = lr_pipeline.named_steps['clf'].coef_[0]
    # Align LR coefficients with feature names (same order as feature_names)
    coef_map = pd.DataFrame({'feature': feature_names, 'coef': lr_coef})
    coef_map = coef_map.sort_values('coef', key=lambda s: s.abs(), ascending=False)
    print("\nTop coefficients (Logistic Regression):")
    print(coef_map.head(8))
except Exception as e:
    print("Could not extract LR coefficients in this environment:", e)
