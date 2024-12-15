from flask import Flask, render_template, request, g
import joblib
import pandas as pd
import numpy as np
from sklearn.metrics import accuracy_score, precision_score, recall_score, classification_report, confusion_matrix
# import matplotlib.pyplot as plt
# from sklearn.decomposition import PCA
from app import create_app, db
from app.routes import setup_routes

# Initialize Flask app
app = create_app()

# Load the model and preprocessor
classifier = joblib.load('saved_models/f1_Classifier_CreditScoring.pkl')
scaler = joblib.load('saved_models/scaler_CreditScoring.pkl')
ohe = joblib.load('saved_models/encoder_CreditScoring.pkl')

# Load dataset and preprocess (similar to your code)
df = pd.read_excel(r"data/a_Dataset_CreditScoring.xlsx")
df = df.drop('ID', axis=1)

# Step 1: Discard rows with more than 50% missing values
row_threshold = len(df.columns) / 2
df = df.dropna(thresh=row_threshold, axis=0)

# Step 2: Fill missing values
for col in df.columns:
    if df[col].dtype == 'object':  # Categorical variables
        df[col].fillna(df[col].mode()[0], inplace=True)
    else:  # Continuous variables
        df[col].fillna(df[col].mean(), inplace=True)

# Step 3: Discard columns with more than 95% missing values
column_threshold = 0.95 * len(df)
df = df.loc[:, df.isnull().sum() < column_threshold]

# Step 4: Define target and features
y = df['TARGET']
X = df.drop(columns=['TARGET'])

# Step 5: Standardization and encoding
categorical_columns = X.select_dtypes(include=['object']).columns
continuous_columns = X.select_dtypes(exclude=['object']).columns

# Encode categorical features
X_categorical = ohe.transform(X[categorical_columns]).toarray()
X_continuous = scaler.transform(X[continuous_columns])

# Combine features
X_transformed = np.hstack([X_continuous, X_categorical])

# Predict using the trained model
y_pred = classifier.predict(X_transformed)

# Step 5: Compute metrics
accuracy = accuracy_score(y, y_pred)
precision = precision_score(y, y_pred, average="weighted")
recall = recall_score(y, y_pred, average="weighted")
conf_matrix = confusion_matrix(y, y_pred)
class_report = classification_report(y, y_pred)

# Step 6: Prepare results to send to HTML
results = {
    'accuracy': accuracy,
    'precision': precision,
    'recall': recall,
    'conf_matrix': conf_matrix,
    'class_report': class_report
}

with app.app_context():
    g.results = results
    g.request = request

setup_routes(app)

if __name__ == '__main__':
    app.run(debug=True)
