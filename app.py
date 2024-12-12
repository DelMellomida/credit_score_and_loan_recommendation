from flask import Flask, render_template, request
import joblib
import pandas as pd
import numpy as np
from sklearn.metrics import accuracy_score, precision_score, recall_score, classification_report, confusion_matrix
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA

# Initialize Flask app
app = Flask(__name__)

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

# Home Route
@app.route('/')
def index():
    return render_template('index.html') 

@app.route('/form')
def form():
    return render_template('predict_form.html') 

@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Extract input features from form
        features = [
            float(request.form['DerogCnt']),
            float(request.form['CollectCnt']),
            float(request.form['BanruptcyInd']),
            float(request.form['InqCnt06']),
            float(request.form['InqTimeLast']),
            float(request.form['InqFinanceCnt24']),
            float(request.form['TLTimeFirst']),
            float(request.form['TLTimeLast']),
            float(request.form['TLCnt03']),
            float(request.form['TLCnt12']),
            float(request.form['TLCnt24']),
            float(request.form['TLCnt']),
            float(request.form['TLSum'].replace('$', '').replace(',', '')),
            float(request.form['TLMaxSum'].replace('$', '').replace(',', '')),
            float(request.form['TLSatCnt']),
            float(request.form['TLDel60Cnt']),
            float(request.form['TLBadCnt24']),
            float(request.form['TL75UtilCnt']),
            float(request.form['TL50UtilCnt']),
            float(request.form['TLBalHCPct'].replace('%', '').replace(',', '')) / 100,
            float(request.form['TLSatPct'].replace('%', '').replace(',', '')) / 100,
            float(request.form['TLDel3060Cnt24']),
            float(request.form['TLDel90Cnt24']),
            float(request.form['TLDel60CntAll']),
            float(request.form['TLOpenPct'].replace('%', '').replace(',', '')) / 100,
            float(request.form['TLBadDerogCnt']),
            float(request.form['TLDel60Cnt24']),
            float(request.form['TLOpen24Pct'].replace('%', '').replace(',', '')) / 100
        ]
        
        # Convert to numpy array and reshape for the model
        input_data = np.array(features).reshape(1, -1)

        # Standardize and encode the input data (adjust accordingly)
        input_data_continuous = scaler.transform(input_data[:, :len(continuous_columns)])
        input_data_categorical = ohe.transform(input_data[:, len(continuous_columns):]).toarray()
        input_data_transformed = np.hstack([input_data_continuous, input_data_categorical])

        # Make prediction
        prediction = classifier.predict(input_data_transformed)
        prediction_probability = classifier.predict_proba(input_data_transformed)[0][1]

        return render_template(
            'results.html',
            prediction=int(prediction[0]),
            probability=round(prediction_probability * 100, 2)
        )
    except Exception as e:
        return f"An error occurred: {e}"


# Metric Route
@app.route('/metric')
def metric():
    return render_template('metric.html', results=results)

if __name__ == '__main__':
    app.run(debug=True)
