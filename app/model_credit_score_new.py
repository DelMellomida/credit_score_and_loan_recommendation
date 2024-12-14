import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, precision_score, recall_score, roc_auc_score
import joblib
import os

# Load the dataset
project_root = os.path.dirname(os.path.abspath(__file__))  # Get the absolute path of the current script
data_file = os.path.join(project_root, '..', 'data', 'f4_NewApplications_CreditScore_Predictions_Augmented.csv')
df = pd.read_csv(data_file)

# Step 1: Dropping identifier column if it exists
if 'ID' in df.columns:
    df = df.drop('ID', axis=1)

# Step 2: Discard rows with more than 50% missing values
row_threshold = len(df.columns) / 2
df = df.dropna(thresh=row_threshold, axis=0)

# Step 3: Fill missing values
for col in df.columns:
    if df[col].dtype == 'object':  # Categorical variables
        df[col].fillna(df[col].mode()[0], inplace=True)
    else:  # Continuous variables
        df[col].fillna(df[col].mean(), inplace=True)

# Step 4: Discard columns with more than 95% missing values
column_threshold = 0.95 * len(df)
df = df.loc[:, df.isnull().sum() < column_threshold]

# Step 5: Splitting features and target using 'Predicted Outcome' as the target column
target_column = 'Predicted Outcome'
X = df.drop([target_column, 'Unnamed: 0'], axis=1)  # Dropping an index-like column as well
y = df[target_column]

# Encode target if it's categorical
if y.dtype == 'object':
    y = y.map({'Good': 1, 'Bad': 0})  # Adjust mapping if needed

# Identify categorical and numerical columns
categorical_columns = X.select_dtypes(include=['object']).columns
numerical_columns = X.select_dtypes(exclude=['object']).columns

# Preprocessing pipeline
preprocessor = ColumnTransformer(transformers=[
    ('num', StandardScaler(), numerical_columns),
    ('cat', OneHotEncoder(handle_unknown='ignore'), categorical_columns)
])

# Splitting the data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Logistic Regression pipeline
pipeline = Pipeline(steps=[
    ('preprocessor', preprocessor),
    ('classifier', LogisticRegression(random_state=42))
])

# Train the model
pipeline.fit(X_train, y_train)

# Save the model
model_filename = "../saved_models/CreditScoring_New.pkl"
joblib.dump(pipeline, model_filename)
print(f"Model saved as {model_filename}")

# Predictions and evaluation
y_pred = pipeline.predict(X_test)
y_prob = pipeline.predict_proba(X_test)[:, 1]

# Metrics
accuracy = accuracy_score(y_test, y_pred)
precision = precision_score(y_test, y_pred)
recall = recall_score(y_test, y_pred)
roc_auc = roc_auc_score(y_test, y_prob)
conf_matrix = confusion_matrix(y_test, y_pred)
classification_rep = classification_report(y_test, y_pred)

# Print metrics
print(f"Accuracy: {accuracy}")
print(f"Precision: {precision}")
print(f"Recall: {recall}")
print(f"ROC-AUC: {roc_auc}")
print(f"Confusion Matrix:\n{conf_matrix}")
print(f"Classification Report:\n{classification_rep}")
