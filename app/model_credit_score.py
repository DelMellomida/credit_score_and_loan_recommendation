import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, precision_score, recall_score, roc_auc_score, roc_curve
import joblib
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.decomposition import PCA
import os

# Construct the path dynamically based on the script's location
project_root = os.path.dirname(os.path.abspath(__file__))  # Get the absolute path of the current script
data_file = os.path.join(project_root, '..', 'data', 'a_Dataset_CreditScoring.xlsx')

df = pd.read_excel(data_file)


# Dropping customer ID column from the dataset
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

# Verify cleaning process
print("\nFinal Missing Values:\n", df.isnull().sum())
print("\nDataset Shape After Cleaning:", df.shape)

print(df.head())

# Step 4: Define target and features
y = df['TARGET']  # Target variable
X = df.drop(columns=['TARGET'])  # Features (excluding target)

# Identify categorical and continuous columns
categorical_columns = X.select_dtypes(include=['object']).columns
continuous_columns = X.select_dtypes(exclude=['object']).columns

# Step 5: Train-Test Split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=0, stratify=y)

# Step 6: Standardization for continuous variables and encoding categorical variables
# Fit and transform OneHotEncoder on categorical features
ohe = OneHotEncoder()
ohe.fit(X_train[categorical_columns])
X_train_categorical = ohe.transform(X_train[categorical_columns]).toarray()
X_test_categorical = ohe.transform(X_test[categorical_columns]).toarray()

# Standardize continuous features
scaler = StandardScaler()
X_train_continuous = scaler.fit_transform(X_train[continuous_columns])
X_test_continuous = scaler.transform(X_test[continuous_columns])

# Combine continuous and categorical features
X_train_transformed = np.hstack([X_train_continuous, X_train_categorical])
X_test_transformed = np.hstack([X_test_continuous, X_test_categorical])

# Get column names
categorical_feature_names = ohe.get_feature_names_out(categorical_columns)
all_columns = np.concatenate([continuous_columns, categorical_feature_names])

# Save the preprocessor components
joblib.dump(scaler, '../saved_models/scaler_CreditScoring.pkl')
joblib.dump(ohe, '../saved_models/encoder_CreditScoring.pkl')

# Step 7: Hyperparameter Tuning with GridSearchCV
param_grid = {'C': [0.1, 1, 10], 'solver': ['lbfgs', 'liblinear']}
grid_search = GridSearchCV(LogisticRegression(), param_grid, cv=5)
grid_search.fit(X_train_transformed, y_train)

# Get the best parameters from grid search
best_params = grid_search.best_params_
print("Best Parameters from GridSearchCV:", best_params)

# Step 8: Train the Logistic Regression model with best parameters
classifier = LogisticRegression(random_state=0, C=best_params['C'], solver=best_params['solver'])
classifier.fit(X_train_transformed, y_train)

# Step 9: Predict on the test set
y_pred = classifier.predict(X_test_transformed)

# Step 10: Evaluate model performance
print("Accuracy:", accuracy_score(y_test, y_pred))
print("Precision:", precision_score(y_test, y_pred, average="weighted"))
print("Recall:", recall_score(y_test, y_pred, average="weighted"))
print("\nConfusion Matrix:\n", confusion_matrix(y_test, y_pred))
print("\nClassification Report:\n", classification_report(y_test, y_pred))

# Step 11: Save the trained model
joblib.dump(classifier, '../saved_models/f1_Classifier_CreditScoring.pkl')

# Optional: Load the saved model for future predictions
# loaded_model = joblib.load('f1_Classifier_CreditScoring.pkl')
# y_new_pred = loaded_model.predict(X_new)  # X_new is new data to classify

# Step 12: Visualizing Coefficients of Logistic Regression
coefficients = classifier.coef_[0]
feature_importance = pd.DataFrame({
    'Feature': all_columns,
    'Coefficient': coefficients
}).sort_values(by='Coefficient', ascending=False)

print("\nFeature Importance (Coefficients):")
print(feature_importance)

# Step 13: Visualizing ROC-AUC Curve
roc_auc = roc_auc_score(y_test, classifier.predict_proba(X_test_transformed)[:, 1])
print(f"ROC-AUC Score: {roc_auc}")

fpr, tpr, thresholds = roc_curve(y_test, classifier.predict_proba(X_test_transformed)[:, 1])
plt.plot(fpr, tpr, label=f'ROC curve (area = {roc_auc:.2f})')
plt.plot([0, 1], [0, 1], linestyle='--')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver Operating Characteristic')
plt.legend(loc='best')
plt.show()

# Step 14: Visualizing Decision Boundaries with PCA (if dataset allows 2D visualization)
pca = PCA(n_components=2)
X_train_pca = pca.fit_transform(X_train_transformed)

classifier.fit(X_train_pca, y_train)

# Visualize decision boundary
xx, yy = np.meshgrid(np.linspace(X_train_pca[:, 0].min(), X_train_pca[:, 0].max(), 100),
                     np.linspace(X_train_pca[:, 1].min(), X_train_pca[:, 1].max(), 100))

Z = classifier.predict(np.c_[xx.ravel(), yy.ravel()])
Z = Z.reshape(xx.shape)

plt.contourf(xx, yy, Z, alpha=0.8)
plt.scatter(X_train_pca[:, 0], X_train_pca[:, 1], c=y_train, edgecolors='k', marker='o', s=50, alpha=0.7)
plt.title('Decision Boundary of Logistic Regression (PCA-reduced Data)')
plt.show()
