import pandas as pd
import numpy as np
from sklearn.utils import shuffle
import os

# Load the original dataset
project_root = os.path.dirname(os.path.abspath(__file__))  # Get the absolute path of the current script
file_path = os.path.join(project_root, '..', 'data', 'f4_NewApplications_CreditScore_Predictions.csv')
df = pd.read_csv(file_path)

# Define the target column and separate features
target_column = 'Predicted Outcome'
original_size = len(df)

# Generate synthetic data
n_samples = 1000  # Total desired rows
current_samples = df.shape[0]
additional_samples = max(0, n_samples - current_samples)

# Handle categorical columns
categorical_columns = df.select_dtypes(include=['object']).columns
numerical_columns = df.select_dtypes(exclude=['object']).columns

# Create synthetic data for numerical columns
synthetic_data = pd.DataFrame()

# Random sampling for numerical columns while preserving statistics
for col in numerical_columns:
    mean = df[col].mean()
    std = df[col].std()
    synthetic_data[col] = np.random.normal(loc=mean, scale=std, size=additional_samples)

# Random sampling for categorical columns while maintaining class distribution
for col in categorical_columns:
    synthetic_data[col] = np.random.choice(df[col].dropna().unique(), size=additional_samples, replace=True)

# Add the target column with class balancing
synthetic_data[target_column] = np.random.choice(df[target_column].unique(), size=additional_samples, replace=True)

# Concatenate original and synthetic data
df_augmented = pd.concat([df, synthetic_data], ignore_index=True)

# Shuffle the augmented dataset
df_augmented = shuffle(df_augmented, random_state=42)

# Save the new dataset
project_root = os.path.dirname(os.path.abspath(__file__))  # Get the absolute path of the current script
augmented_file_path = os.path.join(project_root, '..', 'data', 'f4_NewApplications_CreditScore_Predictions_Augmented.csv')
df_augmented.to_csv(augmented_file_path, index=False)

# Return the augmented file path and dataset size
augmented_file_path, df_augmented.shape
