import pandas as pd
import numpy as np
import joblib
import os
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression

# Define the directory where the model will be saved
SAVE_DIR = "model_files"
os.makedirs(SAVE_DIR, exist_ok=True)  # Create the directory if it doesn't exist

# Load dataset (Ensure the file is in the project directory)
df = pd.read_csv("hospital_readmissions.csv")

# Convert 'readmitted' column to binary values
df['readmitted'] = df['readmitted'].apply(lambda x: 1 if str(x).strip().upper() == "YES" else 0)

# Convert numeric-like object columns to integers
numeric_columns = ['glucose_test', 'A1Ctest']
df[numeric_columns] = df[numeric_columns].apply(pd.to_numeric, errors='coerce').fillna(0)

# One-Hot Encode categorical columns
categorical_columns = ['medical_specialty', 'diag_1', 'diag_2', 'diag_3', 'change', 'diabetes_med']
df = pd.get_dummies(df, columns=categorical_columns, drop_first=True)

# Select relevant features
features = ['time_in_hospital', 'n_lab_procedures', 'n_procedures', 'n_medications',
            'n_outpatient', 'n_inpatient', 'n_emergency'] + list(df.columns[df.columns.str.startswith(tuple(categorical_columns))])

X = df[features]
y = df['readmitted']

# Split dataset
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Standardize features
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# Train Logistic Regression model
model = LogisticRegression()
model.fit(X_train, y_train)

# Save model and scaler in the specific path
joblib.dump(model, os.path.join(SAVE_DIR, "readmission_model.pkl"))
joblib.dump(scaler, os.path.join(SAVE_DIR, "scaler.pkl"))
joblib.dump(features, os.path.join(SAVE_DIR, "feature_names.pkl"))

print(f"Model training completed. Model and scaler saved in {SAVE_DIR}")
