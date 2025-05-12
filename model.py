import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
import joblib

# Load dataset
df = pd.read_csv("Training.csv")

# Handle missing values
df.fillna(0, inplace=True)  # or use df.dropna(inplace=True) if appropriate

# Check dataset info
print("Dataset shape:", df.shape)
print(df.head())

# Split features and labels
X = df.drop(columns=["prognosis"])
y = df["prognosis"]

# Encode labels
encoder = LabelEncoder()
y_encoded = encoder.fit_transform(y)

# Save symptom column names
symptom_columns = X.columns.tolist()
joblib.dump(symptom_columns, "symptom_columns.pkl")

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y_encoded, test_size=0.2, random_state=42)

# Train model
model = RandomForestClassifier(n_estimators=50, max_depth=5, random_state=42)
model.fit(X_train, y_train)

# Save model and encoder
joblib.dump(model, "disease_predictor_model.pkl")
joblib.dump(encoder, "label_encoder.pkl")

# Evaluate
train_accuracy = model.score(X_train, y_train)
test_accuracy = model.score(X_test, y_test)
print(f"✅ Train Accuracy: {train_accuracy * 100:.2f}%")
print(f"✅ Test Accuracy: {test_accuracy * 100:.2f}%")
