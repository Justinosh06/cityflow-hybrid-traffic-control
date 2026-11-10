import os
import joblib
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

def train_random_forest(dataset_path="data/lqf_dataset.csv", model_output_path="models/rf_pretrain.joblib"):
    print(f"Loading dataset from {dataset_path}...")
    if not os.path.exists(dataset_path):
        raise FileNotFoundError(f"Dataset not found at {dataset_path}. Run collector.py first!")
    
    df = pd.read_csv(dataset_path)
    
    # Separate features (X) and action labels (y)
    X = df.drop(columns=['action', 'step', 'intersection_id'], errors='ignore')
    y = df['action']
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    print("Training Random Forest Classifier for supervised pre-training...")
    rf_model = RandomForestClassifier(
        n_estimators=100,
        max_depth=12,
        random_state=42,
        n_jobs=-1
    )
    
    rf_model.fit(X_train, y_train)
    
    # Evaluate performance
    y_pred = rf_model.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    print(f"Pre-training complete! Test Accuracy: {acc * 100:.2f}%")
    
    # Save model artifact
    os.makedirs(os.path.dirname(model_output_path), exist_ok=True)
    joblib.dump(rf_model, model_output_path)
    print(f"Model saved successfully to {model_output_path}")
    
    return rf_model

if __name__ == "__main__":
    train_random_forest()