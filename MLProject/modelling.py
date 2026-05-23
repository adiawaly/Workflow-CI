import argparse
import pandas as pd
import os
import mlflow
import mlflow.sklearn
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

# 1. Setup Argumen
parser = argparse.ArgumentParser()
parser.add_argument("--data_path", type=str, required=True)
args = parser.parse_args()

# 2. Inisialisasi Tracking MLflow Murni (Tanpa dagshub.init)
# MLflow akan otomatis menggunakan MLFLOW_TRACKING_USERNAME & PASSWORD dari GitHub Secrets
tracking_uri = os.getenv("MLFLOW_TRACKING_URI", "https://dagshub.com/adiawaly/MLOps_Food_Delivery.mlflow")
mlflow.set_tracking_uri(tracking_uri)
mlflow.set_experiment("Food_Delivery_Classification")

# 3. Load Data
df = pd.read_csv(args.data_path)
X = df.drop(columns=['Status_Pesanan'])
y = df['Status_Pesanan']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 4. Training
with mlflow.start_run():
    rf = RandomForestClassifier(n_estimators=100, max_depth=5)
    rf.fit(X_train, y_train)
    
    # Log Metric
    y_pred = rf.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    mlflow.log_metric("accuracy", acc)
    
    # Log Model
    mlflow.sklearn.log_model(rf, "model")
    print(f"Training selesai. Akurasi: {acc}")