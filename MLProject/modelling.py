import argparse
import pandas as pd
import dagshub
import mlflow
import mlflow.sklearn
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

# Inisialisasi argument untuk input path
parser = argparse.ArgumentParser()
parser.add_argument("--data_path", type=str, required=True)
args = parser.parse_args()

# Inisialisasi DagsHub (otomatis menggunakan environment variables)
dagshub.init(mlflow=True)
mlflow.set_experiment("Food_Delivery_Classification")

# Load Data
df = pd.read_csv(args.data_path)
X = df.drop(columns=['Status_Pesanan'])
y = df['Status_Pesanan']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Training
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