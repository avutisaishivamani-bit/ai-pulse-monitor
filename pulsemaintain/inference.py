import numpy as np                                                             # Numerical calculations and array shaping
import pandas as pd                                                            # Feature table creation and dataset handling
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier     # Core ML algorithms for RUL and Failure
from sklearn.metrics import mean_absolute_error, accuracy_score, precision_score, recall_score  # Validation metrics
from .model_registry import ModelRegistry                                      # Model artifact versioning and persistence
from .db import get_connection                                                 # SQLite connection helper

class InferenceEngine:
    def __init__(self):
        self.registry = ModelRegistry()       # Registry to store and load active .joblib model versions
        self.vib_history = {}                 # Buffer of last 10 vibration readings per asset (for rolling volatility)
        self.training_data = []               # Accumulates [features, real_rul, failure_48h] pairs for retraining
        
    def extract_features(self, asset_id, vib, temp, pres):
        # Feature Engineering: compute rolling standard deviation of vibration over past 10 hours
        if asset_id not in self.vib_history:
            self.vib_history[asset_id] = []
            
        self.vib_history[asset_id].append(vib)
        if len(self.vib_history[asset_id]) > 10:
            self.vib_history[asset_id].pop(0)  # Keep only 10 most recent vibration values
            
        # Rolling standard deviation measures mechanical chatter/instability
        rolling_std_vib = np.std(self.vib_history[asset_id]) if len(self.vib_history[asset_id]) == 10 else 0.0
        
        # Returns 4 engineered features: [vibration, temperature, pressure, rolling_std_vibration]
        return [vib, temp, pres, rolling_std_vib]

    def add_training_data(self, features, real_rul, failure_48h):
        # Accumulates real-world observed operational samples for model training / retraining
        self.training_data.append({
            "features": features,
            "real_rul": real_rul,
            "failure_48h": failure_48h
        })

    def train_models(self, hour):
        # Prevent training on insufficient sample sizes
        if len(self.training_data) < 50:
            print("[MLOPS] Not enough data to train models.")
            return

        # Inject synthetic edge cases so model learns high vibration/temperature failure patterns early
        synthetic = []
        for i in range(20):
            synthetic.append({
                "features": [6.0 + i*0.2, 90.0 + i*2, 30.0 - i*2, 2.0],
                "real_rul": max(0.0, float(20 - i)),
                "failure_48h": 1
            })
        
        # Assemble training dataset into feature matrix X and targets y
        df = pd.DataFrame(self.training_data + synthetic)
        X = np.stack(df['features'].values)
        y_rul = df['real_rul'].values
        y_fail = df['failure_48h'].values

        # 80/20 train/test split for validation evaluation
        split_idx = int(len(X) * 0.8)
        X_train, X_test = X[:split_idx], X[split_idx:]
        y_rul_train, y_rul_test = y_rul[:split_idx], y_rul[split_idx:]
        y_fail_train, y_fail_test = y_fail[:split_idx], y_fail[split_idx:]

        # MODEL 1: Remaining Useful Life (RUL) Continuous Predictor
        rul_model = RandomForestRegressor(n_estimators=50, random_state=42)  # 50 decision trees
        rul_model.fit(X_train, y_rul_train)                                  # Fit on training features
        rul_pred = rul_model.predict(X_test)                                 # Predict on test holdout
        rul_mae = mean_absolute_error(y_rul_test, rul_pred)                   # Evaluate Mean Absolute Error (MAE)

        # Register new RUL model version in Model_Version table and disk
        self.registry.register_model("RUL_Predictor", rul_model, hour, {"MAE": round(rul_mae, 2)})

        # MODEL 2: Failure Probability (48-Hour Binary Classifier)
        fail_model = RandomForestClassifier(n_estimators=50, random_state=42)  # 50 decision trees
        fail_model.fit(X_train, y_fail_train)                                  # Fit on binary failure targets
        fail_pred = fail_model.predict(X_test)                                 # Predict test labels
        acc = accuracy_score(y_fail_test, fail_pred)                          # Accuracy metric
        prec = precision_score(y_fail_test, fail_pred, zero_division=0)        # Precision metric
        rec = recall_score(y_fail_test, fail_pred, zero_division=0)           # Recall metric

        metrics = {
            "Accuracy": round(acc, 2),
            "Precision": round(prec, 2),
            "Recall": round(rec, 2)
        }
        # Register new Failure model version in Model_Version table and disk
        self.registry.register_model("Failure_Predictor", fail_model, hour, metrics)

    def predict(self, asset_id, hour, vib, temp, pres):
        # Real-time inference pipeline: extracts features, loads active models, and scores input
        features = self.extract_features(asset_id, vib, temp, pres)
        
        # Retrieve currently ACTIVE model versions from model registry
        rul_model, rul_ver, rul_vid = self.registry.get_active_model("RUL_Predictor")
        fail_model, fail_ver, fail_vid = self.registry.get_active_model("Failure_Predictor")
        
        predicted_rul = 0.0
        failure_prob = 0.0
        
        if rul_model and fail_model:
            X = np.array([features])
            predicted_rul = rul_model.predict(X)[0]               # Continuous RUL estimate in hours
            proba = fail_model.predict_proba(X)[0]                # Probability array [prob_healthy, prob_failure]
            if 1 in fail_model.classes_:
                idx = list(fail_model.classes_).index(1)
                failure_prob = proba[idx]                          # Probability of failure (class 1)
            else:
                failure_prob = 0.0
            
            # Log prediction output and model version ID into Prediction_Log table
            conn = get_connection()
            cursor = conn.cursor()
            cursor.execute('''
                INSERT INTO Prediction_Log (
                    asset_id, hour, predicted_rul, failure_probability, 
                    model_version_rul, model_version_prob
                ) VALUES (?, ?, ?, ?, ?, ?)
            ''', (asset_id, hour, float(predicted_rul), float(failure_prob), rul_vid, fail_vid))
            conn.commit()
            conn.close()
            
        return float(predicted_rul), float(failure_prob), features
