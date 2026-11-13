import os                                     # File and directory creation
import json                                   # Serializing performance metrics dict to JSON string
import joblib                                 # Serializing scikit-learn models to binary .joblib files
from .db import get_connection                # SQLite database connection helper
from .config import MODEL_DIR                 # Directory path for model artifacts (/models)

os.makedirs(MODEL_DIR, exist_ok=True)         # Ensure models/ folder exists on filesystem

class ModelRegistry:
    def __init__(self):
        # In-memory cache to avoid re-reading .joblib files from disk thousands of times during loops
        self._cached_models = {}

    def get_active_model(self, model_name):
        # Fast-path: Check if we already have the active model in memory
        if model_name in self._cached_models:
            cached = self._cached_models[model_name]
            return cached["model"], cached["version_number"], cached["version_id"]

        # Queries SQLite for the latest model version tagged with status='ACTIVE'
        conn = get_connection()
        cursor = conn.cursor()
        cursor.execute('''
            SELECT version_number, artifact_path, version_id
            FROM Model_Version
            WHERE model_name = ? AND status = 'ACTIVE'
            ORDER BY version_number DESC LIMIT 1
        ''', (model_name,))
        row = cursor.fetchone()
        conn.close()
        
        if not row:
            return None, None, None           # No active model found (e.g. cold start before hour 50)
            
        model = joblib.load(row['artifact_path'])  # Deserializes the trained sklearn model from disk
        self._cached_models[model_name] = {
            "model": model,
            "version_number": row['version_number'],
            "version_id": row['version_id']
        }
        return model, row['version_number'], row['version_id']

    def register_model(self, model_name, model_obj, hour, metrics):
        conn = get_connection()
        cursor = conn.cursor()
        
        # Calculate auto-incremented version number (v1, v2, v3...)
        cursor.execute('SELECT MAX(version_number) FROM Model_Version WHERE model_name = ?', (model_name,))
        row = cursor.fetchone()
        next_version = (row[0] or 0) + 1
        
        # Zero-downtime model promotion: Retire previously active model versions
        cursor.execute('''
            UPDATE Model_Version
            SET status = 'RETIRED'
            WHERE model_name = ? AND status = 'ACTIVE'
        ''', (model_name,))
        
        # Save model weights to serialized .joblib file artifact
        artifact_path = os.path.join(MODEL_DIR, f"{model_name}_v{next_version}.joblib")
        joblib.dump(model_obj, artifact_path)
        
        # Record new model entry in Model_Version table with ACTIVE status and validation metrics
        cursor.execute('''
            INSERT INTO Model_Version (model_name, version_number, training_hour, artifact_path, metrics, status)
            VALUES (?, ?, ?, ?, ?, 'ACTIVE')
        ''', (model_name, next_version, hour, artifact_path, json.dumps(metrics)))
        new_version_id = cursor.lastrowid
        
        # Update in-memory cache with the new active model immediately
        self._cached_models[model_name] = {
            "model": model_obj,
            "version_number": next_version,
            "version_id": new_version_id
        }
        
        print(f"[MLOPS] Registered new active model version: {model_name} v{next_version}")
        conn.commit()
        conn.close()
