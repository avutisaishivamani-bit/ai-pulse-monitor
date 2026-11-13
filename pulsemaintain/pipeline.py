from .config import DRIFT_CHECK_INTERVAL  # Import how often to check for data drift (every 20h)
from .simulator import Simulator             # Generates synthetic physical sensor readings and degradation
from .ingestion import DataIngestion         # Applies moving average filter and saves telemetry to DB
from .inference import InferenceEngine       # Trains models and generates RUL & failure predictions
from .drift import DriftMonitor              # Tracks Population Stability Index (PSI) to detect drift
from .alerts import AlertManager             # Evaluates thresholds and automatically assigns work orders

class SimulationPipeline:
    def __init__(self):
        # Initialize all sub-modules that form the end-to-end MLOps architecture
        self.simulator = Simulator()                     # Physics and sensor degradation simulator
        self.ingestion = DataIngestion(self.simulator)   # Ingestion layer with Moving Average smoothing
        self.inference = InferenceEngine()               # ML inference & retraining engine
        self.drift_monitor = DriftMonitor()             # Statistical drift detection monitor
        self.alert_manager = AlertManager()             # Alerting & technician dispatch manager
        
    def run_hour(self, hour):
        # STEP 1: Ingestion - Generate raw sensor readings, filter noise, save to database
        data_records = self.ingestion.process_hour(hour)
        
        # STEP 2: Cold-Start Training - Wait until 50 hours of data accumulate before initial ML training
        if hour == 50:
            print(f"[MLOPS] Hour {hour}: Initial model training...")
            self.inference.train_models(hour)  # Train Random Forest models on accumulated telemetry
        
        # Process each machine's hourly readings
        for record in data_records:
            asset_id = record["asset_id"]
            vib = record["vibration_filtered"]      # Noise-filtered vibration
            temp = record["temperature_filtered"]   # Noise-filtered temperature
            pres = record["pressure_filtered"]      # Noise-filtered pressure
            
            # Feature Engineering: Extract telemetry features and append to retraining pool
            features = self.inference.extract_features(asset_id, vib, temp, pres)
            self.inference.add_training_data(features, record["real_rul"], record["failure_48h"])
            
            # STEP 3: Real-Time Inference (once models are trained at hour >= 50)
            if hour >= 50:
                # Predict Remaining Useful Life (RUL) and 48-hour failure probability
                predicted_rul, failure_prob, _ = self.inference.predict(asset_id, hour, vib, temp, pres)
                
                # STEP 4: Automated Risk Evaluation & Maintenance Dispatch
                self.alert_manager.evaluate_risk(asset_id, hour, predicted_rul, failure_prob)
                
        # STEP 5: MLOps Drift Monitoring - Run statistical check every 20 hours after hour 100
        if hour >= 100 and hour % DRIFT_CHECK_INTERVAL == 0:
            should_retrain = self.drift_monitor.check_drift(hour)  # Calculates PSI against baseline
            if should_retrain:
                print(f"[MLOPS] Hour {hour}: Drift confirmed. Initiating model retraining...")
                self.inference.train_models(hour)  # Retrain models with new drifted distribution
