import numpy as np                               # Math and histogram functions
from scipy.stats import ks_2samp                   # Kolmogorov-Smirnov 2-sample test for continuous distribution divergence
from .db import get_connection                      # SQLite connection helper
from .config import PSI_THRESHOLD                   # PSI threshold (0.25) beyond which distribution is unstable

# Calculates Population Stability Index (PSI) between a baseline reference distribution and recent operational data
def calculate_psi(expected, actual, buckets=10):
    # Splits expected distribution into 10 decile buckets (percentiles)
    def build_buckets(data, buckets):
        breaks = np.linspace(0, 100, buckets + 1)
        return np.percentile(data, breaks)
        
    if len(expected) == 0 or len(actual) == 0:
        return 0.0
        
    breakpoints = build_buckets(expected, buckets)
    
    # Calculate percentage of data points falling in each bucket for reference vs actual
    expected_percents = np.histogram(expected, breakpoints)[0] / len(expected)
    actual_percents = np.histogram(actual, breakpoints)[0] / len(actual)
    
    # Sub-PSI calculation formula: (Actual% - Expected%) * ln(Actual% / Expected%)
    def sub_psi(e, a):
        if a == 0:
            a = 0.0001   # Epsilon smoothing to prevent log(0) undefined error
        if e == 0:
            e = 0.0001   # Epsilon smoothing to prevent division by 0
        return (e - a) * np.log(e / a)
        
    # Total PSI is the sum over all 10 buckets
    return np.sum([sub_psi(e, a) for e, a in zip(expected_percents, actual_percents)])

class DriftMonitor:
    def __init__(self):
        self.consecutive_drift_count = 0  # Debouncing counter to prevent retraining on a single outlier spike
        self.cooldown_hours = 0           # Cooldown window to prevent retraining loops right after a retrain
        
    def check_drift(self, hour):
        # If in cooldown, decrement hours and skip check
        if self.cooldown_hours > 0:
            self.cooldown_hours -= 1
            return False
            
        conn = get_connection()
        cursor = conn.cursor()
        
        # Reference dataset: baseline model predictions prior to current hour (up to 500 records)
        cursor.execute('''
            SELECT predicted_rul FROM Prediction_Log 
            WHERE hour < ? ORDER BY id ASC LIMIT 500
        ''', (hour,))
        reference_data = [row[0] for row in cursor.fetchall()]
        
        # Recent dataset: predictions generated across the last 20 operational hours
        cursor.execute('''
            SELECT predicted_rul FROM Prediction_Log 
            WHERE hour >= ? AND hour < ?
        ''', (hour - 20, hour))
        recent_data = [row[0] for row in cursor.fetchall()]
        
        should_retrain = False
        
        # Run statistical tests only when sufficient sample sizes exist
        if len(reference_data) >= 100 and len(recent_data) >= 20:
            psi = calculate_psi(reference_data, recent_data)              # Compute Population Stability Index
            ks_stat, ks_pvalue = ks_2samp(reference_data, recent_data)    # Compute 2-sample Kolmogorov-Smirnov test
            
            is_drift = int(psi > PSI_THRESHOLD)                           # 1 if PSI > 0.25 (Drift Detected), else 0
            
            # Save drift metrics into Model_Drift_Metric table for dashboard visualization
            cursor.execute('''
                INSERT INTO Model_Drift_Metric (hour, model_name, psi_value, ks_stat, ks_pvalue, is_drift_detected)
                VALUES (?, ?, ?, ?, ?, ?)
            ''', (hour, 'Failure_Predictor', psi, ks_stat, ks_pvalue, is_drift))
            
            if is_drift:
                self.consecutive_drift_count += 1
                print(f"[DRIFT] Hour {hour}: PSI={psi:.3f} > {PSI_THRESHOLD}. Consecutive drifts: {self.consecutive_drift_count}")
                # Trigger retraining ONLY after 2 consecutive drift detections to prevent false alarms
                if self.consecutive_drift_count >= 2:
                    should_retrain = True
                    self.consecutive_drift_count = 0
                    self.cooldown_hours = 50   # Enter 50-hour cooldown before running another retrain
            else:
                self.consecutive_drift_count = 0  # Reset counter if distribution returns to normal
                
        conn.commit()
        conn.close()
        
        return should_retrain

