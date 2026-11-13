import numpy as np                                            # Numerical library for Gaussian noise and distributions
from .config import RANDOM_SEED, TOTAL_HOURS, DRIFT_HOUR, SENSOR_BASE, MACHINES  # Simulation parameters

class Simulator:
    def __init__(self, seed=None):
        from . import config
        target_seed = seed if seed is not None else config.RANDOM_SEED
        if target_seed is not None:
            np.random.seed(target_seed)
            self.fast_degraders = {'M01', 'M04', 'M07'}
        else:
            # Dynamic random mode: fresh seed from OS and random fast degraders every run
            all_ids = [m["id"] for m in config.MACHINES]
            num_fast = int(np.random.randint(2, 4))
            self.fast_degraders = set(np.random.choice(all_ids, size=num_fast, replace=False))
            
        self.machines_state = {}  # Tracks each asset's current health score and wear rate
        
    def init_machine(self, asset_id):
        # Fast degraders wear out rapidly (0.75 - 1.4% per hour), while others degrade normally (0.1 - 0.35%)
        if asset_id in self.fast_degraders:
            deg_rate = float(np.random.uniform(0.75, 1.4))   # Fast wear rate
        else:
            deg_rate = float(np.random.uniform(0.1, 0.35))   # Normal slow wear rate
            
        self.machines_state[asset_id] = {
            "health": 100.0,              # Start at 100% factory health
            "degradation_rate": deg_rate  # Health loss per simulated operational hour
        }
        
    def generate_readings(self, asset_id, hour):
        if asset_id not in self.machines_state:
            self.init_machine(asset_id)  # Auto-initialize state if first time seeing machine
            
        state = self.machines_state[asset_id]
        
        # Physical degradation: reduce health score by degradation rate (minimum bound 0.0)
        state["health"] = max(0.0, state["health"] - state["degradation_rate"])
        health_factor = (100 - state["health"]) / 100.0  # Normalized wear: 0.0 (healthy) -> 1.0 (failed)
        
        # Physics degradation simulation with Gaussian noise:
        # Vibration increases as bearings loosen (+2.0 mm/s max shift)
        vib = np.random.normal(SENSOR_BASE["vibration"]["mean"] + health_factor * 2, SENSOR_BASE["vibration"]["std"])
        # Temperature rises as friction increases (+20.0 °C max shift)
        temp = np.random.normal(SENSOR_BASE["temperature"]["mean"] + health_factor * 20, SENSOR_BASE["temperature"]["std"])
        # Pressure drops as seals and valves wear (-30.0 PSI max drop)
        pres = np.random.normal(SENSOR_BASE["pressure"]["mean"] - health_factor * 30, SENSOR_BASE["pressure"]["std"])
        
        # MLOps Drift Injection: at hour >= 200, simulate sensor calibration bias drift
        if hour >= DRIFT_HOUR:
            vib += 1.6   # Sensor calibration offset (false high vibration)
            temp += 14.0  # Sensor calibration offset (false high temperature)
            pres -= 9.0   # Sensor calibration offset (false low pressure)
            
        # Ground Truth Calculation (used for supervised training labels):
        # RUL: remaining hours until health reaches 0 based on current linear degradation rate
        real_rul = state["health"] / state["degradation_rate"] if state["degradation_rate"] > 0 else 0
        # Binary target: 1 if asset will fail within next 48 hours, else 0
        failure_48h = 1 if real_rul <= 48 else 0
            
        return {
            "vibration": round(vib, 4),          # Raw vibration reading
            "temperature": round(temp, 4),      # Raw temperature reading
            "pressure": round(pres, 4),          # Raw pressure reading
            "health": round(state["health"], 2), # Ground-truth health %
            "real_rul": round(real_rul, 2),      # Ground-truth remaining hours
            "failure_48h": failure_48h           # Ground-truth binary failure label
        }
