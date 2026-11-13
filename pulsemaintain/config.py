import os

# Base directory paths
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))  # Root folder path of the project
DB_PATH = os.path.join(BASE_DIR, "data", "pulsemaintain.db")           # SQLite database file location
MODEL_DIR = os.path.join(BASE_DIR, "models")                           # Directory where trained ML models (.joblib) are stored

# Simulation config
TOTAL_HOURS = 400       # Total hours the industrial simulation runs (full lifecycle)
DRIFT_HOUR = 200        # The simulated hour when sensor calibration drift is artificially injected
RANDOM_SEED = None      # Set to None for fresh, unique random data on every run (or integer for fixed seed)

# Filter config
MA_WINDOW_SIZE = 5      # Moving Average window size (averages last 5 readings to remove sensor noise)

# Drift config
DRIFT_CHECK_INTERVAL = 20  # Check for statistical data drift every 20 hours
PSI_THRESHOLD = 0.25       # Population Stability Index limit (>0.25 indicates significant data drift)

# Alerts config
PROBABILITY_THRESHOLD = 0.80  # Trigger alert if failure probability > 80% (0.80)
RUL_THRESHOLD = 150           # Trigger alert if Remaining Useful Life < 150 hours


# Machine config (8 industrial assets monitored in factory)
MACHINES = [
    {"id": "M01", "name": "CNC Milling Machine"},     # High degradation asset (triggers alerts early)
    {"id": "M02", "name": "Industrial Water Pump"},   # Normal operational asset
    {"id": "M03", "name": "Steam Turbine"},           # Normal operational asset
    {"id": "M04", "name": "Air Compressor"},          # High degradation asset (triggers alerts early)
    {"id": "M05", "name": "Hydraulic Press"},         # Normal operational asset
    {"id": "M06", "name": "Conveyor Motor"},          # Normal operational asset
    {"id": "M07", "name": "Cooling Fan"},             # High degradation asset (triggers alerts early)
    {"id": "M08", "name": "Industrial Generator"},    # Normal operational asset
]

# Base sensor settings (normal healthy baseline values)
SENSOR_BASE = {
    "vibration": {"mean": 2.5, "std": 0.5},    # Vibration baseline in mm/s (mean: 2.5, std dev: 0.5)
    "temperature": {"mean": 45.0, "std": 2.0},  # Temperature baseline in °C (mean: 45.0, std dev: 2.0)
    "pressure": {"mean": 120.0, "std": 5.0}     # Pressure baseline in PSI (mean: 120.0, std dev: 5.0)
}
