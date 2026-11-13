import time                                                    # Measure simulation execution time
import sys
from pulsemaintain.db import init_db, reset_db, get_connection     # DB setup and connection utilities
from pulsemaintain.config import TOTAL_HOURS                       # Total simulation duration (400 hours)
from pulsemaintain.pipeline import SimulationPipeline              # Main orchestration pipeline class
import contextlib

def run_simulation():
    # Clean slate: wipe any old database file and recreate tables with seed assets & technicians
    reset_db()
    init_db()
    
    print("[EVENT] Starting PulseMaintain AI Simulation")
    print(f"[EVENT] Target: {TOTAL_HOURS} operating hours")
    
    pipeline = SimulationPipeline()  # Instantiate pipeline (simulator, ingestion, ML models, drift, alerts)
    
    start_time = time.time()         # Start timer
    
    # Main simulation loop: iterate through 400 simulated operational hours
    for hour in range(1, TOTAL_HOURS + 1):
        pipeline.run_hour(hour)      # Execute ingestion, feature extraction, ML inference, drift & alerts
        if hour % 100 == 0:          # Print milestone progress every 100 hours
            print(f"[EVENT] Completed {hour}/{TOTAL_HOURS} simulated hours.")
            
    runtime = time.time() - start_time
    print(f"\n[EVENT] Simulation completed in {runtime:.2f} seconds.")
    
    # Query database to display overall system statistics after the 400-hour run
    conn = get_connection()
    cursor = conn.cursor()
    
    print("\n--- Simulation Summary ---")
    print(f"Total simulated hours: {TOTAL_HOURS}")
    
    cursor.execute("SELECT COUNT(*) FROM Industrial_Asset")   # Verify 8 machines were created
    print(f"Asset count: {cursor.fetchone()[0]}")
    
    cursor.execute("SELECT COUNT(*) FROM Sensor")             # Verify 24 sensors (3 per machine)
    print(f"Sensor count: {cursor.fetchone()[0]}")
    
    cursor.execute("SELECT COUNT(*) FROM Telemetry_Data")     # 400 hours x 8 machines = 3,200 telemetry points
    print(f"Telemetry records: {cursor.fetchone()[0]}")
    
    cursor.execute("SELECT COUNT(*) FROM Prediction_Log")     # Prediction records generated from hour 50 to 400
    print(f"Prediction records: {cursor.fetchone()[0]}")
    
    cursor.execute("SELECT COUNT(*) FROM Maintenance_Alert")  # Number of critical alerts triggered
    print(f"Alert count: {cursor.fetchone()[0]}")
    
    cursor.execute("SELECT COUNT(*) FROM Work_Order")         # Work orders dispatched to technicians
    print(f"Work order count: {cursor.fetchone()[0]}")
    
    # Display registered ML model versions (v1 initial, v2 retrained after drift)
    cursor.execute("SELECT model_name, version_number, status, metrics FROM Model_Version")
    print("\n--- ML Models ---")
    for row in cursor.fetchall():
        print(f"[{row['model_name']}] v{row['version_number']} ({row['status']}) - Metrics: {row['metrics']}")
        
    # Display statistical data drift checks and confirmations
    cursor.execute("SELECT COUNT(*) FROM Model_Drift_Metric")
    drift_checks = cursor.fetchone()[0]
    cursor.execute("SELECT COUNT(*) FROM Model_Drift_Metric WHERE is_drift_detected = 1")
    drift_detected = cursor.fetchone()[0]
    print(f"\n--- MLOps ---")
    print(f"Drift checks performed: {drift_checks}")
    print(f"Drift events detected: {drift_detected}")
    
    conn.close()

if __name__ == "__main__":
    run_simulation()  # Execute simulation when run as a script

