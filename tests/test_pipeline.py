import os
import pytest
import sqlite3
import numpy as np
from unittest.mock import patch
from pulsemaintain import config
from pulsemaintain.db import init_db, get_connection
from pulsemaintain.simulator import Simulator
from pulsemaintain.ingestion import DataIngestion
from pulsemaintain.drift import calculate_psi, DriftMonitor
from pulsemaintain.alerts import AlertManager
from pulsemaintain.pipeline import SimulationPipeline

# Override config for tests
TEST_DB_PATH = os.path.join(config.BASE_DIR, "data", "test_pulsemaintain.db")
config.DB_PATH = TEST_DB_PATH
config.RANDOM_SEED = 42

@pytest.fixture(autouse=True)
def setup_db():
    if os.path.exists(TEST_DB_PATH):
        os.remove(TEST_DB_PATH)
    init_db()
    yield
    if os.path.exists(TEST_DB_PATH):
        os.remove(TEST_DB_PATH)

def test_db_initialization():
    conn = get_connection()
    cursor = conn.cursor()
    cursor.execute("SELECT name FROM sqlite_master WHERE type='table'")
    tables = [row[0] for row in cursor.fetchall()]
    assert 'Industrial_Asset' in tables
    assert 'Work_Order' in tables
    
    # Check foreign keys by ensuring constraints are on (pragma foreign_keys = ON by default in standard setups, or we just trust our schema)
    cursor.execute("SELECT COUNT(*) FROM Technician")
    assert cursor.fetchone()[0] == 3
    conn.close()

def test_moving_average_filtering():
    simulator = Simulator()
    ingestion = DataIngestion(simulator)
    
    # Force some readings
    vib_readings = [2.0, 2.5, 3.0, 3.5, 4.0, 4.5] # 6 readings
    
    with patch.object(simulator, 'generate_readings') as mock_gen:
        for i, v in enumerate(vib_readings):
            mock_gen.return_value = {
                "vibration": v, "temperature": 45.0, "pressure": 120.0,
                "health": 100.0, "real_rul": 200.0, "failure_48h": 0
            }
            records = ingestion.process_hour(i+1)
            
            if i == 4: # Window is full (size 5) -> avg of 2, 2.5, 3, 3.5, 4 = 3.0
                assert records[0]['vibration_filtered'] == 3.0
            if i == 5: # Avg of 2.5, 3, 3.5, 4, 4.5 = 3.5
                assert records[0]['vibration_filtered'] == 3.5

def test_psi_calculation():
    # Test identical distributions
    ref = np.random.normal(0, 1, 1000)
    actual = np.random.normal(0, 1, 1000)
    psi = calculate_psi(ref, actual)
    assert psi < 0.1 # Should be very low
    
    # Test different distributions
    actual_shifted = np.random.normal(2, 1, 1000)
    psi_shifted = calculate_psi(ref, actual_shifted)
    assert psi_shifted > 0.25 # Should trigger drift

def test_alert_thresholds_and_duplicate_prevention():
    alert_manager = AlertManager()
    
    # 1. Trigger alert
    alert_manager.evaluate_risk('M01', 1, predicted_rul=10.0, failure_prob=0.9)
    
    conn = get_connection()
    cursor = conn.cursor()
    cursor.execute("SELECT * FROM Maintenance_Alert WHERE asset_id='M01'")
    alerts = cursor.fetchall()
    assert len(alerts) == 1
    assert alerts[0]['status'] == 'OPEN'
    
    # 2. Prevent duplicate open alert
    alert_manager.evaluate_risk('M01', 2, predicted_rul=5.0, failure_prob=0.95)
    cursor.execute("SELECT * FROM Maintenance_Alert WHERE asset_id='M01'")
    alerts = cursor.fetchall()
    assert len(alerts) == 1 # Still 1
    
    conn.close()

def test_consecutive_drift_detection():
    monitor = DriftMonitor()
    
    # Mock data directly in DB for testing check_drift
    conn = get_connection()
    cursor = conn.cursor()
    
    # Insert 100 ref predictions
    for i in range(1, 101):
        cursor.execute("INSERT INTO Prediction_Log (asset_id, hour, predicted_rul, failure_probability, model_version_rul, model_version_prob) VALUES ('M01', ?, 100, 0.1, 1, 1)", (i,))
        
    # Insert 50 shifted predictions for recent data
    for i in range(101, 151):
        cursor.execute("INSERT INTO Prediction_Log (asset_id, hour, predicted_rul, failure_probability, model_version_rul, model_version_prob) VALUES ('M01', ?, 10, 0.9, 1, 1)", (i,))
        
    conn.commit()
    conn.close()
    
    # Check 1 (Drift detected, should_retrain=False because it's first consecutive)
    retrain1 = monitor.check_drift(151)
    assert retrain1 == False
    assert monitor.consecutive_drift_count == 1
    
    # Check 2 (Drift detected again, should_retrain=True)
    retrain2 = monitor.check_drift(151) # same data
    assert retrain2 == True
    assert monitor.consecutive_drift_count == 0

def test_reproducible_simulation():
    # Because RANDOM_SEED=42, running it twice should give identical results
    simulator1 = Simulator()
    simulator2 = Simulator()
    
    # Reset seed to ensure deterministic behavior before first run
    np.random.seed(config.RANDOM_SEED)
    res1 = simulator1.generate_readings('M01', 1)
    
    # Reset seed to simulate a fresh run
    np.random.seed(config.RANDOM_SEED)
    res2 = simulator2.generate_readings('M01', 1)
    
    assert res1['vibration'] == res2['vibration']
