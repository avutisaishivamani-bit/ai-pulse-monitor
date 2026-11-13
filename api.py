from fastapi import FastAPI, HTTPException  # FastAPI framework and error handling
from pydantic import BaseModel                # Data validation for incoming JSON payloads
import sqlite3                                # Database connector for SQLite
import os                                     # File system checks
from typing import List, Optional             # Type hinting support
from pulsemaintain.config import DB_PATH      # Path to the pulsemaintain.db SQLite file
from pulsemaintain.ingestion import DataIngestion
from pulsemaintain.simulator import Simulator

app = FastAPI(title="PulseMaintain AI API")    # Initialize FastAPI application instance

def get_db():
    # Helper function to open and return an active database connection
    if not os.path.exists(DB_PATH):
        raise HTTPException(status_code=500, detail="Database not initialized")
    conn = sqlite3.connect(DB_PATH)           # Connect to SQLite file
    conn.row_factory = sqlite3.Row            # Return rows as dictionary-like objects (accessible by column name)
    return conn

# Schema for incoming sensor telemetry POST requests
class TelemetryInput(BaseModel):
    asset_id: str      # Machine ID, e.g., 'M01'
    hour: int          # Operational timestamp hour
    vibration: float   # Vibration reading
    temperature: float # Temperature reading in Celsius
    pressure: float    # Pressure reading in PSI

# Endpoint 1: Health check endpoint to verify API server is up
@app.get("/health")
def health_check():
    return {"status": "healthy", "version": "1.0.0"}

# Endpoint 2: Fetch list of all industrial assets and their current health score
@app.get("/assets")
def get_assets():
    conn = get_db()
    cursor = conn.cursor()
    cursor.execute("SELECT * FROM Industrial_Asset")   # Query all asset rows
    assets = [dict(row) for row in cursor.fetchall()] # Convert query rows to Python dictionaries
    conn.close()
    return assets

# Endpoint 3: Ingest real-time external sensor telemetry via HTTP POST
@app.post("/telemetry")
def post_telemetry(data: TelemetryInput):
    conn = get_db()
    cursor = conn.cursor()
    # Check if the target machine exists in the database
    cursor.execute("SELECT asset_id FROM Industrial_Asset WHERE asset_id = ?", (data.asset_id,))
    if not cursor.fetchone():
        conn.close()
        raise HTTPException(status_code=404, detail="Asset not found")  # Return 404 if invalid machine
        
    # Insert raw and filtered telemetry record into the database
    cursor.execute('''
        INSERT INTO Telemetry_Data (asset_id, hour, vibration_raw, temperature_raw, pressure_raw, vibration_filtered, temperature_filtered, pressure_filtered)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?)
    ''', (data.asset_id, data.hour, data.vibration, data.temperature, data.pressure, data.vibration, data.temperature, data.pressure))
    
    conn.commit()  # Save transaction
    conn.close()   # Close connection
    return {"status": "success", "message": "Telemetry received"}

# Endpoint 4: Fetch alerts, with optional filter by status ('OPEN' or 'CLOSED')
@app.get("/alerts")
def get_alerts(status: Optional[str] = None):
    conn = get_db()
    cursor = conn.cursor()
    if status:
        cursor.execute("SELECT * FROM Maintenance_Alert WHERE status = ?", (status,))  # Filter by status
    else:
        cursor.execute("SELECT * FROM Maintenance_Alert")                              # Return all alerts
    alerts = [dict(row) for row in cursor.fetchall()]
    conn.close()
    return alerts

# Endpoint 5: Fetch drift metrics history (PSI and KS-test statistics)
@app.get("/drift")
def get_drift():
    conn = get_db()
    cursor = conn.cursor()
    cursor.execute("SELECT * FROM Model_Drift_Metric ORDER BY hour DESC")  # Most recent drift checks first
    drift_metrics = [dict(row) for row in cursor.fetchall()]
    conn.close()
    return drift_metrics
