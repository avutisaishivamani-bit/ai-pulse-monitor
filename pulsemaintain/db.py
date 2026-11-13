import sqlite3                                 # Built-in lightweight SQL database engine
import os                                      # Path verification and file removal
from .config import DB_PATH, MACHINES          # Database file location and machine seed list

def get_connection():
    # Helper to return a high-performance SQLite connection configured with WAL mode
    os.makedirs(os.path.dirname(DB_PATH), exist_ok=True)
    conn = sqlite3.connect(DB_PATH)
    conn.execute("PRAGMA synchronous = OFF;")  # High speed disk write optimization
    conn.execute("PRAGMA journal_mode = WAL;")  # Write-Ahead Logging allows concurrent readers and writers
    conn.row_factory = sqlite3.Row             # Enables column access by dictionary name (e.g., row['name'])
    return conn

def init_db():
    conn = get_connection()
    cursor = conn.cursor()
    
    # TABLE 1: Industrial_Asset - Stores machines, live health scores (0-100), and accumulated running hours
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS Industrial_Asset (
            asset_id TEXT PRIMARY KEY,
            name TEXT NOT NULL,
            health_score REAL DEFAULT 100.0,
            operating_hours INTEGER DEFAULT 0
        )
    ''')
    
    # TABLE 2: Sensor - Tracks individual sensors (Vibration, Temperature, Pressure) mapped to assets
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS Sensor (
            sensor_id TEXT PRIMARY KEY,
            asset_id TEXT,
            sensor_type TEXT,
            FOREIGN KEY(asset_id) REFERENCES Industrial_Asset(asset_id)
        )
    ''')
    
    # TABLE 3: Telemetry_Data - High-frequency time-series table storing raw and moving-average filtered readings
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS Telemetry_Data (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            asset_id TEXT,
            hour INTEGER,
            vibration_raw REAL,
            temperature_raw REAL,
            pressure_raw REAL,
            vibration_filtered REAL,
            temperature_filtered REAL,
            pressure_filtered REAL,
            FOREIGN KEY(asset_id) REFERENCES Industrial_Asset(asset_id)
        )
    ''')
    
    # TABLE 4: ML_Model - High-level model definitions (RUL_Predictor, Failure_Predictor)
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS ML_Model (
            model_name TEXT PRIMARY KEY,
            model_type TEXT NOT NULL
        )
    ''')
    
    # TABLE 5: Model_Version - Model Registry table tracking version numbers, training hours, file paths, and metrics
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS Model_Version (
            version_id INTEGER PRIMARY KEY AUTOINCREMENT,
            model_name TEXT,
            version_number INTEGER,
            training_hour INTEGER,
            artifact_path TEXT,
            metrics TEXT,
            status TEXT,
            FOREIGN KEY(model_name) REFERENCES ML_Model(model_name)
        )
    ''')
    
    # TABLE 6: Prediction_Log - Historical predictions log (RUL in hours and failure probability) with model version lineage
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS Prediction_Log (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            asset_id TEXT,
            hour INTEGER,
            predicted_rul REAL,
            failure_probability REAL,
            model_version_rul INTEGER,
            model_version_prob INTEGER
        )
    ''')
    
    # TABLE 7: Model_Drift_Metric - MLOps drift audit trail (PSI values, KS-test p-values, drift flags)
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS Model_Drift_Metric (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            hour INTEGER,
            model_name TEXT,
            psi_value REAL,
            ks_stat REAL,
            ks_pvalue REAL,
            is_drift_detected INTEGER
        )
    ''')
    
    # TABLE 8: Maintenance_Alert - Critical alerts triggered when machines cross risk thresholds
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS Maintenance_Alert (
            alert_id INTEGER PRIMARY KEY AUTOINCREMENT,
            asset_id TEXT,
            hour INTEGER,
            reason TEXT,
            severity TEXT,
            status TEXT DEFAULT 'OPEN',
            FOREIGN KEY(asset_id) REFERENCES Industrial_Asset(asset_id)
        )
    ''')
    
    # TABLE 9: Technician - Available maintenance personnel for work order assignment
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS Technician (
            technician_id TEXT PRIMARY KEY,
            name TEXT NOT NULL
        )
    ''')
    
    # TABLE 10: Work_Order - Actionable maintenance jobs assigned to technicians for alerted assets
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS Work_Order (
            work_order_id TEXT PRIMARY KEY,
            asset_id TEXT,
            alert_id INTEGER,
            technician_id TEXT,
            creation_hour INTEGER,
            status TEXT DEFAULT 'OPEN',
            FOREIGN KEY(asset_id) REFERENCES Industrial_Asset(asset_id),
            FOREIGN KEY(alert_id) REFERENCES Maintenance_Alert(alert_id),
            FOREIGN KEY(technician_id) REFERENCES Technician(technician_id)
        )
    ''')
    
    conn.commit()
    
    # Seed Initial Technicians (Alice, Bob, Charlie) if table is empty
    cursor.execute("SELECT COUNT(*) FROM Technician")
    if cursor.fetchone()[0] == 0:
        cursor.executemany("INSERT INTO Technician (technician_id, name) VALUES (?, ?)", [
            ('T01', 'Alice Engineer'),
            ('T02', 'Bob Mechanic'),
            ('T03', 'Charlie Fixer')
        ])
        
    # Seed Initial 8 Industrial Assets and their 24 attached Sensors
    cursor.execute("SELECT COUNT(*) FROM Industrial_Asset")
    if cursor.fetchone()[0] == 0:
        for m in MACHINES:
            cursor.execute("INSERT INTO Industrial_Asset (asset_id, name) VALUES (?, ?)", (m['id'], m['name']))
            cursor.execute("INSERT INTO Sensor (sensor_id, asset_id, sensor_type) VALUES (?, ?, ?)", (f"VIB-{m['id']}", m['id'], 'vibration'))
            cursor.execute("INSERT INTO Sensor (sensor_id, asset_id, sensor_type) VALUES (?, ?, ?)", (f"TEMP-{m['id']}", m['id'], 'temperature'))
            cursor.execute("INSERT INTO Sensor (sensor_id, asset_id, sensor_type) VALUES (?, ?, ?)", (f"PRES-{m['id']}", m['id'], 'pressure'))
            
    # Seed Initial ML Model metadata
    cursor.execute("SELECT COUNT(*) FROM ML_Model")
    if cursor.fetchone()[0] == 0:
        cursor.execute("INSERT INTO ML_Model (model_name, model_type) VALUES (?, ?)", ('RUL_Predictor', 'RandomForestRegressor'))
        cursor.execute("INSERT INTO ML_Model (model_name, model_type) VALUES (?, ?)", ('Failure_Predictor', 'RandomForestClassifier'))

    conn.commit()
    conn.close()

def reset_db():
    # Deletes or completely truncates the database to start fresh on a new simulation run
    if os.path.exists(DB_PATH):
        try:
            os.remove(DB_PATH)
            return
        except (PermissionError, OSError):
            pass
            
    # Robust fallback for Windows: if another process has an open connection, drop all tables cleanly
    if os.path.exists(DB_PATH):
        try:
            conn = sqlite3.connect(DB_PATH)
            cursor = conn.cursor()
            tables = [
                "Telemetry_Data", "Prediction_Log", "Model_Drift_Metric",
                "Maintenance_Alert", "Work_Order", "Model_Version",
                "ML_Model", "Sensor", "Industrial_Asset", "Technician"
            ]
            for t in tables:
                cursor.execute(f"DROP TABLE IF EXISTS {t};")
            conn.commit()
            conn.close()
        except Exception:
            pass
