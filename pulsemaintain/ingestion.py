from .db import get_connection                             # SQLite database connection helper
from .config import MACHINES, MA_WINDOW_SIZE, DRIFT_HOUR      # Machine list and filter settings

class DataIngestion:
    def __init__(self, simulator):
        self.simulator = simulator                             # Sensor reading generator instance
        # Rolling buffer storing last N readings per machine to compute Moving Average
        self.history = {m["id"]: {"vib": [], "temp": [], "pres": []} for m in MACHINES}

    def process_hour(self, hour):
        conn = get_connection()
        cursor = conn.cursor()
        
        # Log drift injection event when hour 200 is reached
        if hour == DRIFT_HOUR:
            print("[EVENT] Simulated hour 200 reached. Introducing calibration drift.")
            print("[EVENT] Drift details: Vibration +1.6, Temperature +14, Pressure -9")
            
        data_records = []
        for m in MACHINES:
            asset_id = m["id"]
            # Generate simulated physics reading for the current hour
            readings = self.simulator.generate_readings(asset_id, hour)
            
            # Append raw reading into history buffer for moving average computation
            self.history[asset_id]["vib"].append(readings["vibration"])
            self.history[asset_id]["temp"].append(readings["temperature"])
            self.history[asset_id]["pres"].append(readings["pressure"])
            
            # Maintain sliding window of size MA_WINDOW_SIZE (5): drop oldest reading if > 5
            if len(self.history[asset_id]["vib"]) > MA_WINDOW_SIZE:
                self.history[asset_id]["vib"].pop(0)
                self.history[asset_id]["temp"].pop(0)
                self.history[asset_id]["pres"].pop(0)
                
            # Compute Moving Average (MA) filtered values: sum of window divided by count
            vib_f = sum(self.history[asset_id]["vib"]) / len(self.history[asset_id]["vib"])
            temp_f = sum(self.history[asset_id]["temp"]) / len(self.history[asset_id]["temp"])
            pres_f = sum(self.history[asset_id]["pres"]) / len(self.history[asset_id]["pres"])
            
            # Update asset's current health score and operating hours in database
            cursor.execute('''
                UPDATE Industrial_Asset
                SET health_score = ?, operating_hours = ?
                WHERE asset_id = ?
            ''', (readings["health"], hour, asset_id))
            
            # Save both raw and filtered telemetry into Telemetry_Data table
            cursor.execute('''
                INSERT INTO Telemetry_Data (
                    asset_id, hour, vibration_raw, temperature_raw, pressure_raw,
                    vibration_filtered, temperature_filtered, pressure_filtered
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            ''', (
                asset_id, hour, 
                readings["vibration"], readings["temperature"], readings["pressure"],
                round(vib_f, 4), round(temp_f, 4), round(pres_f, 4)
            ))
            
            # Package filtered telemetry record for pipeline inference
            data_records.append({
                "asset_id": asset_id,
                "hour": hour,
                "vibration_filtered": vib_f,
                "temperature_filtered": temp_f,
                "pressure_filtered": pres_f,
                "real_rul": readings["real_rul"],
                "failure_48h": readings["failure_48h"]
            })
            
        conn.commit()  # Commit all 8 machine updates in one transaction
        conn.close()
        return data_records
