import uuid                                           # Generate unique alphanumeric Work Order identifiers (e.g. WO-A1B2C3)
from .db import get_connection                             # SQLite database connection helper
from .config import PROBABILITY_THRESHOLD, RUL_THRESHOLD  # 0.80 failure probability and 150 hours RUL limits

class AlertManager:
    def evaluate_risk(self, asset_id, hour, predicted_rul, failure_prob):
        # Trigger condition: Failure probability > 80% OR predicted RUL < 150 hours
        if failure_prob > PROBABILITY_THRESHOLD or (0 < predicted_rul < RUL_THRESHOLD):
            self.create_alert(asset_id, hour, predicted_rul, failure_prob)
            
    def create_alert(self, asset_id, hour, predicted_rul, failure_prob):
        conn = get_connection()
        cursor = conn.cursor()
        
        # Debouncing rule: Check if an OPEN alert already exists for this machine to prevent duplicates
        cursor.execute("SELECT alert_id FROM Maintenance_Alert WHERE asset_id = ? AND status = 'OPEN'", (asset_id,))
        if cursor.fetchone():
            conn.close()
            return # Skip: active open alert already exists
            
        reason = f"High Risk: Failure Prob={failure_prob:.2f}, Pred RUL={predicted_rul:.1f}h"
        
        # Insert new CRITICAL severity alert into Maintenance_Alert table
        cursor.execute('''
            INSERT INTO Maintenance_Alert (asset_id, hour, reason, severity, status)
            VALUES (?, ?, ?, 'CRITICAL', 'OPEN')
        ''', (asset_id, hour, reason))
        alert_id = cursor.lastrowid  # Retrieve auto-incremented alert ID
        
        print(f"[ALERT] CRITICAL Alert created for {asset_id} at hour {hour}.")
        
        # Load balancing algorithm: Find technician currently having the fewest OPEN work orders
        cursor.execute('''
            SELECT t.technician_id, COUNT(w.work_order_id) as open_count
            FROM Technician t
            LEFT JOIN Work_Order w ON t.technician_id = w.technician_id AND w.status = 'OPEN'
            GROUP BY t.technician_id
            ORDER BY open_count ASC, t.technician_id ASC
            LIMIT 1
        ''')
        tech_row = cursor.fetchone()
        if tech_row:
            tech_id = tech_row[0]                                     # Selected least-loaded technician ID
            work_order_id = f"WO-{uuid.uuid4().hex[:6].upper()}"      # Unique work order code
            
            # Automatically assign work order linked to the asset, alert, and technician
            cursor.execute('''
                INSERT INTO Work_Order (work_order_id, asset_id, alert_id, technician_id, creation_hour, status)
                VALUES (?, ?, ?, ?, ?, 'OPEN')
            ''', (work_order_id, asset_id, alert_id, tech_id, hour))
            
            print(f"[MAINT] Work Order {work_order_id} assigned to {tech_id}")
            
        conn.commit()
        conn.close()
