import pandas as pd
from fastapi import FastAPI, Request
import uvicorn
from datetime import datetime
import json

# FastAPI app
app = FastAPI()

# Output CSV file for raw data
output_csv = "speedway_connect_data.csv"

# Initialize output CSV
def init_output_csv():
    pd.DataFrame(columns=["timestamp", "raw_json"]).to_csv(output_csv, index=False)

# FastAPI endpoint to receive and log raw Speedway Connect data
@app.post("/rfid_data")
async def receive_rfid_data(request: Request):
    try:
        # Get raw JSON data
        data = await request.json()
        # Get current timestamp
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        
        # Convert data to JSON string for storage
        raw_json = json.dumps(data, ensure_ascii=False)
        
        # Print received data for inspection
        print(f"Received at {timestamp}: {raw_json}")
        
        # Save to CSV
        pd.DataFrame([{"timestamp": timestamp, "raw_json": raw_json}]).to_csv(
            output_csv, mode="a", header=False, index=False
        )
        
        return {"status": "received"}
    except Exception as e:
        print(f"Error processing request: {e}")
        return {"status": "error", "message": str(e)}

# Main function to start the server
if __name__ == "__main__":
    init_output_csv()
    uvicorn.run(app, host="0.0.0.0", port=8000)