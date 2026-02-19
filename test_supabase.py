from supabase import create_client
from datetime import datetime
from dotenv import load_dotenv
import os

# Load environment variables
load_dotenv()

# Read from your .env file
SUPABASE_URL = os.getenv("SUPABASE_URL")
SUPABASE_KEY = os.getenv("SUPABASE_KEY")

# Connect to Supabase
supabase = create_client(SUPABASE_URL, SUPABASE_KEY)

# Data to insert
data = {
    "type": "Test Alert",
    "message": "Inserted from Python test script",
    "created_at": datetime.utcnow().isoformat()
}

# Insert into table
response = supabase.table("driver_alerts").insert(data).execute()

# Print results
print("Insert response:", response)
