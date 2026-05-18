from pymongo import MongoClient
from .settings import settings
import sys

def initialize_mongodb():
    try:
        # Check if settings has MONGODB_URL
        url = settings.MONGODB_URL
        db_name = settings.MONGODB_DB
        print(f"Connecting to MongoDB at: {url}")
        
        # Connect to client
        client = MongoClient(url)
        # Test connection
        client.admin.command('ping')
        print(f"✅ Successfully connected to MongoDB database: {db_name}")
        return client[db_name]
    except Exception as e:
        print(f"❌ Error connecting to MongoDB: {e}")
        # Try local fallback
        try:
            print("Falling back to local default MongoDB connection...")
            client = MongoClient("mongodb://localhost:27017")
            return client["db_desa"]
        except Exception as e2:
            print(f"❌ Fallback MongoDB connection failed: {e2}")
            return None

db = initialize_mongodb()
