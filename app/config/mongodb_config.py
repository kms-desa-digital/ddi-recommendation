from pymongo import MongoClient
from .settings import settings

def initialize_mongodb():
    try:
        url = settings.MONGODB_URL
        db_name = settings.MONGODB_DB
        print(f"Initializing lazy MongoDB client connection to: {url}")
        
        # Connect to client (lazy, non-blocking on instantiation)
        client = MongoClient(url, serverSelectionTimeoutMS=5000)
        return client[db_name]
    except Exception as e:
        print(f"❌ Error setting up MongoDB client: {e}")
        # Return lazy fallback client
        client = MongoClient("mongodb://localhost:27017", serverSelectionTimeoutMS=2000)
        return client["db_desa"]

db = initialize_mongodb()
