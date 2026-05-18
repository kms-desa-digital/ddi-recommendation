import os
import sys

# Bulletproof path setup for both local execution and Vercel serverless deployments
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
for d in [current_dir, parent_dir]:
    if d not in sys.path:
        sys.path.insert(0, d)

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

try:
    from app.routes import recommendation_routes
except ImportError:
    from routes import recommendation_routes

app = FastAPI(
    title="Innovation Recommendation System",
    description="Sistem rekomendasi inovasi berbasis content-based filtering",
)

# Setup CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# Include router
app.include_router(recommendation_routes.router, prefix="/api/v1")


# Health check endpoint
@app.get("/")
async def root():
    return {"message": "Innovation Recommendation System is running"}


@app.get("/health")
async def health_check():
    return {"status": "healthy"}


if __name__ == "__main__":
    import uvicorn

    # Get port from environment variable or use default
    port = int(os.environ.get("PORT", 8000))

    uvicorn.run(app, host="0.0.0.0", port=port)
