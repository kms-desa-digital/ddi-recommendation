import os
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from app.routes import recommendation_routes

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
