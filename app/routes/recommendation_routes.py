from fastapi import APIRouter, HTTPException
from app.models.recommendation import (
    RecommendationRequest,
    RecommendationListResponse,
)
from app.services.recommendation_service import recommendation_engine

router = APIRouter()


@router.post("/recommendations", response_model=RecommendationListResponse)
async def get_recommendations(request: RecommendationRequest):
    recommendations = recommendation_engine.get_recommendations(
        innovation_id=request.innovation_id,
        top_n=request.top_n
    )

    if not recommendations:
        raise HTTPException(status_code=404, detail="Inovasi tidak ditemukan")

    return {"message": "Rekomendasi berhasil diambil", "data": recommendations}
