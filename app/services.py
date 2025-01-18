from fastapi import APIRouter
from pydantic import BaseModel
from recommendation import generate_podcast_recommendations

router = APIRouter()

class RecommendationRequest(BaseModel):
    user_personalities: list[str]
    available_time_min: int

@router.post("/recommend_podcasts")
def recommend_podcasts(request: RecommendationRequest):
    """Finds the top 5 podcasts based on user's favorite personalities and available time."""
    recommendations = generate_podcast_recommendations(request.user_personalities, request.available_time_min)
    return recommendations
