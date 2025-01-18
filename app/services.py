from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
from recommendation import generate_podcast_recommendations
from summarization import generate_episode_summary
from person_search import search_episodes_by_person

router = APIRouter()

class RecommendationRequest(BaseModel):
    user_personalities: list[str]
    available_time_min: int

@router.post("/recommend_podcasts")
def recommend_podcasts(request: RecommendationRequest):
    """Finds the top 3 podcasts based on user's favorite personalities and available time."""
    recommendations = generate_podcast_recommendations(request.user_personalities, request.available_time_min)
    return recommendations

class EpisodeRequest(BaseModel):
    podcast_name: str
    episode_name: str

@router.post("/summarize_episode")
def summarize_episode(request: EpisodeRequest):
    """Finds the episode and returns a one-page summary."""
    return generate_episode_summary(request.podcast_name, request.episode_name)

class PersonSearchRequest(BaseModel):
    person_name: str  # Expect JSON input instead of query param

@router.post("/search_by_person")
def search_person(request: PersonSearchRequest):
    """Search for podcast episodes where a specific person is mentioned."""
    if not request.person_name:
        raise HTTPException(status_code=400, detail="Person name is required.")

    results = search_episodes_by_person(request.person_name)

    if "error" in results:
        raise HTTPException(status_code=404, detail=results["error"])

    return {"search_results": results}