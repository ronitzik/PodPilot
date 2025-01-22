from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
from recommendation import generate_podcast_recommendations
from summarization import generate_episode_summary
from person_search import search_episodes_by_person
from guest_info import find_guest_info
from voice_summary import generate_voice_summary
from article_podcast import generate_podcast_from_article

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

# Request Model for Guest Information Retrieval
class GuestInfoRequest(BaseModel):
    podcast_name: str
    episode_name: str

@router.post("/find_guest_info")
def get_guest_info(request: GuestInfoRequest):
    """Finds guest information (name & LinkedIn) for a given podcast episode."""
    if not request.podcast_name or not request.episode_name:
        raise HTTPException(status_code=400, detail="Podcast name and episode name are required.")

    guest_info = find_guest_info({"podcast_name": request.podcast_name, "episode_name": request.episode_name})

    if "error" in guest_info:
        raise HTTPException(status_code=404, detail=guest_info["error"])

    return guest_info

# New Feature: Voice Summary of an Episode Segment
class VoiceSummaryRequest(BaseModel):
    podcast_name: str
    episode_name: str
    start_time: str  # Expected format: "HH:MM:SS"
    end_time: str  # Expected format: "HH:MM:SS"

@router.post("/voice_summary")
def voice_summary(request: VoiceSummaryRequest):
    """Generates a voice summary for a specific part of a podcast episode."""
    if not request.podcast_name or not request.episode_name or not request.start_time or not request.end_time:
        raise HTTPException(status_code=400, detail="Podcast name, episode name, start time, and end time are required.")

    summary_audio_url = generate_voice_summary(
        podcast_name=request.podcast_name,
        episode_name=request.episode_name,
        start_time=request.start_time,
        end_time=request.end_time
    )

    if "error" in summary_audio_url:
        raise HTTPException(status_code=404, detail=summary_audio_url["error"])

    return {"summary_audio_url": summary_audio_url}

class ArticlePodcastRequest(BaseModel):
    article_url: str

@router.post("/generate_podcast_from_article")
def generate_podcast(request: ArticlePodcastRequest):
    """Generates a podcast episode based on an article URL."""
    if not request.article_url:
        raise HTTPException(status_code=400, detail="Article URL is required.")

    podcast_audio_url = generate_podcast_from_article(
        article_url=request.article_url
    )

    if "error" in podcast_audio_url:
        raise HTTPException(status_code=500, detail=podcast_audio_url["error"])

    return {"podcast_audio_url": podcast_audio_url}