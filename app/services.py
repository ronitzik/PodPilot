#services.py
from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
from typing import Dict
from save_embeddings import Podcast, UserPreference, user_pref_session, podcast_session
from recommendation import (
    fetch_all_podcasts_by_genre,
    get_podcast_suggestions,
    process_user_feedback,
    save_user_preferences,
    generate_google_search_prompt,
    search_related_podcasts,
    recommend_top_podcasts,
format_podcast_description
)
router = APIRouter()

class UserResponse(BaseModel):
    user_id: int
    responses: Dict[str, str]  # Podcast title -> "yes" or "no"

class UserPreferences(BaseModel):
    user_id: int

class SearchRequest(BaseModel):
    user_id: int
    search_query: str
@router.get("/fetch_podcasts_by_genre")
def get_podcasts_by_genre():
    """Fetch all podcasts grouped by genre."""
    try:
        podcasts_by_genre = fetch_all_podcasts_by_genre()
        return {"podcasts_by_genre": podcasts_by_genre}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@router.post("/suggest_podcasts")
def suggest_podcasts(user_prefs: UserPreferences):
    """Suggest podcasts based on user preferences."""
    try:
        podcasts_by_genre = fetch_all_podcasts_by_genre()
        suggested_titles = set()
        liked_genres = set()
        suggestions = get_podcast_suggestions(podcasts_by_genre, suggested_titles, liked_genres)
        return {"suggestions": [p.title for p in suggestions]}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@router.post("/process_feedback")
def process_feedback(feedback: UserResponse):
    """Process user feedback on suggested podcasts."""
    try:
        liked_podcasts = []
        disliked_podcasts = []
        liked_genres = set()
        podcasts_by_genre = fetch_all_podcasts_by_genre()
        suggested_titles = list(feedback.responses.keys())
        suggestions = [podcast_session.query(Podcast).filter_by(title=title).first() for title in suggested_titles]

        process_user_feedback(suggestions, liked_podcasts, disliked_podcasts, liked_genres, feedback.responses)
        save_user_preferences(feedback.user_id, liked_podcasts, disliked_podcasts)
        return {"message": "Feedback processed successfully"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@router.post("/generate_search_query")
def generate_search(user_prefs: UserPreferences):
    """Generate a Google search query based on user preferences."""
    try:
        search_query = generate_google_search_prompt(user_prefs.user_id)
        if not search_query:
            raise HTTPException(status_code=404, detail="No liked podcasts found for this user.")
        return {"search_query": search_query}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@router.post("/search_related_podcasts")
def search_podcasts(request: SearchRequest):
    """Search related podcasts using Google Search API."""
    try:
        if not request.search_query:
            raise HTTPException(status_code=400, detail="Search query is required.")

        related_podcasts = search_related_podcasts(request.search_query, request.user_id)
        return {"related_podcasts": related_podcasts}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@router.post("/recommend_podcasts")
def recommend_podcasts(user_prefs: UserPreferences):
    """Recommend top 3 podcasts based on user preferences."""
    try:
        search_query = generate_google_search_prompt(user_prefs.user_id)
        related_podcasts = search_related_podcasts(search_query, user_prefs.user_id) if search_query else []
        recommendations = recommend_top_podcasts(user_prefs.user_id, related_podcasts)

        # Fetch podcast descriptions
        recommended_podcasts_info = podcast_session.query(Podcast).filter(Podcast.title.in_(recommendations)).all()

        # Format the response with name and formatted description
        formatted_recommendations = [
            {
                "name": podcast.title,
                "formatted_description": format_podcast_description(podcast.title, podcast.description)
            }
            for podcast in recommended_podcasts_info
        ]

        return {"recommendations": formatted_recommendations}

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))