from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
from recommendation import recommend_top_podcasts, collect_user_preferences, generate_google_search_prompt, search_related_podcasts

router = APIRouter()

# Request model for podcast recommendations
class RecommendationRequest(BaseModel):
    user_id: int

@router.post("/recommend_podcasts")
def recommend_podcasts(request: RecommendationRequest):
    """Finds the top 3 podcasts based on user preferences and embeddings."""
    try:
        # Collect user preferences
        collect_user_preferences(request.user_id)
        # Generate a Google search query for related podcasts
        search_query = generate_google_search_prompt(request.user_id)
        # Search for related podcasts
        related_podcasts = search_related_podcasts(search_query, request.user_id) if search_query else []
        # Recommend top podcasts based on embeddings
        recommendations = recommend_top_podcasts(request.user_id, related_podcasts)
        if not recommendations:
            raise HTTPException(status_code=404, detail="No relevant podcast recommendations found.")

        return {"recommended_podcasts": recommendations}

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error generating recommendations: {str(e)}")