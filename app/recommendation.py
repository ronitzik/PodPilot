import openai
import os
import random
import requests
import numpy as np
from scipy.spatial.distance import cosine
from thefuzz import process, fuzz
from dotenv import load_dotenv
from save_embeddings import Podcast, UserPreference, user_pref_session, podcast_session

# Load env variables
load_dotenv()
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
SEARCH_ENGINE_ID = os.getenv("GOOGLE_CX")
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")
openai.api_key = OPENAI_API_KEY


def get_podcasts_by_genre(genre, exclude_titles=set(), limit=3):
    """Fetch a limited number of podcasts from a specific genre, excluding already suggested ones."""
    return podcast_session.query(Podcast).filter(
        Podcast.genre == genre,
        Podcast.title.notin_(exclude_titles)
    ).limit(limit).all()


def collect_user_preferences(user_id):
    """Dynamically suggest podcasts and adjust based on user feedback until 5 liked podcasts are collected."""
    print("\n Let's learn your podcast preferences!")

    # Get all podcasts categorized by genre
    all_podcasts = podcast_session.query(Podcast).all()
    podcasts_by_genre = {}

    for podcast in all_podcasts:
        if podcast.genre not in podcasts_by_genre:
            podcasts_by_genre[podcast.genre] = []
        podcasts_by_genre[podcast.genre].append(podcast)

    if not podcasts_by_genre:
        print("No podcasts available in the database.")
        return []

    liked_podcasts = []
    disliked_podcasts = []
    suggested_titles = set()

    # Start by suggesting random podcasts from different genres
    available_genres = list(podcasts_by_genre.keys())
    random.shuffle(available_genres)

    while len(liked_podcasts) < 3:
        # If user liked a genre, suggest more from that genre; otherwise, try another genre
        if liked_podcasts:
            preferred_genre = liked_podcasts[-1]["genre"]  # Get the genre of the last liked podcast
            suggestions = get_podcasts_by_genre(preferred_genre, exclude_titles=suggested_titles, limit=3)
        else:
            # Pick a new genre randomly
            genre = available_genres.pop(0) if available_genres else None
            if not genre:
                print(" No more unique genres available.")
                break
            suggestions = get_podcasts_by_genre(genre, exclude_titles=suggested_titles, limit=3)

        if not suggestions:
            print(f" No more suggestions available for genre: {preferred_genre if liked_podcasts else genre}.")
            continue

        for podcast in suggestions:
            suggested_titles.add(podcast.title)  # Keep track of already suggested shows
            response = input(f"Do you like '{podcast.title}'? (yes/no): ").strip().lower()

            if response == "yes":
                liked_podcasts.append(
                    {"podcast_id": podcast.podcast_id, "title": podcast.title, "genre": podcast.genre})
                if len(liked_podcasts) >=3:
                    break
            else:
                disliked_podcasts.append(
                    {"podcast_id": podcast.podcast_id, "title": podcast.title, "genre": podcast.genre})

    try:
        # Save preferences in the database
        user_pref = user_pref_session.query(UserPreference).filter_by(user_id=user_id).first()
        liked_podcast_names = ", ".join([podcast["title"] for podcast in liked_podcasts])
        disliked_podcast_names = ", ".join([podcast["title"] for podcast in disliked_podcasts])

        if user_pref:
            user_pref.liked_podcasts = liked_podcast_names
            user_pref.disliked_podcasts = disliked_podcast_names
        else:
            user_pref = UserPreference(
                user_id=user_id,
                liked_podcasts=liked_podcast_names,
                disliked_podcasts=disliked_podcast_names
            )
            user_pref_session.add(user_pref)

        user_pref_session.commit()

    except Exception as e:
        print(f" Error saving preferences: {e}")
        user_pref_session.rollback()

    finally:
        user_pref_session.close()

    print(f"\n Preferences saved for user {user_id}.")

def generate_google_search_prompt(user_id):
    """Generates a prompt for OpenAI to find related shows."""

    # Fetch user preferences
    user_pref = user_pref_session.query(UserPreference).filter_by(user_id=user_id).first()
    if not user_pref or not user_pref.liked_podcasts:
        print("No liked podcasts found for this user.")
        return None

    liked_titles = user_pref.liked_podcasts.split(", ")

    # OpenAI prompt
    prompt = f"""
    Given these podcast shows: {', '.join(liked_titles)}, 
    write a Google Search API query that will help me find similar podcast shows in ISRAEL 
    that the user will likely enjoy. Search in spotify.com and Answer in plain text ONLY the query in english.
    """

    response = openai.ChatCompletion.create(
        model="gpt-4o-mini",
        messages=[{"role": "user", "content": prompt}]
    )
    search_query = response["choices"][0]["message"]["content"]
    return search_query


def search_related_podcasts(query, user_id):
    """Find related podcasts using improved fuzzy search, but exclude already liked ones."""

    # Fetch all podcast titles from the database
    podcast_titles = [podcast.title for podcast in podcast_session.query(Podcast).all()]

    if not podcast_titles:
        print(" No podcasts found in the database.")
        return []

    # Fetch user's liked podcasts
    user_pref = user_pref_session.query(UserPreference).filter_by(user_id=user_id).first()
    liked_podcasts = set(user_pref.liked_podcasts.split(", ")) if user_pref and user_pref.liked_podcasts else set()

    # Search Google API for related podcast titles
    url = f"https://www.googleapis.com/customsearch/v1?q={query}&key={GOOGLE_API_KEY}&cx={SEARCH_ENGINE_ID}"
    response = requests.get(url)
    data = response.json()

    # Extract potential podcast names from Google search results
    related_podcasts_raw = [item.get("title") for item in data.get("items", []) if item.get("title")]

    if not related_podcasts_raw:
        print(" No related podcasts found via Google Search.")
        return []

    # Perform enhanced fuzzy matching
    related_podcasts = set()

    for raw_title in related_podcasts_raw:
        # Get top 3 best matches using a flexible scorer
        best_matches = process.extract(raw_title, podcast_titles, scorer=fuzz.partial_ratio, limit=3)

        for match, score in best_matches:
            # Ensure the match is not identical to a liked podcast
            if score >= 80 and match not in liked_podcasts:
                related_podcasts.add(match)

    related_podcasts = list(related_podcasts)
    return related_podcasts

def get_podcast_embedding(title):
    """Fetch podcast embedding from the database given the title."""
    podcast = podcast_session.query(Podcast).filter_by(title=title).first()
    if podcast and podcast.embedding:
        return np.frombuffer(podcast.embedding, dtype=np.float32)  # Convert binary to NumPy array
    return None


def recommend_top_podcasts(user_id, related_podcasts):
    """Recommend top 3 new podcasts based on embedding similarity."""

    # Fetch user's liked podcasts
    user_pref = user_pref_session.query(UserPreference).filter_by(user_id=user_id).first()
    liked_podcasts = set(user_pref.liked_podcasts.split(", ")) if user_pref and user_pref.liked_podcasts else set()

    if not liked_podcasts:
        print("No liked podcasts found for this user.")
        return []

    # Get embeddings for liked podcasts
    liked_embeddings = [get_podcast_embedding(title) for title in liked_podcasts if
                        get_podcast_embedding(title) is not None]

    if not liked_embeddings:
        print(" No valid embeddings found for liked podcasts.")
        return []

    # Get embeddings for related podcasts (that the user has not already liked)
    related_embeddings = []
    related_podcast_titles = []

    for podcast in related_podcasts:
        if podcast not in liked_podcasts:
            embedding = get_podcast_embedding(podcast)
            if embedding is not None:
                related_embeddings.append(embedding)
                related_podcast_titles.append(podcast)

    if not related_embeddings:
        print("No valid embeddings found for related podcasts.")
        return []

    # Compute average embedding for liked podcasts
    avg_liked_embedding = np.mean(liked_embeddings, axis=0)

    # Compute similarity scores
    similarity_scores = []
    for i, embedding in enumerate(related_embeddings):
        similarity = 1 - cosine(avg_liked_embedding, embedding)  # Cosine similarity
        similarity_scores.append((related_podcast_titles[i], similarity))

    # Sort by similarity (descending)
    similarity_scores.sort(key=lambda x: x[1], reverse=True)

    # Get the top 3 recommendations
    top_recommendations = [podcast for podcast, score in similarity_scores[:3]]

    print(f"Based on the information I learned I think you should try listening to: {top_recommendations}")
    return top_recommendations

if __name__ == "__main__":
    user_id = int(input("Enter your user ID: "))

    # Collect User Preferences
    collect_user_preferences(user_id)

    # Generate Search Query from OpenAI
    search_query = generate_google_search_prompt(user_id)

    if search_query:
        # Fetch Related Podcasts Using Google Search API
        related_podcasts = search_related_podcasts(search_query, user_id)
        if related_podcasts:
            recommend_top_podcasts(user_id, related_podcasts)
