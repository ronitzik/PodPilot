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
SEARCH_ENGINE_ID = os.getenv("GOOGLE_CX")
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")
openai.api_key = os.getenv("OPENAI_API_KEY")


def get_podcasts_by_genre(genre, exclude_titles=set(), limit=3):
    """Fetch a limited number of podcasts from a specific genre, excluding already suggested ones."""
    return podcast_session.query(Podcast).filter(
        Podcast.genre == genre,
        Podcast.title.notin_(exclude_titles)
    ).limit(limit).all()


def collect_user_preferences(user_id):
    """Dynamically suggest podcasts and adjust based on user feedback until 3 liked podcasts are collected."""
    print("\nLet's learn your podcast preferences!")

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
    liked_genres = set()

    # Start by suggesting random genres
    available_genres = list(podcasts_by_genre.keys())
    random.shuffle(available_genres)

    while len(liked_podcasts) < 3:
        # Prioritize suggesting from genres the user already liked
        if liked_genres:
            possible_genres = list(liked_genres)  # Use only liked genres
        else:
            possible_genres = available_genres  # Use all available genres if none are liked yet

        found_suggestions = False

        for genre in possible_genres:
            remaining_podcasts = [p for p in podcasts_by_genre[genre] if p.title not in suggested_titles]
            if remaining_podcasts:
                found_suggestions = True
                break

        # If no liked genres have available podcasts, pick a random genre
        if not found_suggestions:
            for genre in available_genres:
                remaining_podcasts = [p for p in podcasts_by_genre[genre] if p.title not in suggested_titles]
                if remaining_podcasts:
                    found_suggestions = True
                    break

        # If no genres have available podcasts, stop suggesting
        if not found_suggestions:
            print("No more podcasts available to suggest.")
            break

        # Suggest up to 3 podcasts from the selected genre
        suggestions = [p for p in podcasts_by_genre[genre] if p.title not in suggested_titles]
        for podcast in random.sample(suggestions, min(3, len(suggestions))):
            suggested_titles.add(podcast.title)
            response = input(f"Do you like '{podcast.title}'? (yes/no): ").strip().lower()

            if response == "yes":
                liked_podcasts.append(
                    {"podcast_id": podcast.podcast_id, "title": podcast.title, "genre": podcast.genre}
                )
                liked_genres.add(podcast.genre)
                if len(liked_podcasts) >= 3:
                    break
            else:
                disliked_podcasts.append(
                    {"podcast_id": podcast.podcast_id, "title": podcast.title, "genre": podcast.genre}
                )

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

    print(f"\nPreferences saved for user {user_id}.")


def generate_google_search_prompt(user_id):
    """Generates a prompt for OpenAI to find related shows."""

    # Fetch user preferences
    user_pref = user_pref_session.query(UserPreference).filter_by(user_id=user_id).first()
    if not user_pref or not user_pref.liked_podcasts:
        print("No liked podcasts found for this user.")
        return None

    liked_titles = user_pref.liked_podcasts.split(", ")

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
        return []

    # Fetch user's liked podcasts
    user_pref = user_pref_session.query(UserPreference).filter_by(user_id=user_id).first()
    liked_podcasts = set(user_pref.liked_podcasts.split(", ")) if user_pref and user_pref.liked_podcasts else set()

    # Construct Google Search API URL
    url = f"https://www.googleapis.com/customsearch/v1?q={query}&key={GOOGLE_API_KEY}&cx={SEARCH_ENGINE_ID}"
    try:
        response = requests.get(url)
        data = response.json()
        # Extract podcast names from search results
        related_podcasts_raw = set()

        for item in data.get("items", []):
            title = item.get("title", "")

            # Fallback to 'og:title' if available
            meta_tags = item.get("pagemap", {}).get("metatags", [])
            og_title = meta_tags[0].get("og:title", "") if meta_tags else ""

            # Clean the title by removing unwanted suffixes
            clean_title = title or og_title
            clean_title = clean_title.replace(" | Podcast on Spotify", "").strip()

            # Ensure it's a valid podcast title
            if clean_title and clean_title not in liked_podcasts:
                related_podcasts_raw.add(clean_title)
        if not related_podcasts_raw:
            return []

        # Perform enhanced fuzzy matching
        related_podcasts = set()

        for raw_title in related_podcasts_raw:
            best_matches = process.extract(raw_title, podcast_titles, scorer=fuzz.partial_ratio, limit=3)

            for match, score in best_matches:
                if score >= 80 and match not in liked_podcasts:
                    related_podcasts.add(match)

        related_podcasts = list(related_podcasts)
        return related_podcasts if related_podcasts else []

    except Exception as e:
        print(f"Error fetching related podcasts: {e}")
        return []

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
        print("No valid embeddings found for liked podcasts.")
        return []

    if related_podcasts:
        print("Found related podcasts via Google Search, Now I have more data for the recommendation.")
    else:
        print("No related podcasts found via Google Search, recommending only based on liked podcasts.")

    # Get embeddings for related podcasts (that the user has not already liked)
    related_embeddings = []
    related_podcast_titles = []

    if related_podcasts:
        for podcast in related_podcasts:
            if podcast not in liked_podcasts:
                embedding = get_podcast_embedding(podcast)
                if embedding is not None:
                    related_embeddings.append(embedding)
                    related_podcast_titles.append(podcast)

    # Find similar podcasts from the full database
    all_podcasts = podcast_session.query(Podcast.title).all()
    for podcast in all_podcasts:
        title = podcast[0]  # Extract title from tuple
        if title not in liked_podcasts:  # Ensure it's not a liked podcast
            embedding = get_podcast_embedding(title)
            if embedding is not None:
                related_embeddings.append(embedding)
                related_podcast_titles.append(title)

    if not related_embeddings:
        print("No valid embeddings found for recommendations.")
        return []
    # Compute average embedding for liked podcasts
    avg_liked_embedding = np.mean(liked_embeddings, axis=0)

    # Compute similarity scores
    similarity_scores = []
    for i, embedding in enumerate(related_embeddings):
        similarity = 1 - cosine(avg_liked_embedding, embedding)  # Cosine similarity
        similarity_scores.append((related_podcast_titles[i], similarity))

    # Sort by similarity
    similarity_scores.sort(key=lambda x: x[1], reverse=True)
    # Get the top 3 recommendations
    top_recommendations = [podcast for podcast, score in similarity_scores[:3]]

    # Fetch descriptions for recommended podcasts
    recommended_podcasts_info = podcast_session.query(Podcast).filter(Podcast.title.in_(top_recommendations)).all()

    print("\nBased on the information I learned, I think you should try listening to:")
    for podcast in recommended_podcasts_info:
        formatted_description = format_podcast_description(podcast.title, podcast.description)
        print(formatted_description + "\n")

    return top_recommendations

def format_podcast_description(name, description):
    """Use OpenAI to format podcast recommendations."""
    prompt = f"""
    Format this podcast description:
    The podcast name: {name}, the description: {description} (Use this info to write what is this podcast about) 
    
    Answer in this format:
    "The podcast name: [Podcast Name, if the name is in Hebrew leave it that way]. Description: The podcast is about...[Use only English]"
    """

    response = openai.ChatCompletion.create(
        model="gpt-4o-mini",
        messages=[{"role": "user", "content": prompt}]
    )

    return response["choices"][0]["message"]["content"].strip()


if __name__ == "__main__":
    user_id = int(input("Enter your user ID: "))

    # Collect User Preferences
    collect_user_preferences(user_id)

    # Generate Search Query from OpenAI
    search_query = generate_google_search_prompt(user_id)
    if search_query:
        # Fetch Related Podcasts Using Google Search API
        related_podcasts = search_related_podcasts(search_query, user_id)
        recommend_top_podcasts(user_id, related_podcasts)

