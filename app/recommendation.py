# recommendation.py
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

def fetch_all_podcasts_by_genre():
    """Fetch all podcasts and group them by genre."""
    all_podcasts = podcast_session.query(Podcast).all()
    podcasts_by_genre = {}

    for podcast in all_podcasts:
        if podcast.genre not in podcasts_by_genre:
            podcasts_by_genre[podcast.genre] = []
        podcasts_by_genre[podcast.genre].append(podcast)

    return podcasts_by_genre


def get_podcast_suggestions(podcasts_by_genre, suggested_titles, liked_genres):
    """Suggests podcasts to the user based on liked genres or random genres."""
    available_genres = list(podcasts_by_genre.keys())
    random.shuffle(available_genres)

    possible_genres = list(liked_genres) if liked_genres else available_genres
    found_suggestions = False

    # Prioritize liked genres if available
    for genre in possible_genres:
        remaining_podcasts = [p for p in podcasts_by_genre[genre] if p.title not in suggested_titles]
        if remaining_podcasts:
            found_suggestions = True
            break

    # Fallback: pick random genres if no liked genres are available
    if not found_suggestions:
        for genre in available_genres:
            remaining_podcasts = [p for p in podcasts_by_genre[genre] if p.title not in suggested_titles]
            if remaining_podcasts:
                found_suggestions = True
                break

    # No suggestions available
    if not found_suggestions:
        return []

    # Suggest up to 3 podcasts from the selected genre
    suggestions = [p for p in podcasts_by_genre[genre] if p.title not in suggested_titles]
    return random.sample(suggestions, min(3, len(suggestions)))


def process_user_feedback(suggestions, liked_podcasts, disliked_podcasts, liked_genres, user_responses):
    """Processes the user's feedback on suggested podcasts, stopping after 3 liked shows."""
    for podcast_title, response in user_responses.items():
        if response == "yes":
            liked_podcasts.append(podcast_title)  # Store only title
        else:
            disliked_podcasts.append(podcast_title)  # Store only title

        # Stop processing if 3 liked shows are collected
        if len(liked_podcasts) >= 3:
            break


def save_user_preferences(user_id, liked_podcasts, disliked_podcasts):
    """Saves the user's liked and disliked podcasts in the database."""
    try:
        user_pref = user_pref_session.query(UserPreference).filter_by(user_id=user_id).first()

        liked_podcast_names = ", ".join(liked_podcasts)  # Ensure only titles are stored
        disliked_podcast_names = ", ".join(disliked_podcasts)

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
        print("\n✅ Preferences successfully stored in the database.")

    except Exception as e:
        print(f"⚠️ Error saving preferences: {e}")
        user_pref_session.rollback()
    finally:
        user_pref_session.close()


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


def main():
    """Handles user interaction for collecting podcast preferences."""
    user_id = int(input("Enter your user ID: "))

    # Fetch podcasts grouped by genre
    podcasts_by_genre = fetch_all_podcasts_by_genre()
    if not podcasts_by_genre:
        print("No podcasts available in the database.")
        return

    liked_podcasts, disliked_podcasts = [], []
    suggested_titles = set()
    liked_genres = set()

    while len(liked_podcasts) < 3:
        suggestions = get_podcast_suggestions(podcasts_by_genre, suggested_titles, liked_genres)
        if not suggestions:
            print("No more podcasts available to suggest.")
            break

        # Gather user responses
        user_responses = {}
        for podcast in suggestions:
            suggested_titles.add(podcast.title)
            response = input(f"Do you like '{podcast.title}'? (yes/no): ").strip().lower()
            user_responses[podcast] = response

        # Process responses
        process_user_feedback(suggestions, liked_podcasts, disliked_podcasts, liked_genres, user_responses)

    # Save user preferences
    save_user_preferences(user_id, liked_podcasts, disliked_podcasts)
    print(f"\nPreferences saved for user {user_id}.")
    # Generate Search Query from OpenAI
    search_query = generate_google_search_prompt(user_id)
    if search_query:
        # Fetch Related Podcasts Using Google Search API
        related_podcasts = search_related_podcasts(search_query, user_id)
        recommend_top_podcasts(user_id, related_podcasts)

if __name__ == "__main__":
    main()

