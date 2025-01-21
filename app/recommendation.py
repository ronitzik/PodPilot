import pickle
import numpy as np
import openai
from annoy import AnnoyIndex
from database import get_db_connection
from dotenv import load_dotenv
import os

load_dotenv()

# Initialize OpenAI API
openai.api_key = os.getenv("OPENAI_API_KEY")

# Define official genre categories
OFFICIAL_GENRES = [
    "Arts", "Business", "Comedy", "Education", "Health & Fitness", "History",
    "Kids & Family", "Leisure", "Music", "News", "Religion & Spirituality",
    "Science", "Society & Culture", "Sports", "Technology", "True Crime", "TV & Film"
]

def classify_personality_genre(personalities):
    """Uses OpenAI to determine the most relevant genre for given personalities."""
    prompt = f"""
    Given the following famous personalities: {', '.join(personalities)},
    determine which of these podcast genre they are MOSTLY associated with: {', '.join(OFFICIAL_GENRES)}.
    Respond with only the one most relevant genre.
    """

    try:
        response = openai.ChatCompletion.create(
            model="gpt-4o-mini",
            messages=[{"role": "user", "content": prompt}]
        )
        predicted_genres = response["choices"][0]["message"]["content"].strip()
        matched_genres = [genre for genre in OFFICIAL_GENRES if genre in predicted_genres]

        if not matched_genres:
            print(f"No valid genres found, defaulting to 'Unknown'")
            return ["Unknown"]

        return matched_genres

    except Exception as e:
        print(f"Error classifying personalities' genres with OpenAI: {e}")
        return ["Unknown"]

def load_podcast_data():
    """Loads podcast names, embeddings, durations, and genres from the database."""
    conn = get_db_connection()
    cursor = conn.cursor()

    cursor.execute("SELECT title, embedding, duration_ms, genre FROM podcasts WHERE embedding IS NOT NULL")
    rows = cursor.fetchall()

    podcast_dict = {}
    podcast_durations = {}
    podcast_genres = {}

    for row in rows:
        title = row["title"].lower()
        embedding = pickle.loads(row["embedding"])
        duration = row["duration_ms"]
        genre = row["genre"]

        podcast_dict[title] = embedding
        podcast_durations[title] = duration
        podcast_genres[title] = genre

    conn.close()
    return podcast_dict, podcast_durations, podcast_genres

def generate_podcast_recommendations(user_personalities, available_time_min, num_trees=10):
    """Recommends podcasts based on personalities' associated genres, highest similarity, and closest duration match."""

    # Get genres based on user-selected personalities
    personality_genres = classify_personality_genre(user_personalities)
    podcast_dict, podcast_durations, podcast_genres = load_podcast_data()

    if not podcast_dict:
        return {"error": "No podcasts found with embeddings"}

    # Convert available time from minutes to milliseconds
    target_duration_ms = available_time_min * 60 * 1000

    # Build an Annoy index
    dim = len(next(iter(podcast_dict.values())))
    annoy_index = AnnoyIndex(dim, "angular")

    podcast_titles = list(podcast_dict.keys())

    for i, (title, embedding) in enumerate(podcast_dict.items()):
        annoy_index.add_item(i, embedding)

    annoy_index.build(num_trees)

    # Filter podcasts by matching genres
    filtered_podcasts = [
        title for title, genre in podcast_genres.items() if any(g in genre for g in personality_genres)
    ]

    if not filtered_podcasts:
        return {"error": "No podcasts found for the given personalities' genres"}

    # Get embeddings for the filtered genre podcasts
    filtered_embeddings = [
        (title, podcast_dict[title]) for title in filtered_podcasts if title in podcast_dict
    ]

    if not filtered_embeddings:
        return {"error": "No embeddings found for filtered genre podcasts"}

    # Compute the average embedding for the filtered genre podcasts
    filtered_titles, filtered_vectors = zip(*filtered_embeddings)
    average_vector = np.mean(filtered_vectors, axis=0)

    # Find the nearest neighbors using Annoy
    nearest_neighbors = annoy_index.get_nns_by_vector(average_vector, len(filtered_titles), include_distances=True)

    # Get the recommended podcasts with similarity scores
    recommendations = []
    for idx, dist in zip(nearest_neighbors[0], nearest_neighbors[1]):
        show_name = podcast_titles[idx]
        if show_name not in filtered_podcasts:
            continue
        duration = podcast_durations.get(show_name, None)

        if duration is not None:
            similarity_score = 1 - dist
            similarity_score = max(0, min(similarity_score, 1))
            similarity_percentage = round(similarity_score * 100, 3)

            recommendations.append((show_name, similarity_percentage, duration))

    # Sort recommendations by similarity score (highest first)
    recommendations.sort(key=lambda x: x[1], reverse=True)

    # From the sorted list, find the top 3 recommendations closest to the user's available time
    recommendations.sort(key=lambda x: abs(x[2] - target_duration_ms))

    return {
        "recommended_podcasts": [
            {"title": rec[0], "similarity": rec[1], "duration_ms": rec[2]}
            for rec in recommendations[:3]
        ]
    }
