import os
import pickle
import numpy as np
import openai
import spotipy
from spotipy.oauth2 import SpotifyClientCredentials
from dotenv import load_dotenv
from database import get_db_connection

load_dotenv()

# Initialize OpenAI API
openai.api_key = os.getenv("OPENAI_API_KEY")

# Initialize Spotify API
sp = spotipy.Spotify(auth_manager=SpotifyClientCredentials(
    client_id=os.getenv("SPOTIFY_CLIENT_ID"),
    client_secret=os.getenv("SPOTIFY_CLIENT_SECRET")
))

# Define official genre categories
OFFICIAL_GENRES = [
    "Arts", "Business", "Comedy", "Education", "Health & Fitness", "History",
    "Kids & Family", "Leisure", "Music", "News", "Religion & Spirituality",
    "Science", "Society & Culture", "Sports", "Technology", "True Crime", "TV & Film"
]

def classify_genre(title: str, description: str):
    """Sends the podcast name and description to ChatGPT to classify its genre."""
    prompt = f"""
    Given the following podcast title and description, classify it into one of these genres: {', '.join(OFFICIAL_GENRES)}.
    Title: {title}
    Description: {description}
    Respond with only the genre name.
    """

    try:
        response = openai.ChatCompletion.create(
            model="gpt-4o-mini",
            messages=[{"role": "user", "content": prompt}]
        )
        predicted_genre = response["choices"][0]["message"]["content"].strip()

        if predicted_genre not in OFFICIAL_GENRES:
            print(f"Warning: '{predicted_genre}' is not a valid genre. Setting as 'Unknown'.")
            return "Unknown"

        return predicted_genre

    except Exception as e:
        print(f"⚠️ Error classifying genre with ChatGPT: {e}")
        return "Unknown"

def generate_embedding(text: str):
    """Generates an embedding for the given text using OpenAI API."""
    response = openai.Embedding.create(
        model="text-embedding-ada-002",
        input=text
    )
    return np.array(response["data"][0]["embedding"], dtype=np.float32)

def fetch_podcasts(country="IL", total=150):
    """Fetches the top 150 podcasts from Spotify for a given country and retrieves episode duration."""
    print(f"Fetching the top {total} podcast shows from Spotify in {country}...")
    podcasts = []
    limit = 50
    num_requests = total // limit

    for i in range(num_requests):
        offset = i * limit

        results = sp.search(q="podcasts", type="show", market=country, limit=limit, offset=offset)

        for show in results["shows"]["items"]:
            podcast_id = show["id"]
            title = show["name"]
            description = show.get("description", "No description available")

            # Fetch episode details to get duration
            try:
                episodes = sp.show_episodes(podcast_id, market=country, limit=1)["items"]
                if episodes:
                    first_episode = episodes[0]
                    duration_ms = first_episode.get("duration_ms", None)
                else:
                    duration_ms = None
            except Exception as e:
                print(f"Warning: Failed to fetch episode details for {title}: {e}")
                duration_ms = None

            podcasts.append({
                "podcast_id": podcast_id,
                "title": title,
                "description": description,
                "duration_ms": duration_ms,
            })

    print(f"Fetched {len(podcasts)} podcasts from Spotify.")
    return podcasts

def save_embeddings_to_db():
    """Fetches podcast shows, classifies genres using ChatGPT, generates embeddings, and stores them in SQLite."""
    podcasts = fetch_podcasts()
    conn = get_db_connection()
    cursor = conn.cursor()

    for podcast in podcasts:
        cursor.execute("SELECT 1 FROM podcasts WHERE podcast_id = ?", (podcast["podcast_id"],))
        if cursor.fetchone():
            print(f"Skipping {podcast['title']} (already exists)")
            continue

        print(f"Classifying genre for {podcast['title']} using ChatGPT...")
        genre = classify_genre(podcast["title"], podcast["description"])

        print(f"Generating embedding for {podcast['title']}...")
        embedding = generate_embedding(f"{podcast['title']} {genre} {podcast['description']}")

        # Store in SQLite
        cursor.execute(
            """
            INSERT INTO podcasts (podcast_id, title, genre, description, embedding, duration_ms)
            VALUES (?, ?, ?, ?, ?, ?)
            """,
            (podcast["podcast_id"], podcast["title"], genre, podcast["description"], pickle.dumps(embedding),
             podcast["duration_ms"])
        )

    conn.commit()
    conn.close()
    print("All podcasts saved successfully!")

if __name__ == "__main__":
    save_embeddings_to_db()
