import os
import requests
import numpy as np
import torch
from dotenv import load_dotenv
from transformers import AutoTokenizer, AutoModel
from sqlalchemy import create_engine, Column, Integer, String, Text, LargeBinary, JSON
from sqlalchemy.orm import sessionmaker, declarative_base

# Load environment variables
load_dotenv()
X_USER_ID = os.getenv("X_USER_ID")
X_API_KEY = os.getenv("X_API_KEY")


# Set up the ORM base class
Base = declarative_base()


# Database 1: Store Podcasts and Embeddings
class Podcast(Base):
    __tablename__ = 'podcasts'

    id = Column(Integer, primary_key=True, autoincrement=True)
    podcast_id = Column(String, unique=True, index=True)  # From Taddy API
    title = Column(String, index=True)
    description = Column(Text)
    genre = Column(String)
    embedding = Column(LargeBinary)  # Store embeddings as binary data


# Database 2: Store User Preferences
class UserPreference(Base):
    __tablename__ = 'user_preferences'

    id = Column(Integer, primary_key=True, autoincrement=True)
    user_id = Column(Integer, unique=True, index=True)  # Ensure one record per user
    liked_podcasts = Column(JSON)


# Initialize Databases
podcast_engine = create_engine("sqlite:///podcasts.db")
user_pref_engine = create_engine("sqlite:///user_preferences.db")

Base.metadata.create_all(podcast_engine)
Base.metadata.create_all(user_pref_engine)

PodcastSession = sessionmaker(bind=podcast_engine)
UserPrefSession = sessionmaker(bind=user_pref_engine)

podcast_session = PodcastSession()
user_pref_session = UserPrefSession()

# Load Pre-trained Embedding Model
tokenizer = AutoTokenizer.from_pretrained("sentence-transformers/all-MiniLM-L6-v2")
model = AutoModel.from_pretrained("sentence-transformers/all-MiniLM-L6-v2")


def generate_embedding(text: str):
    """Generates an embedding for the given text using Hugging Face Transformers."""
    if not text.strip():
        return np.zeros(384)

    inputs = tokenizer(text, return_tensors="pt", truncation=True, padding=True)
    with torch.no_grad():
        outputs = model(**inputs)
    embedding = outputs.last_hidden_state.mean(dim=1).squeeze().numpy()
    return embedding


def fetch_top_podcasts():
    """Fetches the top 200 podcasts in Israel from the Taddy API."""
    API_URL = "https://api.taddy.org"

    headers = {
        "Content-Type": "application/json",
        "X-USER-ID": X_USER_ID,
        "X-API-KEY": X_API_KEY
    }

    all_podcasts = []
    page = 1
    max_pages = 8

    while len(all_podcasts) < 200 and page <= max_pages:
        graphql_query = {
            "query": f"""
            {{
                getTopChartsByCountry(
                    taddyType: PODCASTSERIES, 
                    country: ISRAEL, 
                    limitPerPage: 25, 
                    page: {page}
                ) {{
                    topChartsId
                    podcastSeries {{
                        uuid
                        name
                        description  
                        genres
                    }}
                }}
            }}
            """
        }

        try:
            response = requests.post(API_URL, headers=headers, json=graphql_query)
            response.raise_for_status()

            data = response.json()
            # Extract podcast series data
            podcasts = data.get("data", {}).get("getTopChartsByCountry", {}).get("podcastSeries", [])

            if not podcasts:
                print(f"⚠️ No more podcasts returned on page {page}. Stopping pagination.")
                break

            all_podcasts.extend(podcasts)
            page += 1  # Move to the next page

        except requests.exceptions.RequestException as e:
            print(f" Error fetching data from Taddy API on page {page}: {e}")
            break

    print(f" Fetched {len(all_podcasts)} podcasts from Taddy API.")
    return all_podcasts[:300]  # Ensure we return exactly 300 results


def save_podcasts_to_db():
    """Fetches top 200 podcasts, generates embeddings, and stores them in SQLite."""
    podcasts = fetch_top_podcasts()

    if not podcasts:
        print("No podcasts fetched.")
        return

    podcast_objects = []

    for podcast in podcasts:
        podcast_id = podcast.get("uuid")  # Use `uuid` instead of `id`
        title = podcast.get("name", "Unknown Title")  # Ensure title is not None
        if title is None:  # Additional safety check
            title = "Unknown Title"

        description = podcast.get("description", "No description available")
        # Extract the genre
        genre_data = podcast.get("genres", [])
        if isinstance(genre_data, list) and genre_data:
            genre = genre_data[0]
        else:
            genre = "Unknown Genre"

            # Generate embedding for the podcast
        full_text = f"{title}. {description}. Genre: {genre}"
        embedding = generate_embedding(full_text)
        embedding_binary = np.array(embedding).tobytes()

        # Store podcast details in the database
        podcast_objects.append(Podcast(
            podcast_id=podcast_id,
            title=title,
            description=description,
            genre=genre,
            embedding=embedding_binary,
        ))

    podcast_session.bulk_save_objects(podcast_objects)  # Faster than adding one by one
    podcast_session.commit()
    print(f" Saved {len(podcasts)} podcasts with embeddings to the database.")


def save_user_preference(user_id, podcast_id, podcast_name, liked):
    """Saves user preference (liked podcast) into user_preferences.db."""
    preference = UserPreference(
        user_id=user_id,
        podcast_id=podcast_id,
        podcast_name=podcast_name,
        liked=liked
    )
    user_pref_session.add(preference)
    user_pref_session.commit()
    print(f"✅ User {user_id} liked podcast {podcast_name}.")


if __name__ == "__main__":
    save_podcasts_to_db()
