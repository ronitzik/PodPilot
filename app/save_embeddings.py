import os
import requests
import numpy as np
import torch
from dotenv import load_dotenv
from transformers import AutoTokenizer, AutoModel
from sqlalchemy import create_engine, Column, Integer, String, Text, LargeBinary
from sqlalchemy.orm import sessionmaker, declarative_base
import torch.nn.functional as F
import openai
from langdetect import detect

# Load environment variables
load_dotenv()
X_USER_ID = os.getenv("X_USER_ID")
X_API_KEY = os.getenv("X_API_KEY")
openai.api_key = os.getenv("OPENAI_API_KEY")

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
    language = Column(String)
    embedding = Column(LargeBinary)  # Store embeddings as binary data

# Database 2: Store User Preferences
class UserPreference(Base):
    __tablename__ = 'user_preferences'

    id = Column(Integer, primary_key=True, autoincrement=True)
    user_id = Column(Integer, unique=True, index=True)  # Ensure one record per user
    liked_podcasts = Column(Text)
    disliked_podcasts = Column(Text)
    preferred_language = Column(String)

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

def mean_pooling(model_output, attention_mask):
    """Applies mean pooling to obtain a single embedding for the sentence."""
    token_embeddings = model_output[0]
    input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
    return torch.sum(token_embeddings * input_mask_expanded, 1) / torch.clamp(input_mask_expanded.sum(1), min=1e-9)

def generate_embedding(text: str):
    """Generates a 384-dimensional embedding for the given text using SBERT."""
    if not text.strip():
        return np.zeros(384)

    encoded_input = tokenizer(text, padding=True, truncation=True, return_tensors="pt")

    with torch.no_grad():
        model_output = model(**encoded_input)

    sentence_embedding = mean_pooling(model_output, encoded_input['attention_mask'])
    sentence_embedding = F.normalize(sentence_embedding, p=2, dim=1)

    return sentence_embedding.squeeze().numpy()

def detect_language(text):
    """Detects the language of the given text."""
    try:
        return detect(text)
    except:
        return "unknown"

def fetch_top_podcasts():
    """Fetches the top podcasts in Israel from the Taddy API."""
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
            podcasts = data.get("data", {}).get("getTopChartsByCountry", {}).get("podcastSeries", [])

            if not podcasts:
                print(f"No more podcasts returned on page {page}. Stopping pagination.")
                break

            all_podcasts.extend(podcasts)
            page += 1

        except requests.exceptions.RequestException as e:
            print(f"Error fetching data from Taddy API on page {page}: {e}")
            break

    print(f"Fetched {len(all_podcasts)} podcasts from Taddy API.")
    return all_podcasts[:300]

def optimize_description(description):
    """Uses OpenAI to rewrite a Hebrew description in an embedding-efficient way."""
    if not description.strip():
        return "No description available."

    prompt = f"""
    Rewrite in English the following podcast description in the most concise and embedding-optimized way, preserving only the essential meaning:
    "{description}"
    """

    response = openai.ChatCompletion.create(
        model="gpt-4o-mini",
        messages=[{"role": "user", "content": prompt}]
    )

    return response["choices"][0]["message"]["content"].strip()

def save_podcasts_to_db():
    """Fetches top podcasts, optimizes their descriptions, detects language, generates embeddings, and stores them in SQLite."""
    podcasts = fetch_top_podcasts()

    if not podcasts:
        print("No podcasts fetched.")
        return

    podcast_objects = []

    for podcast in podcasts:
        podcast_id = podcast.get("uuid")
        title = podcast.get("name", "Unknown Title") or "Unknown Title"
        description = (podcast.get("description") or "").strip()
        genre_data = podcast.get("genres", [])

        genre = genre_data[0] if isinstance(genre_data, list) and genre_data else "Unknown Genre"

        optimized_description = optimize_description(description)

        # Detect language
        detected_language = detect_language(title + " " + description)

        # Structured input for embedding (includes language)
        embedding_text = f"Podcast Title: {title}. Genre: {genre}. Language: {detected_language}. About: {optimized_description}"

        # Generate embedding
        embedding = generate_embedding(embedding_text)
        embedding_binary = np.array(embedding).tobytes()

        podcast_objects.append(Podcast(
            podcast_id=podcast_id,
            title=title,
            description=optimized_description,
            genre=genre,
            language=detected_language,  # Store detected language
            embedding=embedding_binary,
        ))

    # Bulk insert into database
    podcast_session.bulk_save_objects(podcast_objects)
    podcast_session.commit()
    print(f" Saved {len(podcasts)} podcasts with optimized embeddings to the database.")


if __name__ == "__main__":
    save_podcasts_to_db()
