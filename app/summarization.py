import openai
import os
import spotipy
import requests
from spotipy.oauth2 import SpotifyClientCredentials
from dotenv import load_dotenv

load_dotenv()

# Initialize OpenAI API
openai.api_key = os.getenv("OPENAI_API_KEY")

# Initialize Spotify API
sp = spotipy.Spotify(auth_manager=SpotifyClientCredentials(
    client_id=os.getenv("SPOTIFY_CLIENT_ID"),
    client_secret=os.getenv("SPOTIFY_CLIENT_SECRET")
))

def fetch_episode_details(podcast_name, episode_name):
    """Fetches episode details including description and audio preview URL from Spotify API."""
    try:
        # Search for the podcast show
        results = sp.search(q=podcast_name, type="show", limit=1)
        if not results["shows"]["items"]:
            return None, None, "Podcast not found."

        podcast_id = results["shows"]["items"][0]["id"]

        # Fetch episodes of the show
        episodes = sp.show_episodes(podcast_id, limit=50)["items"]
        for episode in episodes:
            if episode_name.lower() in episode["name"].lower():
                description = episode.get("description", "No description available")
                audio_url = episode.get("audio_preview_url", None)
                return description, audio_url, None

        return None, None, "Episode not found."

    except Exception as e:
        return None, None, f"Error fetching episode: {e}"


def transcribe_audio(audio_url):
    """Downloads and transcribes the episode's audio preview using OpenAI Whisper API."""
    if not audio_url:
        return None

    try:
        response = requests.get(audio_url, stream=True)
        if response.status_code != 200:
            return None

        # Save audio file temporarily
        audio_path = "temp_audio.mp3"
        with open(audio_path, "wb") as f:
            f.write(response.content)

        # Transcribe using OpenAI Whisper API
        with open(audio_path, "rb") as audio_file:
            transcription = openai.Audio.transcribe(
                model="whisper-1",
                file=audio_file,
                response_format="text"
            )

        os.remove(audio_path)  # Clean up temporary file
        return transcription.strip()

    except Exception as e:
        print(f"⚠️ Error transcribing audio: {e}")
        return None


def summarize_text(description, transcript):
    """Summarizes the episode using OpenAI, combining text and transcript if available."""
    combined_text = f"""
    Summarize the following podcast episode into a clear and structured one-page summary. 
    The summary should capture the main themes, key discussions, and any emotional or factual highlights. 
    Keep the summary engaging and concise, ensuring it reflects the core of the episode. 
    If an audio preview transcript is available, incorporate relevant insights from it. 

Episode Description:
{description}

Audio Preview Transcription:
{transcript if transcript else 'No audio preview available.'}

Provide a structured, plain-text summary, ensuring readability without bullet points or special formatting.
"""
    try:
        response = openai.ChatCompletion.create(
            model="gpt-4o-mini",
            messages=[{"role": "user", "content": combined_text}]
        )
        return response["choices"][0]["message"]["content"].strip()

    except Exception as e:
        return f"Error generating summary: {e}"


def generate_episode_summary(podcast_name, episode_name):
    """Fetches the episode, transcribes audio, and generates a one-page summary."""
    description, audio_url, error = fetch_episode_details(podcast_name, episode_name)

    if error:
        return {"error": error}

    transcript = transcribe_audio(audio_url)
    summary = summarize_text(description, transcript)

    return {
        "podcast_name": podcast_name,
        "episode_name": episode_name,
        "summary": summary
    }
