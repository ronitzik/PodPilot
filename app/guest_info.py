import os
import requests
import openai
import spotipy
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
    """
    Fetches episode details from Spotify API, including its description.
    Returns the episode description and error message (if any).
    """
    try:
        results = sp.search(q=podcast_name, type="show", limit=1)
        if not results["shows"]["items"]:
            return None, "Podcast not found."

        podcast_id = results["shows"]["items"][0]["id"]

        episodes = sp.show_episodes(podcast_id, limit=50)["items"]
        for episode in episodes:
            if episode_name.lower() in episode["name"].lower():
                description = episode.get("description", "No description available")
                return description, None

        return None, "Episode not found."

    except Exception as e:
        return None, f"Error fetching episode: {e}"


def extract_guest_name(episode_name, description):
    """
    Uses OpenAI to extract guest name(s) from episode title & description.
    Returns first name and last name.
    """
    prompt = f"""
    Extract the guest's full name from the following podcast episode details.
    Respond ONLY in English with the full name in 'First Last' format, no extra text.

    Episode Title: {episode_name}
    Description: {description}
    """

    try:
        response = openai.ChatCompletion.create(
            model="gpt-4o-mini",
            messages=[{"role": "user", "content": prompt}]
        )
        guest_name = response["choices"][0]["message"]["content"].strip()

        # Extract first and last name
        name_parts = guest_name.split()
        if len(name_parts) < 2:
            return None, None  # Not a valid full name

        first_name = name_parts[0]
        last_name = name_parts[-1]  # Get last name (handles cases like "Elon Reeve Musk")

        return first_name, last_name

    except Exception as e:
        print(f"Error extracting guest name: {e}")
        return None, None


def generate_linkedin_url(first_name, last_name):
    """
    Generates the LinkedIn public search URL for the guest.
    """
    if not first_name or not last_name:
        return None, "Could not extract guest name."

    return f"https://www.linkedin.com/pub/dir/{first_name}/{last_name}", None


def find_guest_info(data):
    """
    Fetches the guest name from episode details and returns their LinkedIn search URL.
    """
    podcast_name = data.get("podcast_name")
    episode_name = data.get("episode_name")

    if not podcast_name or not episode_name:
        return {"error": "Podcast name and episode name are required."}

    description, error = fetch_episode_details(podcast_name, episode_name)
    if error:
        return {"error": error}

    first_name, last_name = extract_guest_name(episode_name, description)
    if not first_name or not last_name:
        return {"error": "Could not determine guest name from episode details."}

    linkedin_url, linkedin_error = generate_linkedin_url(first_name, last_name)
    if linkedin_error:
        return {"error": linkedin_error}

    return {
        "guest_name": f"{first_name} {last_name}",
        "linkedin_url": linkedin_url
    }
