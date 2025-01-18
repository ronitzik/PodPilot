import os
import spotipy
from spotipy.oauth2 import SpotifyClientCredentials
from dotenv import load_dotenv

load_dotenv()

# Initialize Spotify API
sp = spotipy.Spotify(auth_manager=SpotifyClientCredentials(
    client_id=os.getenv("SPOTIFY_CLIENT_ID"),
    client_secret=os.getenv("SPOTIFY_CLIENT_SECRET")
))


def search_episodes_by_person(person_name: str, country="IL", limit=3):
    """Searches Spotify for podcast episodes where a specific person is mentioned in title or description."""
    try:
        # Search for episodes related to the person's name
        query = f"{person_name}"
        results = sp.search(q=query, type="episode", market=country, limit=limit)

        if "episodes" not in results or "items" not in results["episodes"]:
            return {"error": f"No episodes found for {person_name}"}

        episodes = results["episodes"]["items"]

        # Extract relevant details
        episode_list = []
        for episode in episodes:
            episode_list.append({
                "episode_name": episode.get("name", "Unknown"),
                "spotify_url": episode["external_urls"]["spotify"],
                "release_date": episode.get("release_date", "Unknown")
            })

        return {"episodes": episode_list}
    except Exception as e:
        return {"error": f"Failed to search episodes: {e}"}
