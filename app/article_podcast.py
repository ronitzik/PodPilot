import os
import requests
from podcastfy.client import generate_podcast
from bs4 import BeautifulSoup
from dotenv import load_dotenv

load_dotenv()

def fetch_article_content(article_url):
    """
    Fetches the full text content from a given article URL using BeautifulSoup.
    """
    try:
        headers = {"User-Agent": "Mozilla/5.0"}
        response = requests.get(article_url, headers=headers)

        if response.status_code != 200:
            return None, f"Failed to fetch article. HTTP Status: {response.status_code}"

        soup = BeautifulSoup(response.text, "html.parser")

        # Extract main text content from the article
        paragraphs = soup.find_all("p")
        article_text = "\n".join([p.get_text() for p in paragraphs])

        if not article_text:
            return None, "Could not extract meaningful content from the article."

        return article_text, None

    except Exception as e:
        return None, f"Error fetching article: {e}"

def generate_podcast_from_article(article_url):
    """
    Fetches an article, extracts its text, and generates a podcast episode using Podcastfy.
    """
    article_text, error = fetch_article_content(article_url)
    if error:
        return {"error": error}

    try:
        # Generate podcast using Podcastfy
        audio_file = generate_podcast(text=article_text)

        if not audio_file:
            return {"error": "Failed to generate podcast audio."}

        return {"podcast_audio_url": audio_file}

    except Exception as e:
        return {"error": f"Error generating podcast: {e}"}
