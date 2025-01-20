import yt_dlp
import os
import re
import requests
import time
from urllib.parse import quote
from dotenv import load_dotenv
import openai

load_dotenv()

# AssemblyAI API key
ASSEMBLYAI_API_KEY = os.getenv("ASSEMBLYAI_API_KEY")

# OpenAI API Key
openai.api_key = os.getenv("OPENAI_API_KEY")


def search_apple_podcast(podcast_name, episode_name):
    """
    Search for a podcast episode on Apple Podcasts and return the episode URL.
    """
    search_url = f"https://itunes.apple.com/search?term={quote(podcast_name)}&media=podcast"
    response = requests.get(search_url)
    response.raise_for_status()
    data = response.json()

    for result in data.get("results", []):
        if podcast_name.lower() in result.get("collectionName", "").lower():
            feed_url = result.get("feedUrl")
            if feed_url:
                return extract_episode_url(feed_url, episode_name)
    return None


def extract_episode_url(feed_url, episode_name):
    """
    Fetch the RSS feed and extract the episode URL.
    """
    response = requests.get(feed_url)
    response.raise_for_status()

    matches = re.findall(r'<item>.*?<title>(.*?)</title>.*?<enclosure url="(.*?)"', response.text, re.DOTALL)
    for title, episode_url in matches:
        if episode_name.lower() in title.lower():
            return episode_url
    return None


def download_podcast_episode(podcast_name, episode_name):
    """
    Download a podcast episode from Apple Podcasts given the podcast name and episode name.
    Returns the path to the downloaded file.
    """
    output_file = f"{podcast_name}_{episode_name}.mp3"

    if os.path.exists(output_file):
        print(f"File '{output_file}' already exists. Skipping download.")
        return os.path.abspath(output_file)

    episode_url = search_apple_podcast(podcast_name, episode_name)
    if not episode_url:
        raise ValueError(f"Episode '{episode_name}' of podcast '{podcast_name}' not found.")

    ydl_opts = {
        'format': 'bestaudio/best',
        'outtmpl': output_file,
        'postprocessors': [{
            'key': 'FFmpegExtractAudio',
            'preferredcodec': 'mp3',
            'preferredquality': '192',
        }],
    }

    with yt_dlp.YoutubeDL(ydl_opts) as ydl:
        ydl.download([episode_url])

    return os.path.abspath(output_file)


def upload_to_assemblyai(file_path):
    """
    Uploads the podcast episode to AssemblyAI for transcription.
    """
    headers = {'authorization': ASSEMBLYAI_API_KEY}
    with open(file_path, 'rb') as f:
        response = requests.post("https://api.assemblyai.com/v2/upload", headers=headers, files={'file': f})

    if response.status_code == 200:
        return response.json().get("upload_url")
    else:
        raise ValueError("❌ Failed to upload audio file to AssemblyAI.")


def transcribe_audio(file_url):
    """
    Sends an uploaded file URL to AssemblyAI for transcription.
    """
    headers = {'authorization': ASSEMBLYAI_API_KEY, 'content-type': 'application/json'}
    data = {'audio_url': file_url}

    response = requests.post("https://api.assemblyai.com/v2/transcript", json=data, headers=headers)

    if response.status_code == 200:
        transcript_id = response.json().get("id")
    else:
        raise ValueError("❌ Failed to initiate transcription.")

    # Wait for transcription to complete
    while True:
        status_response = requests.get(f"https://api.assemblyai.com/v2/transcript/{transcript_id}", headers=headers)
        status_json = status_response.json()

        if status_json.get("status") == "completed":
            return status_json.get("text")
        elif status_json.get("status") == "failed":
            raise ValueError("❌ Transcription failed.")

        time.sleep(10)


def summarize_text(transcript):
    """
    Uses OpenAI GPT to summarize the episode transcript into a structured, one-page summary.
    """
    prompt = f"""
    You are an expert in summarizing podcast episodes. 
    Your task is to create a structured and engaging one-page summary that captures the key themes, discussions, and insights of the episode.
    Ensure the summary is easy to understand, concise, and well-structured.

    **Episode Transcript:**
    {transcript}
    
    Answer as a plain-text, start you answer with "This episode is about..."
    """

    response = openai.ChatCompletion.create(
        model="gpt-4o-mini",
        messages=[{"role": "user", "content": prompt}]    )
    return response["choices"][0]["message"]["content"].strip()


def generate_episode_summary(podcast_name, episode_name):
    """
    Fetches, downloads (if not already downloaded), transcribes, and summarizes a podcast episode.
    """
    try:
        #  Download episode (if not already downloaded)
        file_path = download_podcast_episode(podcast_name, episode_name)

        #  Upload to AssemblyAI
        file_url = upload_to_assemblyai(file_path)

        #  Get transcript
        transcript = transcribe_audio(file_url)

        #  Generate summary
        summary = summarize_text(transcript)

        return {
            "podcast_name": podcast_name,
            "episode_name": episode_name,
            "summary": summary
        }

    except Exception as e:
        return {"error": str(e)}
