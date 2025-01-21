import os
import openai
from dotenv import load_dotenv
from pydub import AudioSegment
from pydub.utils import which
from summarization import  download_podcast_episode, transcribe_audio,upload_to_assemblyai
from pathlib import Path
from gtts import gTTS

load_dotenv()

openai.api_key = os.getenv("OPENAI_API_KEY")
assemblyai_api_key = os.getenv("ASSEMBLYAI_API_KEY")

ASSEMBLYAI_URL = "https://api.assemblyai.com/v2/transcript"

def extract_audio_segment(file_path, start_time, end_time, output_format="mp3"):
    """Extracts a specific time segment from the downloaded audio file."""
    try:
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"Error: The file '{file_path}' does not exist.")

        AudioSegment.converter = which("ffmpeg")
        if not AudioSegment.converter:
            raise EnvironmentError("ffmpeg not found. Ensure it's installed and added to PATH.")

        file_extension = os.path.splitext(file_path)[-1][1:].lower()
        if file_extension not in ["mp3", "wav", "ogg", "flac", "m4a"]:
            raise ValueError(f"Unsupported audio format: {file_extension}")

        audio = AudioSegment.from_mp3(file_path)

        def time_to_ms(time_str):
            parts = list(map(int, time_str.split(":")))
            return sum(x * 60 ** i * 1000 for i, x in enumerate(reversed(parts)))

        start_ms = time_to_ms(start_time)
        end_ms = time_to_ms(end_time)

        if start_ms < 0 or end_ms > len(audio) or start_ms >= end_ms:
            raise ValueError(f"Invalid time range: {start_time} - {end_time}")

        segment = audio[start_ms:end_ms]

        output_segment_path = f"segment_{os.path.basename(file_path).split('.')[0]}.{output_format}"
        segment.export(output_segment_path, format=output_format)
        print(f"Audio segment saved at: {output_segment_path}")
        return output_segment_path
    except Exception as e:
        raise RuntimeError(f"Failed to extract audio segment: {str(e)}")



def summarize_text(transcript):
    """Summarizes the transcribed text using OpenAI."""
    prompt = f"""
    You are an expert in summarizing part of podcast episodes. I will send you the beginning of the episode and I need you
    to Summarize the following podcast segment concisely:
    {transcript}
    Provide a structured, engaging, one-paragraph summary. Start with "The last time you listen to the episode it was about.."
    """
    response = openai.ChatCompletion.create(
        model="gpt-4o-mini",
        messages=[{"role": "user", "content": prompt}]
    )
    return response["choices"][0]["message"]["content"].strip()


def generate_voice_summary(podcast_name, episode_name, start_time, end_time):
    """Generates a voice summary for the given episode segment."""
    episode_path = download_podcast_episode(podcast_name, episode_name)
    if isinstance(episode_path, dict) and "error" in episode_path:
        return episode_path

    segment_path = extract_audio_segment(episode_path, start_time, end_time)
    file_url = upload_to_assemblyai(segment_path)
    transcript = transcribe_audio(file_url)
    if not transcript:
        return {"error": "Failed to transcribe the audio segment."}

    summary_text = summarize_text(transcript)

    # Define the output path for the TTS file
    speech_file_path = Path(__file__).parent / f"summary_{os.path.basename(segment_path)}.mp3"

    try:
        # Generate TTS using gTTS
        tts = gTTS(text=summary_text, lang="en")
        tts.save(speech_file_path)
        print(f"Voice summary saved at: {speech_file_path}")
        return str(speech_file_path)
    except Exception as e:
        print(f"Failed to generate TTS audio: {e}")
        return {"error": "Failed to generate TTS audio."}

