import json
import random
from save_embeddings import Podcast, UserPreference, user_pref_session, podcast_session

def get_random_podcasts():
    """Fetch 10 random podcasts from different genres."""
    podcasts_by_genre = {}

    # Get all podcasts
    all_podcasts = podcast_session.query(Podcast).all()

    # Organize by genre
    for podcast in all_podcasts:
        if podcast.genre not in podcasts_by_genre:
            podcasts_by_genre[podcast.genre] = []
        podcasts_by_genre[podcast.genre].append(podcast)

    # Select 10 random podcasts from different genres
    selected_podcasts = []
    genres_selected = set()

    while len(selected_podcasts) < 10 and len(genres_selected) < len(podcasts_by_genre):
        genre = random.choice(list(podcasts_by_genre.keys()))
        if genre not in genres_selected and podcasts_by_genre[genre]:
            selected_podcasts.append(random.choice(podcasts_by_genre[genre]))
            genres_selected.add(genre)

    return selected_podcasts


def collect_user_preferences(user_id):
    """Ask the user for preferences and save all liked podcasts in a single record."""
    print("\n🎧 Let's learn your podcast preferences!")
    podcasts = get_random_podcasts()

    liked_podcasts = []

    for podcast in podcasts:
        response = input(f"Do you like '{podcast.title}'? (yes/no): ").strip().lower()
        if response == "yes":
            liked_podcasts.append({"podcast_id": podcast.podcast_id, "title": podcast.title})

    # Convert liked podcasts to JSON format
    liked_podcasts_json = json.dumps(liked_podcasts)

    # Check if user already has a record
    user_pref = user_pref_session.query(UserPreference).filter_by(user_id=user_id).first()

    if user_pref:
        # Update existing record
        user_pref.liked_podcasts = liked_podcasts_json
    else:
        # Create a new record
        user_pref = UserPreference(
            user_id=user_id,
            liked_podcasts=liked_podcasts_json
        )
        user_pref_session.add(user_pref)

    user_pref_session.commit()
    print(f"\n✅ Preferences saved for user {user_id}.")


if __name__ == "__main__":
    user_id = int(input("Enter your user ID: "))
    collect_user_preferences(user_id)
