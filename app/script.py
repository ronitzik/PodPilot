#script.py
import requests

# Base URL of the FastAPI server
BASE_URL = "http://127.0.0.1:8000"


def get_user_id():
    """Prompt the user to enter their user ID."""
    while True:
        try:
            user_id = int(input("Enter your user ID: "))
            return user_id
        except ValueError:
            print("Invalid input. Please enter a numeric user ID.")


def collect_user_preferences(user_id):
    """Interact with the user to collect their podcast preferences until at least 3 liked shows are found."""
    print("\n🎧 Let's learn your podcast preferences!")

    liked_podcasts = set()  # Using a set to prevent duplicate likes
    disliked_podcasts = set()
    user_responses = {}
    seen_podcasts = set()  # Track suggested podcasts to prevent repeats

    while len(liked_podcasts) < 3:
        # Fetch new podcast suggestions
        response = requests.post(f"{BASE_URL}/suggest_podcasts", json={"user_id": user_id})

        if response.status_code != 200:
            print("⚠️ Error fetching podcast suggestions:", response.json().get("detail"))
            return None

        suggested_podcasts = response.json().get("suggestions", [])

        # Filter out already suggested or disliked podcasts
        filtered_podcasts = [p for p in suggested_podcasts if p not in seen_podcasts and p not in disliked_podcasts]

        if not filtered_podcasts:
            print("\nNo more new podcasts to suggest, but you need to like at least 3. Trying again...")
            continue  # Ensures the loop continues

        for podcast_title in filtered_podcasts:
            seen_podcasts.add(podcast_title)  # Mark as suggested
            response = input(f"Do you like '{podcast_title}'? (yes/no): ").strip().lower()
            user_responses[podcast_title] = response

            if response == "yes":
                liked_podcasts.add(podcast_title)  # Store only titles
            else:
                disliked_podcasts.add(podcast_title)  # Prevent re-suggesting this title

            if len(liked_podcasts) >= 3:
                break  # Stop once we have 3 liked shows

    if len(liked_podcasts) < 3:
        print("⚠️ You need to like at least 3 podcasts. Please try again.")
        collect_user_preferences(user_id)
        return

    # Send user responses to the API
    response = requests.post(f"{BASE_URL}/process_feedback", json={"user_id": user_id, "responses": user_responses})

    if response.status_code == 200:
        print("\n✅ Preferences saved successfully!")
    else:
        print("⚠️ Error saving preferences:", response.json().get("detail"))


def generate_search_query(user_id):
    """Request a Google search query for related podcasts."""
    response = requests.post(f"{BASE_URL}/generate_search_query", json={"user_id": user_id})

    if response.status_code == 200:
        search_query = response.json()["search_query"]
        return search_query
    else:
        print("⚠️ Error generating search query:", response.json().get("detail"))
        return None


def search_related_podcasts(search_query, user_id):
    """Search for related podcasts based on the generated search query."""
    response = requests.post(f"{BASE_URL}/search_related_podcasts", json={"user_id": user_id, "search_query": search_query})

    if response.status_code == 200:
        related_podcasts = response.json().get("related_podcasts", [])
        if not related_podcasts:
            print("\nNo related podcasts found via Google Search, recommending only based on liked podcasts.")
        else:
            print(f"\nFound related podcasts via Google Search, Now I have more data for the recommendation.")
        return related_podcasts
    else:
        print("⚠️ Error searching related podcasts:", response.json().get("detail"))
        return []


def recommend_top_podcasts(user_id, related_podcasts):
    """Recommend top 3 podcasts based on user preferences."""

    response = requests.post(f"{BASE_URL}/recommend_podcasts", json={"user_id": user_id})

    if response.status_code == 200:
        recommendations = response.json().get("recommendations", [])

        if recommendations:
            print("\n🎙️ Based on the information I learned, I think you should try listening to:\n")
            for i, podcast in enumerate(recommendations, 1):
                podcast_name = podcast["name"]
                formatted_description = podcast["formatted_description"]
                print(f"🎧 {i}. {podcast_name}\n   {formatted_description}\n")
        else:
            print("\n⚠️ No relevant podcast recommendations found.")
    else:
        print("⚠️ Error fetching recommendations:", response.json().get("detail"))


def main():
    """Main workflow to get podcast recommendations."""
    print("🎧 Welcome to the Podcast Recommendation System!")
    user_id = get_user_id()

    # Step 1: Collect user preferences until at least 3 liked podcasts are found
    collect_user_preferences(user_id)

    # Step 2: Generate a Google search query for related podcasts
    search_query = generate_search_query(user_id)
    if search_query:
        # Step 3: Search for related podcasts
        related_podcasts = search_related_podcasts(search_query, user_id)

        # Step 4: Get final recommendations
        recommend_top_podcasts(user_id, related_podcasts)


if __name__ == "__main__":
    main()
