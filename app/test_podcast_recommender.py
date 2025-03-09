import unittest
from unittest.mock import patch, MagicMock
import numpy as np
import torch
from recommendation import (
    collect_user_preferences,
    generate_google_search_prompt,
    search_related_podcasts,
    recommend_top_podcasts
)
from save_embeddings import (
    generate_embedding,
    fetch_top_podcasts,
    optimize_description,
    save_podcasts_to_db
)

class TestPodcastRecommender(unittest.TestCase):

    @patch("recommendation.podcast_session")
    @patch("recommendation.user_pref_session")
    def test_collect_user_preferences(self, mock_user_pref_session, mock_podcast_session):
        """Test collecting user preferences"""
        mock_podcast_session.query().all.return_value = [
            MagicMock(title="Tech Trends", genre="Technology"),
            MagicMock(title="Health Talks", genre="Health"),
            MagicMock(title="AI Insights", genre="Technology")
        ]

        mock_user_pref_session.query().filter_by().first.return_value = None
        mock_user_pref_session.add = MagicMock()
        mock_user_pref_session.commit = MagicMock()

        with patch("builtins.input", side_effect=["yes", "no", "yes"]):
            collect_user_preferences(user_id=123)

        mock_user_pref_session.commit.assert_called_once()

    @patch("openai.ChatCompletion.create")
    @patch("recommendation.user_pref_session")
    def test_generate_google_search_prompt(self, mock_user_pref_session, mock_openai):
        """Test generating a search query using OpenAI"""
        mock_user_pref_session.query().filter_by().first.return_value = MagicMock(liked_podcasts="Tech Trends, AI Insights")

        mock_openai.return_value = {
            "choices": [{"message": {"content": "site:spotify.com AI podcasts Israel"}}]
        }

        query = generate_google_search_prompt(user_id=123)
        self.assertEqual(query, "site:spotify.com AI podcasts Israel")

    @patch("requests.get")
    def test_search_related_podcasts(self, mock_requests):
        """Test searching related podcasts using Google API"""
        mock_requests.return_value.json.return_value = {
            "items": [
                {"title": "AI Podcast | Podcast on Spotify"},
                {"title": "Tech Deep Dive | Podcast on Spotify"}
            ]
        }

        with patch("recommendation.podcast_session.query") as mock_query:
            mock_query.return_value.all.return_value = [
                MagicMock(title="AI Podcast"),
                MagicMock(title="Tech Deep Dive")
            ]

            result = search_related_podcasts(query="site:spotify.com AI podcasts", user_id=123)
            self.assertIn("AI Podcast", result)
            self.assertIn("Tech Deep Dive", result)

    @patch("recommendation.get_podcast_embedding")
    @patch("recommendation.user_pref_session")
    @patch("recommendation.podcast_session")
    def test_recommend_top_podcasts(self, mock_podcast_session, mock_user_pref_session, mock_get_embedding):
        """Test recommending top podcasts based on embeddings"""
        mock_user_pref_session.query().filter_by().first.return_value = MagicMock(
            liked_podcasts="Tech Trends, AI Insights"
        )

        # Mock embeddings
        mock_get_embedding.side_effect = lambda title: np.array([0.1, 0.2, 0.3]) if title else None

        mock_podcast_session.query().all.return_value = [
            MagicMock(title="AI Podcast"),
            MagicMock(title="Tech Deep Dive"),
            MagicMock(title="Health Talks")
        ]

        recommendations = recommend_top_podcasts(user_id=123, related_podcasts=["AI Podcast", "Tech Deep Dive"])
        self.assertEqual(len(recommendations), 3)
        self.assertIn("AI Podcast", recommendations)
        self.assertIn("Tech Deep Dive", recommendations)

class TestPodcastEmbedding(unittest.TestCase):

    @patch("save_embeddings.tokenizer")
    @patch("save_embeddings.model")
    def test_generate_embedding(self, mock_model, mock_tokenizer):
        """Test if generate_embedding returns a valid 384-dimensional embedding."""
        mock_tokenizer.return_value = {
            "input_ids": torch.tensor([[101, 102]]),
            "attention_mask": torch.tensor([[1, 1]])
        }
        mock_model.return_value = (torch.rand(1, 2, 384),)

        embedding = generate_embedding("Sample text")
        self.assertEqual(len(embedding), 384)
        self.assertTrue(isinstance(embedding, np.ndarray))

    @patch("save_embeddings.requests.post")
    def test_fetch_top_podcasts(self, mock_post):
        """Test if fetch_top_podcasts correctly retrieves and processes podcast data with pagination handled."""

        mock_response = MagicMock()
        mock_response.json.side_effect = [
            {
                "data": {
                    "getTopChartsByCountry": {
                        "podcastSeries": [
                            {
                                "uuid": "123",
                                "name": "Tech Podcast",
                                "description": "Tech insights",
                                "genres": ["Technology"]
                            },
                            {
                                "uuid": "456",
                                "name": "AI Podcast",
                                "description": "AI innovations",
                                "genres": ["AI"]
                            }
                        ]
                    }
                }
            },
            {
                "data": {
                    "getTopChartsByCountry": {
                        "podcastSeries": []
                    }
                }
            }
        ]
        mock_post.return_value = mock_response

        podcasts = fetch_top_podcasts()
        expected_count = 2
        self.assertEqual(len(podcasts), expected_count)

        # Check if extracted podcast data is correct
        self.assertEqual(podcasts[0]["uuid"], "123")
        self.assertEqual(podcasts[0]["name"], "Tech Podcast")
        self.assertEqual(podcasts[0]["description"], "Tech insights")
        self.assertEqual(podcasts[0]["genres"], ["Technology"])

    @patch("save_embeddings.fetch_top_podcasts")
    @patch("save_embeddings.generate_embedding")
    @patch("save_embeddings.podcast_session")
    @patch("save_embeddings.optimize_description")
    def test_save_podcasts_to_db(self, mock_optimize, mock_podcast_session, mock_generate_embedding, mock_fetch):
        """Test saving podcasts, ensuring the function fetches, processes, and stores data correctly."""
        mock_fetch.return_value = [
            {
                "uuid": "123",
                "name": "AI Podcast",
                "description": "Discussing AI innovations.",
                "genres": ["Technology"]
            }
        ]
        mock_optimize.return_value = "AI innovations discussion."
        mock_generate_embedding.return_value = np.random.rand(384)

        mock_session = MagicMock()
        mock_podcast_session.bulk_save_objects = mock_session.bulk_save_objects
        mock_podcast_session.commit = mock_session.commit
        save_podcasts_to_db()
        mock_session.bulk_save_objects.assert_called()
        mock_session.commit.assert_called()

if __name__ == "__main__":
    unittest.main()
