import requests
import base64
import sys
from googleapiclient.discovery import build
import os
from clean_comments import clean_comments
from fetch_comments import fetch
from baseline import perform_comment_analysis
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import networkx as nx
from sklearn.cluster import KMeans
from fetch_comments import normalize_artist_name
from compare_fanbase import load_and_compare_all_artists

# Constants for Spotify API
CLIENT_ID = "1f61ff30c31e405995b1c8834a188996"  # Replace with your Spotify client ID
CLIENT_SECRET = (
    "9786890a1e714b95a6007da6f9479196"  # Replace with your Spotify client secret
)
SPOTIFY_AUTH_URL = "https://accounts.spotify.com/api/token"
SPOTIFY_API_URL = "https://api.spotify.com/v1/artists"


def get_spotify_token():
    """Get an access token from the Spotify API."""
    headers = {
        "Authorization": "Basic "
        + base64.b64encode(f"{CLIENT_ID}:{CLIENT_SECRET}".encode()).decode(),
    }
    payload = {"grant_type": "client_credentials"}
    response = requests.post(SPOTIFY_AUTH_URL, headers=headers, data=payload)
    response.raise_for_status()  # Will raise an error for a status code != 200
    return response.json()["access_token"]


def get_related_artists(spotify_token, artist_id):
    """Get related artists from the Spotify API given an artist ID."""
    headers = {
        "Authorization": f"Bearer {spotify_token}",
    }
    url = f"{SPOTIFY_API_URL}/{artist_id}/related-artists"
    response = requests.get(url, headers=headers)
    response.raise_for_status()
    return response.json()


def get_youtube_channel_id(artist_name, youtube):
    # Search for channels
    request = (
        youtube.search()
        .list(q=artist_name, part="snippet", maxResults=1, type="channel")
        .execute()
    )

    # Extract channel ID
    results = request.get("items", [])
    if results:
        return results[0]["id"]["channelId"], results[0]["snippet"]["title"]
    return None, None


if __name__ == "__main__":
    if len(sys.argv) != 4:
        print(
            "Usage: python main.py <artist_name> <youtube_channel_id> <spotify_artist_id>"
        )
        sys.exit(1)

    artist_name = sys.argv[1]
    main_artist_name = artist_name
    youtube_channel_id = sys.argv[2]
    spotify_artist_id = sys.argv[3]

    try:
        # Process main artist
        all_comments_count = fetch(artist_name, youtube_channel_id)
        print(f"Fetched {all_comments_count} comments for {artist_name}.")
        clean_comments(artist_name)
        perform_comment_analysis(artist_name)

        # Authenticate with Spotify and get related artists
        spotify_token = get_spotify_token()
        related_artists = get_related_artists(spotify_token, spotify_artist_id)
        top_artists = related_artists["artists"][:5]

        # Set up YouTube API client
        api_key = os.getenv("YOUTUBE_API_KEY")
        if not api_key:
            raise ValueError(
                "YouTube API key must be set as an environment variable 'YOUTUBE_API_KEY'."
            )
        youtube = build(
            "youtube", "v3", developerKey="AIzaSyDE2_NCHxi5RnD-vmlquqeiaxHBZnsRyh8"
        )

        print("Top 5 Related Artists:")
        for idx, artist in enumerate(top_artists, start=1):
            artist_name = artist["name"]
            spotify_artist_id = artist["id"]
            print(
                f"{idx}. {artist_name} (Spotify Artist ID: {spotify_artist_id})"
            )  # Get top 5 related artists

        # Process related artists
        related_artist_names = []
        for artist in top_artists:
            related_artist_name = normalize_artist_name(artist["name"])
            related_artist_names.append(related_artist_name)
            print(f"Processing related artist: {related_artist_name}")
            youtube_channel_id, channel_title = get_youtube_channel_id(
                artist_name, youtube
            )

            all_comments_count = fetch(related_artist_name, youtube_channel_id)
            print(f"Fetched {all_comments_count} comments for {related_artist_name}.")

            # Clean comments
            clean_comments(related_artist_name)

            # Analysis on comments
            perform_comment_analysis(related_artist_name)

        # similarity_ranking, graph, comments_with_clusters = compare_artists(
        #    normalize_artist_name(main_artist_name), related_artist_names
        # )

        similarity_ranking, graph, comments_with_clusters = (
            load_and_compare_all_artists()
        )

        print("Similarity Ranking:")
        for artist, score in similarity_ranking:
            print(f"{artist}: {score}")

        print("\nGraph Edges:")
        for u, v, attr in graph.edges(data=True):
            print(f"{u} -- {v}: {attr['weight']}")

        print("\nClustered Comments:")
        print(comments_with_clusters)

    except requests.exceptions.HTTPError as http_err:
        print(f"HTTP error occurred: {http_err}")
    except Exception as err:
        print(f"An error occurred: {err}")
