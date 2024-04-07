import requests
import base64
import sys
from googleapiclient.discovery import build
import os

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
    if len(sys.argv) != 2:
        print("Usage: python get_related_artists.py <spotify_artist_id>")
        sys.exit(1)

    spotify_artist_id = sys.argv[1]
    try:
        # Authenticate with Spotify and get related artists
        spotify_token = get_spotify_token()
        related_artists = get_related_artists(spotify_token, spotify_artist_id)

        # Set up YouTube API client
        api_key = os.getenv("YOUTUBE_API_KEY")
        if not api_key:
            raise ValueError(
                "YouTube API key must be set as an environment variable 'YOUTUBE_API_KEY'."
            )
        youtube = build(
            "youtube", "v3", developerKey="AIzaSyC7sUZ5hCMaswsmr_NcUo2awGDcbYAYZdw"
        )

        # Fetch YouTube channel IDs for each related artist
        for artist in related_artists["artists"]:
            artist_name = artist["name"]
            youtube_channel_id, channel_title = get_youtube_channel_id(
                artist_name, youtube
            )
            if youtube_channel_id:
                print(
                    f"Artist: {artist_name}, YouTube Channel: {channel_title}, ID: {youtube_channel_id}"
                )
            else:
                print(
                    f"Artist: {artist_name} does not have a YouTube channel that could be found."
                )

    except requests.exceptions.HTTPError as http_err:
        print(f"HTTP error occurred: {http_err}")
    except Exception as err:
        print(f"An error occurred: {err}")
