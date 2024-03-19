import spotipy
from spotipy.oauth2 import SpotifyClientCredentials
import os

# Set your Spotify API credentials
os.environ["SPOTIPY_CLIENT_ID"] = "YOUR_CLIENT_ID"
os.environ["SPOTIPY_CLIENT_SECRET"] = "YOUR_CLIENT_SECRET"


def get_related_artists(artist_name):
    """
    Queries Spotify for the artist by name, then fetches related artists.
    """
    # Authenticate with Spotify
    client_credentials_manager = SpotifyClientCredentials()
    sp = spotipy.Spotify(client_credentials_manager=client_credentials_manager)

    # Search for the artist to get their Spotify ID
    result = sp.search(q="artist:" + artist_name, type="artist")
    try:
        artist_id = result["artists"]["items"][0]["id"]
    except IndexError:
        print(f"Artist '{artist_name}' not found.")
        return

    # Fetch related artists using the artist's Spotify ID
    related_artists = sp.artist_related_artists(artist_id)

    # Print the names of related artists
    print(f"Artists related to {artist_name}:")
    for artist in related_artists["artists"]:
        print(artist["name"])
