from fetch_comments import fetch_and_save_comments
from clean_comments import clean_comments
from profile_fanbase import profile_fanbase
import os

API_KEY = "Insert API Key"


def main():
    # Dictionary of artist names and their YouTube channel IDs
    artists = {
        "Ken Carson": "UCtjO8cix9XxwOcAv6nww-TQ",
        "Playboi Carti": "UC652oRUvX1onwrrZ8ADJRPw",
        "Destroy Lonely": "UCylvCifeaY7gi1zjhRfpcMg",
        "Yeat": "UCV4UK9LNNLViFP4qZA_Wmfw",
        "Lil Uzi Vert": "UCqwxMqUcL-XC3D9-fTP93Mg",
        "Lancey Foux": "UCPVCEEf4oyLZt5ZE3yPiwMg",
        "Travis Scott": "UCtxdfwb9wfkoGocVUAJ-Bmg",
        "Pierre Bourne": "UCN27za3wzItmyRV2O3N91vw",
        "Pink Floyd": "UCY2qt3dw2TQJxvBrDiYGHdQ",
    }

    for artist_name, channel_id in artists.items():

        cleaned_csv = f"output/cleaned_{artist_name}_comments.csv"

        # Fetch and save comments if they haven't been fetched yet
        print(f"Fetching comments for {artist_name}")
        fetch_and_save_comments(API_KEY, channel_id, artist_name)

        # Clean comments if they haven't been cleaned yet
        print(f"Cleaning comments for {artist_name}")
        clean_comments(f"data/{artist_name}_comments.csv", cleaned_csv)


if __name__ == "__main__":
    main()
    print("Completed data processing for all artists.")
