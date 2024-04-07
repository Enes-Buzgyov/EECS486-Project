import sys
from googleapiclient.discovery import build
from googleapiclient.errors import HttpError
import csv
import datetime
import operator
import json
import os


# Hardcoded weights for calculating engagement score
WEIGHTS = {
    "likeCount": 0.3,
    "commentCount": 0.3,
    "viewCount": 0.3,
    "engagementRate": 0.1,  # Giving slightly less weight for the engagement rate
}


def calculate_engagement_score(video_stats):
    engagement_score = (
        WEIGHTS["likeCount"] * video_stats["likeCount"]
        + WEIGHTS["commentCount"] * video_stats["commentCount"]
        + WEIGHTS["viewCount"] * video_stats["viewCount"]
        + WEIGHTS["engagementRate"] * video_stats["engagementRate"]
    )
    return engagement_score


def get_video_ids(youtube, channel_id, max_videos):
    print(f"Fetching videos for channel ID: {channel_id}")
    video_ids = []
    video_details = {}
    max_values = {"likeCount": 0, "commentCount": 0, "viewCount": 0}

    # Fetch video IDs
    request = youtube.search().list(
        part="snippet",
        channelId=channel_id,
        maxResults=max_videos,  # Adjust according to your needs
        order="date",  # Fetching the most recent videos
        type="video",
    )
    response = request.execute()

    for item in response.get("items", []):
        video_ids.append(item["id"]["videoId"])

    # Fetch statistics for each video
    stats_request = youtube.videos().list(
        part="statistics,snippet", id=",".join(video_ids)
    )
    stats_response = stats_request.execute()

    for item in stats_response.get("items", []):
        video_id = item["id"]
        stats = item["statistics"]
        snippet = item["snippet"]

        like_count = int(stats.get("likeCount", 0))
        comment_count = int(stats.get("commentCount", 0))
        view_count = int(stats.get("viewCount", 0))
        published_at = datetime.datetime.strptime(
            snippet["publishedAt"], "%Y-%m-%dT%H:%M:%SZ"
        )
        time_since_upload = (datetime.datetime.utcnow() - published_at).total_seconds()

        # Update max_values if necessary
        max_values["likeCount"] = max(max_values["likeCount"], like_count)
        max_values["commentCount"] = max(max_values["commentCount"], comment_count)
        max_values["viewCount"] = max(max_values["viewCount"], view_count)

        # Calculate normalized values
        normalized_likes = like_count / max_values["likeCount"]
        normalized_comments = comment_count / max_values["commentCount"]
        normalized_views = view_count / max_values["viewCount"]
        engagement_rate = (
            (like_count + comment_count) / time_since_upload * 3600
        )  # Per hour

        # Store engagement score for each video
        video_stats = {
            "likeCount": normalized_likes,
            "commentCount": normalized_comments,
            "viewCount": normalized_views,
            "engagementRate": engagement_rate,
        }
        video_details[video_id] = calculate_engagement_score(video_stats)

    # Sort videos by engagement score in descending order
    sorted_videos = sorted(
        video_details.items(), key=operator.itemgetter(1), reverse=True
    )

    # Get the top video IDs based on engagement score
    top_video_ids = [video_id for video_id, _ in sorted_videos[:10]]

    print("Top Video Links:")
    for video_id in top_video_ids:
        print(f"https://www.youtube.com/watch?v={video_id}")

    return top_video_ids


def save_comments_to_csv(comments, filename):
    # Ensure the output directory exists
    os.makedirs(os.path.dirname(filename), exist_ok=True)
    with open(filename, mode="w", newline="", encoding="utf-8") as file:
        writer = csv.writer(file)
        for comment in comments:
            writer.writerow([comment])


def get_comments(youtube, video_id):
    comments = []
    next_page_token = None

    try:
        while True:
            request = youtube.commentThreads().list(
                part="snippet",
                videoId=video_id,
                maxResults=100,  # Maximum allowed by YouTube Data API v3
                pageToken=next_page_token,
                textFormat="plainText",
            )
            response = request.execute()
            comments.extend(
                [
                    item["snippet"]["topLevelComment"]["snippet"]["textDisplay"]
                    for item in response.get("items", [])
                ]
            )

            next_page_token = response.get("nextPageToken")
            if not next_page_token:
                break
    except HttpError as e:
        print(f"An HTTP error occurred: {e}")
        return comments  # It returns the comments fetched so far

    return comments


def fetch_and_save_comments(api_key, channel_id, artist_name, max_videos=50):
    youtube = build("youtube", "v3", developerKey=api_key)

    video_ids = get_video_ids(youtube, channel_id, 50)

    all_comments = []
    for video_id in video_ids:
        comments = get_comments(youtube, video_id)
        all_comments.extend(comments)

    filename = f"data/{artist_name}_comments.csv"
    save_comments_to_csv(all_comments, filename)
    print(f"Comments for {artist_name} saved to {filename}")
    return len(all_comments)


def get_channel_resource(youtube, channel_id):
    # Fetch the channel resource using the YouTube Data API
    request = youtube.channels().list(
        part="snippet,contentDetails,statistics", id=channel_id
    )
    response = request.execute()

    # Assuming there is only one channel matching the ID
    channel_resource = response["items"][0] if response["items"] else None

    return channel_resource


def save_channel_resource_to_file(channel_resource, artist_name):
    # Create the RAG knowledge Base directory if it doesn't exist
    rag_knowledge_base_dir = "RAG knowledge Base"
    os.makedirs(rag_knowledge_base_dir, exist_ok=True)

    # Define the filename based on the artist's name
    filename = os.path.join(
        rag_knowledge_base_dir, f"{artist_name}_channel_resource.json"
    )

    # Write the channel resource to a JSON file
    with open(filename, "w", encoding="utf-8") as file:
        json.dump(channel_resource, file, ensure_ascii=False, indent=4)

    print(f"Channel resource for {artist_name} saved to {filename}")


def fetch(artist_name, channel_id):
    # Best practice: Retrieve the API key from an environment variable for security
    api_key = os.getenv("YOUTUBE_API_KEY")
    if not api_key:
        raise ValueError(
            "YouTube API key must be set as an environment variable 'YOUTUBE_API_KEY'."
        )

    youtube = build("youtube", "v3", developerKey=api_key)

    # Assuming max_videos is either predefined or based on your specific requirements
    max_videos = 5
    video_ids = get_video_ids(youtube, channel_id, max_videos)

    all_comments = []
    for video_id in video_ids:
        comments = get_comments(youtube, video_id)
        all_comments.extend(comments)

    # Create a directory named 'data' if it doesn't exist
    if not os.path.exists("data"):
        os.makedirs("data")

    filename = f"data/{artist_name}_comments.csv"
    save_comments_to_csv(all_comments, filename)
    print(f"Comments for {artist_name} saved to {filename}")

    # Get the channel resource and save it
    channel_resource = get_channel_resource(youtube, channel_id)
    if channel_resource:
        save_channel_resource_to_file(channel_resource, artist_name)
    else:
        print(f"No channel resource found for channel ID: {channel_id}")


if __name__ == "__main__":
    if len(sys.argv) < 3:
        print("Usage: python fetch_comments.py 'artist_name' 'channel_id'")
        sys.exit(1)

    artist_name = sys.argv[1]
    channel_id = sys.argv[2]

    # Execute the fetch process
    all_comments_count = fetch(artist_name, channel_id)
    print(f"Total comments fetched and saved for {artist_name}: {all_comments_count}")
