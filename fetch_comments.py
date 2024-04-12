import sys
from googleapiclient.discovery import build
from googleapiclient.errors import HttpError
import csv
import datetime
import operator
import json
import os
import isodate

# Hardcoded weights for calculating engagement score
WEIGHTS = {
    "likeCount": 0.2,
    "commentCount": 0.2,
    "viewCount": 0.2,
    "engagementRate": 0.15,
    "timelessness": 0.1,
    "viewToLikeRatio": 0.05,
    "relevance": 0.15,
    "subscriberEngagement": 0.1,
}


def calculate_engagement_score(video_stats):
    engagement_score = (
        WEIGHTS["likeCount"] * video_stats["likeCount"]
        + WEIGHTS["commentCount"] * video_stats["commentCount"]
        + WEIGHTS["viewCount"] * video_stats["viewCount"]
        + WEIGHTS["engagementRate"] * video_stats["engagementRate"]
        + WEIGHTS["timelessness"] * video_stats["timelessness"]
        + WEIGHTS["viewToLikeRatio"] * video_stats["viewToLikeRatio"]
        + WEIGHTS["relevance"] * video_stats["relevance"]
        + WEIGHTS["subscriberEngagement"] * video_stats["subscriberEngagement"]
    )
    print(f"Engagement Score: {engagement_score}")
    return engagement_score


def get_video_ids(youtube, channel_id, max_videos, artist_name, next_page_token=None):
    print(f"Fetching videos for channel ID: {channel_id}")
    video_ids = []
    video_details = {}
    max_values = {
        "likeCount": 0,
        "commentCount": 0,
        "viewCount": 0,
        "timelessness": 0,
        "viewToLikeRatio": 0,
        "relevance": 0,
        "subscriberEngagement": 0,
    }

    # Fetch video IDs
    request = youtube.search().list(
        part="snippet",
        channelId=channel_id,
        maxResults=max_videos,  # Adjust according to your needs
        order="date",  # Fetching the most recent videos
        type="video",
        pageToken=next_page_token,
    )
    response = request.execute()

    for item in response.get("items", []):
        video_ids.append(item["id"]["videoId"])

    print(f"Found {len(video_ids)} videos.")

    # Fetch statistics and content details for each video
    stats_request = youtube.videos().list(
        part="statistics,snippet,contentDetails", id=",".join(video_ids)
    )
    stats_response = stats_request.execute()

    for item in stats_response.get("items", []):
        video_id = item["id"]
        stats = item["statistics"]
        snippet = item["snippet"]
        content_details = item["contentDetails"]

        # Filter out YouTube Shorts based on video duration
        duration = isodate.parse_duration(content_details["duration"])
        if duration.total_seconds() < 60:  # Adjust the threshold as needed
            continue

        like_count = int(stats.get("likeCount", 0))
        comment_count = int(stats.get("commentCount", 0))
        view_count = int(stats.get("viewCount", 0))
        published_at = datetime.datetime.strptime(
            snippet["publishedAt"], "%Y-%m-%dT%H:%M:%SZ"
        ).replace(tzinfo=datetime.timezone.utc)
        time_since_upload = (
            datetime.datetime.now(datetime.timezone.utc) - published_at
        ).total_seconds()

        # Update max_values if necessary
        max_values["likeCount"] = max(max_values["likeCount"], like_count)
        max_values["commentCount"] = max(max_values["commentCount"], comment_count)
        max_values["viewCount"] = max(max_values["viewCount"], view_count)
        max_values["timelessness"] = max(max_values["timelessness"], time_since_upload)
        max_values["viewToLikeRatio"] = max(
            max_values["viewToLikeRatio"], view_count / (like_count + 1)
        )

        # Calculate relevance score based on title and description matching
        title_match = artist_name.lower() in snippet["title"].lower()
        description_match = artist_name.lower() in snippet["description"].lower()
        relevance_score = int(title_match) + int(description_match)
        max_values["relevance"] = max(max_values["relevance"], relevance_score)

        # Calculate subscriber engagement score
        subscriber_count = int(stats.get("subscriberCount", 0))
        subscriber_engagement = (like_count + comment_count) / (subscriber_count + 1)
        max_values["subscriberEngagement"] = max(
            max_values["subscriberEngagement"], subscriber_engagement
        )

        # Calculate normalized values
        normalized_likes = like_count / max_values["likeCount"]
        normalized_comments = comment_count / max_values["commentCount"]
        normalized_views = view_count / max_values["viewCount"]
        normalized_timelessness = time_since_upload / max_values["timelessness"]
        normalized_view_to_like_ratio = (view_count / (like_count + 1)) / max_values[
            "viewToLikeRatio"
        ]
        normalized_relevance = relevance_score / max_values["relevance"]
        normalized_subscriber_engagement = (
            subscriber_engagement / max_values["subscriberEngagement"]
        )
        engagement_rate = (
            (like_count + comment_count) / time_since_upload * 3600
        )  # Per hour

        print(f"Video ID: {video_id}")
        print(f"Like Count: {like_count}, Normalized: {normalized_likes}")
        print(f"Comment Count: {comment_count}, Normalized: {normalized_comments}")
        print(f"View Count: {view_count}, Normalized: {normalized_views}")
        print(
            f"Timelessness: {time_since_upload}, Normalized: {normalized_timelessness}"
        )
        print(
            f"View-to-Like Ratio: {view_count / (like_count + 1)}, Normalized: {normalized_view_to_like_ratio}"
        )
        print(f"Relevance Score: {relevance_score}, Normalized: {normalized_relevance}")
        print(
            f"Subscriber Engagement: {subscriber_engagement}, Normalized: {normalized_subscriber_engagement}"
        )
        print(f"Engagement Rate: {engagement_rate}")
        print()

        # Store engagement score for each video
        video_stats = {
            "likeCount": normalized_likes,
            "commentCount": normalized_comments,
            "viewCount": normalized_views,
            "engagementRate": engagement_rate,
            "timelessness": normalized_timelessness,
            "viewToLikeRatio": normalized_view_to_like_ratio,
            "relevance": normalized_relevance,
            "subscriberEngagement": normalized_subscriber_engagement,
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

    # Get the next page token for pagination
    next_page_token = response.get("nextPageToken")

    return top_video_ids, next_page_token


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

            for item in response.get("items", []):
                comment = item["snippet"]["topLevelComment"]["snippet"]["textDisplay"]
                comments.append(comment)

            next_page_token = response.get("nextPageToken")
            if not next_page_token:
                break
    except HttpError as e:
        print(f"An HTTP error occurred while fetching comments: {e}")
        return comments  # It returns the comments fetched so far

    print(f"Fetched {len(comments)} comments for video ID: {video_id}")
    return comments


def fetch_and_save_comments(api_key, channel_id, artist_name, max_videos=50):
    youtube = build("youtube", "v3", developerKey=api_key)

    all_comments = []
    next_page_token = None

    while len(all_comments) < 12000:
        batch_video_ids, next_page_token = get_video_ids(
            youtube, channel_id, 50, artist_name, next_page_token
        )

        for video_id in batch_video_ids:
            comments = get_comments(youtube, video_id)
            all_comments.extend(comments)

            if len(all_comments) >= 12000:
                break

        if not next_page_token:
            break

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

    # Check if the 'items' key exists in the response
    if "items" in response and response["items"]:
        channel_resource = response["items"][0]
    else:
        channel_resource = None
        print(f"No channel resource found for channel ID: {channel_id}")

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

    all_comments_count = fetch_and_save_comments(api_key, channel_id, artist_name)

    # Create a directory named 'data' if it doesn't exist
    if not os.path.exists("data"):
        os.makedirs("data")

    filename = f"data/{artist_name}_comments.csv"
    print(f"Comments for {artist_name} saved to {filename}")

    # Get the channel resource and save it
    channel_resource = get_channel_resource(youtube, channel_id)
    if channel_resource:
        save_channel_resource_to_file(channel_resource, artist_name)

    return all_comments_count


if __name__ == "__main__":
    if len(sys.argv) < 3:
        print("Usage: python fetch_comments.py 'artist_name' 'channel_id'")
        sys.exit(1)

    artist_name = sys.argv[1]
    channel_id = sys.argv[2]

    # Execute the fetch process
    all_comments_count = fetch(artist_name, channel_id)
    print(f"Total comments fetched and saved for {artist_name}: {all_comments_count}")
